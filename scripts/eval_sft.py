# eval_sft.py
import unsloth
import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

from unsloth import FastLanguageModel, FastModel
from transformers import set_seed, AutoModelForSequenceClassification

from src.configs.config_manager import ConfigManager
from src.utils.data_utils import prepare_dataset
from src.utils.huggingface_utils import init_hub_env
# from src.utils.seed_utils import set_all_seeds

CURRENT_TEST_TYPE = "sft"

# 라벨 고정 매핑
LABEL2ID = {"appropriate": 0, "inappropriate": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_gold_from_dev(dev_path: str) -> dict:
    """
    dev 파일에서 각 utterance id -> 정답 라벨(문자열) 매핑을 생성
    형식 가정:
    [
      {
        "id": "...",
        "input": {"document_id": "...", "utterance": [ ... ]},
        "output": [
          {"id": "<utt_id>", "label": "inappropriate", ...}, ...
        ]
      }, ...
    ]
    """
    with open(dev_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold = {}
    for doc in data:
        outputs = doc.get("output", [])
        for o in outputs:
            uid = o["id"]
            gold[uid] = o["label"].strip().lower()
    assert len(gold) > 0
    return gold


def save_confusion_matrix(y_true, y_pred, save_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["appropriate(0)", "inappropriate(1)"],
        yticklabels=["appropriate(0)", "inappropriate(1)"],
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true, y_score, save_path: str) -> float:
    """
    y_score: positive class(=1, 'inappropriate')의 확률 점수
    반환: ROC AUC
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (positive: 'inappropriate')")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return roc_auc


def save_diagnostics_csv(rows: list[dict], save_path: str) -> None:
    """
    rows: 각 샘플에 대해 기록할 정보의 리스트
    필드 예시:
      document_id, utterance_id, true_label, pred_label, correct, score_inappropriate(optional), score_appropriate(optional)
    """
    cols = [
        "document_id",
        "utterance_id",
        "true_label",
        "pred_label",
        "correct",
        "score_appropriate",
        "score_inappropriate",
    ]
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            line = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, str):
                    v = v.replace("\n", " ").replace("\r", " ").replace(",", " ")
                line.append(str(v))
            f.write(",".join(line) + "\n")


def parse_cot_answer(answer: str) -> dict:
    """CoT 답변을 파싱하여 think와 answer 부분을 분리"""
    result = {}

    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, answer, re.DOTALL)

    if think_match:
        think_content = think_match.group(1)
        think_content = think_content.strip()
        think_content = re.sub(r"\n+", " ", think_content)
        result["think"] = think_content
    else:
        result["think"] = ""

    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, answer, re.DOTALL)

    if answer_match:
        answer_content = answer_match.group(1)
        answer_content = answer_content.strip()
        answer_content = re.sub(r"\n+", " ", answer_content)
        result["answer"] = answer_content
    else:
        clean_answer = re.sub(think_pattern, "", answer, flags=re.DOTALL).strip()
        clean_answer = re.sub(r"\n+", " ", clean_answer)
        result["answer"] = clean_answer

    return result


def convert_answer_to_label(answer: str, config_manager: ConfigManager) -> str:
    """모델 답변을 적절한 label로 변환"""
    answer = answer.strip().lower()

    if config_manager.system.only_decode:
        if "부적절" in answer or "inappropriate" in answer:
            return "inappropriate"
        elif "적절" in answer or "appropriate" in answer:
            return "appropriate"
        else:
            print(
                f"Warning: Could not parse answer '{answer}', defaulting to 'appropriate'"
            )
            return "appropriate"
    else:
        # 분류 모델의 경우: 외부에서 logits로 결정 → 여기서는 문자열 그대로
        return answer


def init_config_manager_for_test(save_dir: str = "configs") -> ConfigManager:
    cm = ConfigManager()
    config_dir = os.path.join(save_dir, "configs")
    cm.load_all_configs(config_dir=config_dir)

    adapter_dir = os.path.join(save_dir, "lora_adapter")
    test_result_dir = os.path.join(save_dir, "test_result")
    os.makedirs(test_result_dir, exist_ok=True)
    print(f"Test results will be saved to: {test_result_dir}")

    cm.update_config(
        "system",
        {
            "save_dir": save_dir,
            "adapter_dir": adapter_dir,
            "test_result_dir": test_result_dir,
        },
    )
    cm.print_all_configs()
    return cm


def main(cm: ConfigManager):
    # 모델 로드 인자
    add_args = {}
    if cm.model.num_classes != -1:
        add_args["num_labels"] = cm.model.num_classes

    if not cm.system.only_decode:
        add_args["auto_model"] = AutoModelForSequenceClassification
        fast_model = FastModel
        print("Using FastModel for classification training.")
    else:
        fast_model = FastLanguageModel
        print("Using FastLanguageModel for generation training.")

    model_path = cm.system.adapter_dir if cm.model.full_finetune else cm.model.model_id
    model, tokenizer = fast_model.from_pretrained(
        model_name=model_path,
        max_seq_length=cm.model.max_seq_length,
        dtype=cm.model.dtype if cm.system.only_decode else None,
        load_in_4bit=cm.model.load_in_4bit,
        load_in_8bit=cm.model.load_in_8bit,
        full_finetuning=cm.model.full_finetune,
        trust_remote_code=True,
        **add_args,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token

    model = fast_model.for_inference(model)
    if not cm.model.full_finetune and cm.system.only_decode:
        model.load_adapter(cm.system.adapter_dir)

    # dev 데이터 사용(정답 포함)
    original_data = {}
    dev_file_path = os.path.join(cm.system.data_raw_dir, "dev.json")
    with open(dev_file_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
        for item in dev_data:
            original_data[item["id"]] = item["input"]

    # 정답 매핑: utterance_id -> gold label(str)
    gold_map = load_gold_from_dev(dev_file_path)

    # 평가 수집기
    y_true = []
    y_pred = []
    y_score = []  # 분류모델일 때 부적절(=1) 확률
    diag_rows = []

    # 데이터셋 로드
    test_dataset = prepare_dataset(
        config_manager=cm,
        tokenizer=tokenizer,
        task_type=CURRENT_TEST_TYPE,
        is_train=False,
        eval_split="dev",   # ★ dev로 지정 (정답 포함)
    )
    print(f"Dev dataset size: {len(test_dataset)}")

    document_results = defaultdict(lambda: {"id": "", "input": {}, "output": []})

    debug_count = 0
    for sample in tqdm(test_dataset, desc="Evaluating(dev)", unit="sample"):
        document_id = sample["document_id"]
        utterance_id = sample["utterance_id"]
        utterance_idx = sample["utterance_idx"]

        # 데이터 정합성 강제
        assert utterance_id in gold_map

        gold_label_str = gold_map[utterance_id]
        gold_label_id = LABEL2ID[gold_label_str]

        if cm.system.only_decode:
            # 생성형: ROC 불가(확률 점수 없음)
            inputs = sample["input_ids"].unsqueeze(0).to(model.device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)

            outputs = model.generate(
                inputs,
                max_new_tokens=cm.model.max_new_tokens,
                do_sample=cm.model.do_sample,
                attention_mask=attention_mask,
                use_cache=cm.system.use_cache,
            )
            answer = tokenizer.decode(
                outputs[0][inputs.shape[-1] :], skip_special_tokens=True
            )

            if answer.startswith("답변: "):
                answer = answer[4:]
            elif answer.startswith("답변:"):
                answer = answer[3:]
            if "#" in answer:
                answer = answer.split("#")[0].strip()

            if cm.system.is_cot:
                parsed_output = parse_cot_answer(answer)
                final_answer = parsed_output["answer"]
            else:
                final_answer = answer
                parsed_output = {"think": ""}

            predicted_label_str = convert_answer_to_label(final_answer, cm)
            predicted_class = LABEL2ID[predicted_label_str]

            score_inappr = ""
            score_appr = ""

        else:
            # 분류모델: 소프트맥스 확률로 ROC 가능
            inputs = sample["input_ids"].unsqueeze(0).to(model.device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)

            with torch.no_grad():
                outputs = model(inputs, attention_mask=attention_mask)
                logits = outputs.logits
                logits_float = logits.float()
                probs = torch.softmax(logits_float, dim=-1)[0].cpu().numpy()
                predicted_class = int(np.argmax(probs))
                score_appr = float(probs[0])
                score_inappr = float(probs[1])

                if debug_count < 5:
                    print(f"\nSample {debug_count + 1}:")
                    print(f"  Document ID: {document_id}")
                    print(f"  Utterance ID: {utterance_id}")
                    print(
                        f"  Logits: [{float(logits_float[0,0]):.4f}, {float(logits_float[0,1]):.4f}]"
                    )
                    print(
                        f"  Predicted class: {predicted_class}"
                    )
                    print(
                        f"  Softmax probs: [적절: {score_appr:.4f}, 부적절: {score_inappr:.4f}]"
                    )
                    debug_count += 1

            predicted_label_str = ID2LABEL[predicted_class]

        # 누적
        y_true.append(gold_label_id)
        y_pred.append(predicted_class)
        if not cm.system.only_decode:
            y_score.append(score_inappr)

        if not document_results[document_id]["id"]:
            document_results[document_id]["id"] = document_id
            document_results[document_id]["input"] = original_data.get(document_id, {})

        output_item = {
            "id": utterance_id,
            "label": predicted_label_str,
            "utterance_idx": utterance_idx,
        }
        if cm.system.only_decode and cm.system.is_cot:
            output_item["think"] = parsed_output.get("think", "")

        document_results[document_id]["output"].append(output_item)

        diag_rows.append(
            {
                "document_id": document_id,
                "utterance_id": utterance_id,
                "true_label": gold_label_str,
                "pred_label": predicted_label_str,
                "correct": int(predicted_class == gold_label_id),
                "score_appropriate": score_appr if not cm.system.only_decode else "",
                "score_inappropriate": score_inappr if not cm.system.only_decode else "",
            }
        )

    # 문서별 결과 정리(JSON)
    final_results = []
    for document_id, doc_data in document_results.items():
        doc_data["output"].sort(key=lambda x: x["utterance_idx"])
        for output in doc_data["output"]:
            del output["utterance_idx"]
        final_results.append(
            {"id": doc_data["id"], "input": doc_data["input"], "output": doc_data["output"]}
        )

    # 고정 파일명으로 저장
    base_name = "dev_eval"
    out_dir = cm.system.test_result_dir

    output_json_path = os.path.join(out_dir, f"{base_name}.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    # 정확도/혼동행렬/ROC/진단CSV 저장
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    acc = accuracy_score(y_true_arr, y_pred_arr)
    print(f"\n[DEV] Accuracy: {acc:.6f}")

    cm_png = os.path.join(out_dir, f"{base_name}_confusion_matrix.png")
    save_confusion_matrix(y_true_arr, y_pred_arr, cm_png)
    print(f"Confusion matrix saved to: {cm_png}")

    if not cm.system.only_decode:
        y_score_arr = np.array(y_score, dtype=float)
        roc_png = os.path.join(out_dir, f"{base_name}_roc.png")
        auc_val = save_roc_curve(y_true_arr, y_score_arr, roc_png)
        print(f"ROC curve saved to: {roc_png} (AUC={auc_val:.6f})")
    else:
        print("ROC curve skipped: generation-only setting has no calibrated probability scores.")


    diag_csv = os.path.join(out_dir, f"{base_name}_diagnostics.csv")
    save_diagnostics_csv(diag_rows, diag_csv)
    print(f"Diagnostics CSV saved to: {diag_csv}")

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SFT Model on dev.json")
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Experiment save dir that contains configs/"
    )
    args = parser.parse_args()

    config_manager = init_config_manager_for_test(save_dir=args.save_dir)
    config_manager.update_config(CURRENT_TEST_TYPE, {"seed": config_manager.system.seed})

    init_hub_env(config_manager.system.hf_token)
    set_seed(config_manager.system.seed)
    # set_all_seeds(config_manager.system.seed, deterministic=config_manager.system.deterministic)

    main(config_manager)
