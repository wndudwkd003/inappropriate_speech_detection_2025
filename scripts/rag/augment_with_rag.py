
import os, json, re
from tqdm.auto import tqdm
from argparse import ArgumentParser
from transformers import set_seed
from src.utils.seed_utils import set_all_seeds
from src.configs.config_manager import ConfigManager
from src.rag.retriever.base_retriever import BaseRetriever
from src.utils.huggingface_utils import init_hub_env
from src.rag.retriever_factory import build_retriever

def parse_multiple_choice_options(question):
    """
    선다형 문제에서 보기들을 파싱합니다.
    Returns: (question_text, options_list) or (None, None) if not multiple choice
    """
    # 패턴: 숫자. 내용 형태로 된 선택지들을 찾음
    pattern = r'(\d+\.\s*[^0-9]+?)(?=\s*\d+\.|$)'
    matches = re.findall(pattern, question)

    if len(matches) >= 2:  # 최소 2개 이상의 선택지가 있어야 선다형으로 판단
        # 질문 부분과 선택지 부분을 분리
        first_option_start = question.find(matches[0])
        question_text = question[:first_option_start].strip()

        # 각 선택지에서 번호 제거하고 내용만 추출
        options = []
        for match in matches:
            # "1. 손이 크다" -> "손이 크다"
            option_text = re.sub(r'^\d+\.\s*', '', match).strip()
            options.append(option_text)

        return question_text, options

    return None, None

def augment_dataset_with_rag(
    input_path,
    output_path,
    retriever: BaseRetriever,
    top_k=5,
    context_field="retrieved_context",
    batch_size=32
):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 선다형과 일반형 문제를 분리하여 처리
    multiple_choice_indices = []
    regular_indices = []

    for i, example in enumerate(data):
        question = example["input"]["question"]
        question_text, options = parse_multiple_choice_options(question)

        if options and len(options) >= 2:
            multiple_choice_indices.append(i)
        else:
            regular_indices.append(i)

    print(f"Multiple choice questions: {len(multiple_choice_indices)}")
    print(f"Regular questions: {len(regular_indices)}")

    # 1. 선다형 문제 처리 - 각 보기별로 개별 검색
    if multiple_choice_indices:
        print("Processing multiple choice questions...")
        for idx in tqdm(multiple_choice_indices, desc="Augmenting multiple choice"):
            example = data[idx]
            question = example["input"]["question"]
            question_text, options = parse_multiple_choice_options(question)

            # 각 보기별로 검색 (top_k=1로 제한)
            retrieved_contexts = []
            for option in options:
                docs = retriever.retrieve(option, top_k=1)
                if docs:
                    retrieved_contexts.extend(docs)

            # 검색된 컨텍스트들을 결합
            example["input"][context_field] = " ".join(retrieved_contexts)

    # 2. 일반 문제 처리 - 배치로 전체 질문 검색
    if regular_indices:
        print("Processing regular questions...")
        regular_questions = [data[i]["input"]["question"] for i in regular_indices]

        for i in tqdm(range(0, len(regular_questions), batch_size), desc="Augmenting regular questions"):
            batch_questions = regular_questions[i:i+batch_size]
            batch_results = retriever.retrieve_batch(batch_questions, top_k=top_k)

            for j, docs in enumerate(batch_results):
                original_idx = regular_indices[i + j]
                data[original_idx]["input"][context_field] = " ".join(docs)

    # 결과 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Augmented data saved to {output_path}")

def main(config_manager: ConfigManager):
    rag_cfg = config_manager.rag

    # 이미 구축된 인덱스를 활용해 retriever 준비
    retriever = build_retriever(rag_cfg)
    print(f"Retriever batch size: {retriever.batch_size}")

    # 각 데이터셋 split을 증강
    os.makedirs(rag_cfg.output_dir, exist_ok=True)

    for split in ["train", "dev", "test"]:
        input_path = os.path.join(config_manager.system.data_raw_dir, f"{split}.json")
        output_path = os.path.join(rag_cfg.output_dir, f"{split}.json")

        if not os.path.exists(input_path):
            print(f"[!] Skipping: {input_path} (not found)")
            continue

        augment_dataset_with_rag(
            input_path=input_path,
            output_path=output_path,
            retriever=retriever,
            top_k=rag_cfg.top_k,
            context_field=rag_cfg.context_field,
            batch_size=retriever.batch_size
        )

    print("RAG augmentation completed successfully.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Run RAG augmentation using prebuilt index")
    parser.add_argument("--config", type=str, default="configs", help="Path to the configuration directory")
    args = parser.parse_args()

    config_manager = ConfigManager()
    config_manager.load_all_configs(config_dir=args.config)
    config_manager.print_all_configs()

    init_hub_env(config_manager.system.hf_token)
    set_seed(config_manager.system.seed)
    set_all_seeds(config_manager.system.seed, deterministic=config_manager.system.deterministic)

    main(config_manager)
