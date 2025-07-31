import os, json
from tqdm.auto import tqdm
from argparse import ArgumentParser

from transformers import set_seed
from src.utils.seed_utils import set_all_seeds
from src.configs.config_manager import ConfigManager
from src.rag.retriever.base_retriever import BaseRetriever
from src.utils.huggingface_utils import init_hub_env
from src.rag.retriever_factory import build_retriever


# def augment_dataset_with_rag(
#     input_path,
#     output_path,
#     retriever: BaseRetriever,
#     top_k=5,
#     context_field="retrieved_context"
# ):
#     with open(input_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     for example in tqdm(data, desc=f"Augmenting {os.path.basename(input_path)}"):
#         question = example["input"]["question"]
#         docs = retriever.retrieve(question, top_k=top_k)
#         example["input"][context_field] = " ".join(docs)

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)


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

    questions = [ex["input"]["question"] for ex in data]

    for i in tqdm(range(0, len(questions), batch_size), desc=f"Augmenting {os.path.basename(input_path)}"):
        batch_questions = questions[i:i+batch_size]
        batch_results = retriever.retrieve_batch(batch_questions, top_k=top_k)

        for j, docs in enumerate(batch_results):
            data[i + j]["input"][context_field] = " ".join(docs)

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
        input_path = os.path.join(config_manager.system.data_raw_dir, f"{split}.json") # 원본 데이터
        output_path = os.path.join(rag_cfg.output_dir, f"{split}.json")                # 저장되는 데이터

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
