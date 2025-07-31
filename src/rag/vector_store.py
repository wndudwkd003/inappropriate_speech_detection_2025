import os, faiss, json

def load_faiss_index(
    index_dir: str,
    index_name: str = "index.faiss",
    corpus_name: str = "corpus.json",
) -> tuple[faiss.Index, list[str]]:
    index_path = os.path.join(index_dir, index_name)
    corpus_path = os.path.join(index_dir, corpus_name)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index 파일이 존재하지 않습니다: {index_path}")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus 파일이 존재하지 않습니다: {corpus_path}")

    index = faiss.read_index(index_path)

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    return index, corpus
