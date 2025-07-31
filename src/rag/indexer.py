import faiss
import json
import numpy as np
import os


def build_faiss_index(
    embeddings: np.ndarray,
    output_dir: str,
    index_name: str = "index.faiss"
):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, index_name))


# def save_corpus(
#     corpus: list[str],
#     output_dir: str,
#     corpus_name: str = "corpus.json"
# ):
#     os.makedirs(output_dir, exist_ok=True)
#     with open(os.path.join(output_dir, corpus_name), "w", encoding="utf-8") as f:
#         json.dump(corpus, f, ensure_ascii=False, indent=2)



def save_corpus(
    chunks: list[str],
    metadata: list[dict],
    output_dir: str,
    corpus_name: str = "corpus.json"
):
    """청크와 메타데이터를 함께 저장"""

    os.makedirs(output_dir, exist_ok=True)

    # 청크와 title을 함께 저장
    corpus_data = []
    for chunk_text, meta in zip(chunks, metadata):
        corpus_data.append({
            'text': chunk_text,
            'title': meta['title'],
            'chunk_idx': meta.get('chunk_idx', 0)
        })

    with open(os.path.join(output_dir, corpus_name), 'w', encoding='utf-8') as f:
        json.dump(corpus_data, f, ensure_ascii=False, indent=2)
