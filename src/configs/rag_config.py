from dataclasses import field, dataclass

@dataclass
class RAGConfig:
    source_files: list[str] = field(default_factory=lambda: ["data/rag/corpus.pdf"])
    style: str = "kowiki"
    output_dir: str = "data/rag_results"
    index_dir: str = "data/rag_results/index"
    chunk_size: int = 256
    chunk_overlap: int = 10
    min_last_chunk_ratio: float = 0.2
    model_id: str = "nlpai-lab/KURE-v1"
    top_k: int = 5
    retriever_type: str = "dense"
    hybrid_alpha: int|float = 0.5
    context_text: str = "[관련 정보]"
    context_field: str = "retrieved_context"
    index_name: str = "index.faiss"
    corpus_name: str = "corpus.json"
    batch_size: int = 32
    use_rag: bool = False
