from src.rag.retriever.dense_retriever import DenseRetriever
from src.rag.retriever.bm25_retriever import BM25Retriever
from src.rag.retriever.hybrid_retriever import HybridRetriever
from src.configs.rag_config import RAGConfig

def build_retriever(rag_cfg: RAGConfig):
    retriever_type = rag_cfg.retriever_type.lower()

    if retriever_type == "dense":
        return DenseRetriever(rag_cfg)

    elif retriever_type == "bm25":
        return BM25Retriever(rag_cfg)

    elif retriever_type == "hybrid":
        bm25 = BM25Retriever(rag_cfg)
        dense = DenseRetriever(rag_cfg)
        return HybridRetriever(bm25, dense, rag_cfg.hybrid_alpha)

    else:
        raise ValueError(f"Unsupported retriever type: {retriever_type}")
