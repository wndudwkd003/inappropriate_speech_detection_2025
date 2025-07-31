from src.rag.retriever.base_retriever import BaseRetriever
from src.rag.retriever.bm25_retriever import BM25Retriever
from src.rag.retriever.dense_retriever import DenseRetriever
from collections import defaultdict

class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        alpha=0.5
    ):
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int) -> list[str]:
        bm25_docs = self.bm25.retrieve(query, top_k * 2)
        dense_docs = self.dense.retrieve(query, top_k * 2)

        score_map = defaultdict(float)
        for i, doc in enumerate(dense_docs):
            score_map[doc] += self.alpha * (len(dense_docs) - i)
        for i, doc in enumerate(bm25_docs):
            score_map[doc] += (1 - self.alpha) * (len(bm25_docs) - i)

        ranked_docs = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:top_k]]
