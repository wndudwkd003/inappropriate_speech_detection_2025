import os, json
from rank_bm25 import BM25Okapi
from src.rag.retriever.base_retriever import BaseRetriever
from src.configs.rag_config import RAGConfig
from konlpy.tag import Okt

class BM25Retriever(BaseRetriever):
    def __init__(self, rag_cfg: RAGConfig):
        self.corpus_path = os.path.join(rag_cfg.index_dir, rag_cfg.corpus_name)
        self.tokenizer = Okt()
        self.batch_size = rag_cfg.batch_size

        # JSON 파일에서 문서 불러오기
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        # ✅ 형태소 기반 토큰화
        self.tokenized_corpus = [self._tokenize_korean(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int) -> list[str]:
        # ✅ 질의도 동일하게 형태소 토큰화
        tokenized_query = self._tokenize_korean(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.documents[i] for i in top_indices]

    def retrieve_batch(self, queries: list[str], top_k: int) -> list[list[str]]:
        return [self.retrieve(query, top_k) for query in queries]



    def _tokenize_korean(self, text: str) -> list[str]:
        return self.tokenizer.morphs(text)
