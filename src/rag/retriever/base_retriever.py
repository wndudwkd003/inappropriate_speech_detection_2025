from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[str]:
        pass

    def retrieve_batch(self, queries: list[str], top_k: int) -> list[list[str]]:
        pass
