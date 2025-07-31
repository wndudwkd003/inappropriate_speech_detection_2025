from sentence_transformers import SentenceTransformer
import torch


class TextEmbedder:
    def __init__(
        self,
        model_name: str = "",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64
    ) -> list[list[float]]:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )



# def get_text_embeddings(texts: list[str], model_name: str) -> list[list[float]]:
#     embedder = TextEmbedder(model_name)
#     return embedder.encode(texts)
