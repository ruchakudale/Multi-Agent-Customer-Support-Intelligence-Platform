from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingModel:
    """
    Wrapper around the BGE-small embedding model.
    """

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Convert text or a list of texts into normalized embeddings.

        Parameters
        ----------
        texts : str or list[str]
            Text to embed.

        Returns
        -------
        numpy.ndarray
            Normalized embeddings.
        """

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return np.asarray(
            embeddings,
            dtype="float32"
        )


if __name__ == "__main__":

    embedding_model = EmbeddingModel()

    test_text = "Where is my package?"

    embedding = embedding_model.encode(test_text)

    print("Embedding shape:", embedding.shape)
    print("Embedding dtype:", embedding.dtype)