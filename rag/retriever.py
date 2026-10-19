from pathlib import Path

import faiss
import pandas as pd

from .embeddings import EmbeddingModel


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_STORE_DIR = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
)

INDEX_PATH = (
    VECTOR_STORE_DIR
    / "faq_index.faiss"
)

METADATA_PATH = (
    VECTOR_STORE_DIR
    / "faq_metadata.csv"
)


# =========================================================
# FAQ RETRIEVER
# =========================================================

class FAQRetriever:
    """
    Semantic FAQ retriever using BGE embeddings and FAISS.
    """

    def __init__(
        self,
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    ):

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        # -------------------------------------------------
        # Validate files
        # -------------------------------------------------

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"FAQ metadata not found: {self.metadata_path}"
            )

        # -------------------------------------------------
        # Load FAISS index
        # -------------------------------------------------

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(self.index_path)
        )

        # -------------------------------------------------
        # Load FAQ metadata
        # -------------------------------------------------

        print("Loading FAQ metadata...")

        self.metadata = pd.read_csv(
            self.metadata_path
        )

        # -------------------------------------------------
        # Load embedding model
        # -------------------------------------------------

        print("Loading embedding model...")

        self.embedding_model = EmbeddingModel()

        print(
            f"Retriever ready. "
            f"Vectors: {self.index.ntotal}"
        )

    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query,
        top_k=5
    ):
        """
        Retrieve the most relevant FAQs for a query.

        Parameters
        ----------
        query : str
            Customer support query.

        top_k : int
            Number of FAQ results to return.

        Returns
        -------
        list[dict]
            Ranked FAQ retrieval results.
        """

        if not query or not query.strip():
            return []

        # -------------------------------------------------
        # Convert query to embedding
        # -------------------------------------------------

        query_embedding = self.embedding_model.encode(
            query
        )

        # -------------------------------------------------
        # FAISS search
        # -------------------------------------------------

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        # -------------------------------------------------
        # Convert FAISS results to dictionaries
        # -------------------------------------------------

        for rank, (score, index_id) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):

            # FAISS can return -1 when there aren't enough
            # vectors. Protect against invalid indexes.
            if index_id < 0:
                continue

            faq = self.metadata.iloc[index_id]

            results.append({
                "rank": rank,
                "score": float(score),
                "faq_id": faq["faq_id"],
                "category": faq["category"],
                "question": faq["question"],
                "answer": faq["answer"]
            })

        return results

# =========================================================
# RETRIEVAL FUNCTION FOR AGENTS
# =========================================================

_retriever = None


def retrieve_documents(
    query,
    top_k=5
):
    """
    Retrieve relevant FAQ documents for an agent.

    Uses a shared FAQRetriever instance so that
    the FAISS index and embedding model are loaded
    only once.
    """

    global _retriever

    # Initialize retriever only when first needed
    if _retriever is None:
        _retriever = FAQRetriever()

    return _retriever.search(
        query,
        top_k=top_k
    )


# =========================================================
# TEST RETRIEVER
# =========================================================

if __name__ == "__main__":

    retriever = FAQRetriever()

    test_queries = [
        "Where is my package?",
        "My payment was charged twice",
        "I want to return this product",
        "I cannot login to my account",
        "The app keeps crashing"
    ]

    for query in test_queries:

        print("\n" + "=" * 80)
        print("QUERY:", query)
        print("=" * 80)

        results = retriever.search(
            query,
            top_k=3
        )

        for result in results:

            print(
                f"\nRank: {result['rank']}"
            )

            print(
                f"Score: {result['score']:.4f}"
            )

            print(
                f"FAQ ID: {result['faq_id']}"
            )

            print(
                f"Category: {result['category']}"
            )

            print(
                f"Question: {result['question']}"
            )

            print(
                f"Answer: {result['answer']}"
            )