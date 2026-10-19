from pathlib import Path
import pandas as pd
import faiss

from .embeddings import EmbeddingModel


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

# build_faiss_index.py is inside:
# project_root/rag/
#
# Therefore:
# parent     = rag/
# parent.parent = project_root/

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FAQ_PATH = PROJECT_ROOT / "data" / "raw" / "faq.csv"

VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

INDEX_PATH = VECTOR_STORE_DIR / "faq_index.faiss"

METADATA_PATH = VECTOR_STORE_DIR / "faq_metadata.csv"


# ---------------------------------------------------------
# BUILD FAISS INDEX
# ---------------------------------------------------------

def build_faiss_index():

    print("=" * 60)
    print("BUILDING FAQ FAISS INDEX")
    print("=" * 60)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nFAQ path:")
    print(FAQ_PATH)

    print("\nFAQ exists:")
    print(FAQ_PATH.exists())

    if not FAQ_PATH.exists():
        raise FileNotFoundError(
            f"FAQ file not found: {FAQ_PATH}"
        )

    # -----------------------------------------------------
    # Load FAQ
    # -----------------------------------------------------

    print("\nLoading FAQ data...")

    faq_df = pd.read_csv(FAQ_PATH)

    print("FAQ documents:", len(faq_df))

    # -----------------------------------------------------
    # Prepare documents
    # -----------------------------------------------------

    faq_df["document_text"] = (
        "Category: "
        + faq_df["category"].astype(str)
        + "\nQuestion: "
        + faq_df["question"].astype(str)
        + "\nAnswer: "
        + faq_df["answer"].astype(str)
    )

    documents = faq_df["document_text"].tolist()

    # -----------------------------------------------------
    # Load embedding model
    # -----------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = EmbeddingModel()

    # -----------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------

    print("\nGenerating embeddings...")

    embeddings = embedding_model.encode(
        documents
    )

    print("Embedding shape:", embeddings.shape)

    # -----------------------------------------------------
    # Create FAISS index
    # -----------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print("\nFAISS index created.")
    print("Total vectors:", index.ntotal)

    # -----------------------------------------------------
    # Create vector store directory
    # -----------------------------------------------------

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Save FAISS index
    # -----------------------------------------------------

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    # -----------------------------------------------------
    # Save metadata
    # -----------------------------------------------------

    faq_df[
        [
            "faq_id",
            "question",
            "answer",
            "category"
        ]
    ].to_csv(
        METADATA_PATH,
        index=False
    )

    # -----------------------------------------------------
    # Final confirmation
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("FAISS INDEX BUILT SUCCESSFULLY")
    print("=" * 60)

    print("Vectors:", index.ntotal)
    print("Index:", INDEX_PATH)
    print("Metadata:", METADATA_PATH)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    build_faiss_index()