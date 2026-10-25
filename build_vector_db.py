
"""
build_vector_db.py

Purpose:
1. Load the Tamil text chunks from chunks.json.
2. Generate numerical embeddings using Sentence Transformers.
3. Build a FAISS index for similarity search.
4. Save the index and chunk metadata for later retrieval.

This script does not modify app.py or the source PDFs.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# STEP 1: Define project paths
# --------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent

# JSON file containing the 442 text chunks
CHUNKS_FILE = PROJECT_FOLDER / "chunks.json"

# Folder where the FAISS index and metadata will be stored
VECTOR_DB_FOLDER = PROJECT_FOLDER / "vector_db"

# Files created by this script
INDEX_FILE = VECTOR_DB_FOLDER / "jothidam.index"
METADATA_FILE = VECTOR_DB_FOLDER / "chunks_metadata.json"


# --------------------------------------------------
# STEP 2: Configure the embedding model
# --------------------------------------------------

# This multilingual model supports Tamil and many other languages.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# --------------------------------------------------
# STEP 3: Load the text chunks
# --------------------------------------------------

def load_chunks():
    """Load and validate chunks from the JSON file."""

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}\n"
            "Run prepare_chunks.py first."
        )

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json contains no usable chunks.")

    # Keep only records that contain non-empty text.
    valid_chunks = [
        chunk
        for chunk in chunks
        if isinstance(chunk, dict)
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]

    if not valid_chunks:
        raise ValueError("No non-empty text chunks were found.")

    return valid_chunks


# --------------------------------------------------
# STEP 4: Generate embeddings and build the index
# --------------------------------------------------

def main():
    """Build and save the FAISS vector database."""

    # Load the source chunks
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} text chunks.")

    # Create the output directory if necessary
    VECTOR_DB_FOLDER.mkdir(parents=True, exist_ok=True)

    # Load the multilingual embedding model.
    # The first run may download the model from Hugging Face.
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # Extract the text that will be embedded
    texts = [chunk["text"] for chunk in chunks]

    # Convert text into numerical vectors.
    # Normalization makes cosine similarity searchable with inner product.
    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # FAISS requires a two-dimensional float32 array
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Verify that the embeddings are valid
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("The generated embeddings have an unexpected shape.")

    if not np.isfinite(embeddings).all():
        raise ValueError("The embeddings contain invalid numerical values.")

    # Build an exact-search FAISS index using inner product.
    # Since vectors are normalized, this ranks by cosine similarity.
    vector_dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(vector_dimension)

    # Add every chunk embedding to the index
    index.add(embeddings)

    # Save the FAISS index to disk
    faiss.write_index(index, str(INDEX_FILE))

    # Save the chunk text and identifiers in matching index order.
    # FAISS result position 0 corresponds to metadata record 0, etc.
    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Display a summary
    print("\n" + "=" * 50)
    print("FAISS VECTOR DATABASE CREATED")
    print("=" * 50)
    print(f"Chunks indexed: {index.ntotal}")
    print(f"Embedding dimensions: {vector_dimension}")
    print(f"Index file: {INDEX_FILE}")
    print(f"Metadata file: {METADATA_FILE}")


# --------------------------------------------------
# STEP 5: Run the program
# --------------------------------------------------

if __name__ == "__main__":
    main()
