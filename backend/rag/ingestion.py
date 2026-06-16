# backend/rag/ingestion.py

import os
import re
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

RAW_DATA_PATH = "backend/data/raw"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "serendib_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk settings
CHUNK_SIZE = 300        # words per chunk
CHUNK_OVERLAP = 50      # overlapping words between chunks



# READ & CLEAN TEXT

def clean_text(text: str) -> str:
    """
    Cleans raw text from files.
    Removes extra spaces, blank lines, weird characters.
    """
    # Remove extra blank lines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove weird characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\/\'\"\n]', '', text)

    # Remove extra spaces
    text = re.sub(r' {2,}', ' ', text)

    # Strip each line
    lines = [line.strip() for line in text.splitlines()]

    # Remove very short lines (less than 3 words)
    lines = [line for line in lines if len(line.split()) >= 3]

    return '\n'.join(lines)


def extract_metadata(text: str, filepath: str) -> dict:
    """
    Extracts metadata from file header.

    Reads:
    TOPIC: Colombo
    CATEGORY: destination
    BEST FOR: Solo Traveler, Cultural Traveler
    """
    metadata = {
        "source": Path(filepath).name,
        "category": "general",
        "topic": "Sri Lanka",
        "best_for": "All Travelers",
        "folder": Path(filepath).parent.name
    }

    lines = text.split('\n')

    for line in lines[:10]:  # Only check first 10 lines
        if line.startswith("TOPIC:"):
            metadata["topic"] = line.replace("TOPIC:", "").strip()
        elif line.startswith("CATEGORY:"):
            metadata["category"] = line.replace("CATEGORY:", "").strip()
        elif line.startswith("BEST FOR:"):
            metadata["best_for"] = line.replace("BEST FOR:", "").strip()

    return metadata


# CHUNK TEXT

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into overlapping chunks.

    Why overlapping?
    So that information at the boundary of
    two chunks is not lost.

    Example with overlap of 2 words:
    Chunk 1: "Colombo is the capital city of Sri Lanka"
    Chunk 2: "of Sri Lanka and is the main entry point"
    """
    # Split into words
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])

        if len(chunk.strip()) > 50:  # Skip very short chunks
            chunks.append(chunk)

        # Move forward by chunk_size minus overlap
        start += chunk_size - overlap

    return chunks


# READ ALL FILES

def load_all_files(data_path: str) -> list[dict]:
    """
    Walks through all folders and reads every .txt file.
    Returns list of {text, metadata, filepath}
    """
    all_files = []
    data_dir = Path(data_path)

    # Walk through all subfolders
    for filepath in data_dir.rglob("*.txt"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()

            if len(raw_text.strip()) < 100:  # Skip empty files
                print(f"   ⚠️  Skipping (too short): {filepath.name}")
                continue

            # Clean the text
            cleaned = clean_text(raw_text)

            # Extract metadata from header
            metadata = extract_metadata(raw_text, str(filepath))

            all_files.append({
                "text": cleaned,
                "metadata": metadata,
                "filepath": str(filepath)
            })

            print(f"   ✅ Loaded: {filepath.name}")

        except Exception as e:
            print(f"   ❌ Error reading {filepath}: {e}")

    return all_files


# BUILD DOCUMENTS FOR CHROMADB

def build_documents(all_files: list[dict]) -> list[Document]:
    """
    Converts loaded files into LangChain Documents.
    Each chunk becomes one Document with metadata.
    """
    documents = []

    for file_data in all_files:
        text = file_data["text"]
        metadata = file_data["metadata"]

        # Split into chunks
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)

    return documents

# LOAD INTO CHROMADB
def load_into_chromadb(documents: list[Document]):
    """
    Converts documents to embeddings and stores in ChromaDB.
    """
    print("\n⏳ Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    print("✅ Embedding model loaded!")

    print(f"\n⏳ Storing {len(documents)} chunks in ChromaDB...")

    # Create or overwrite ChromaDB collection
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH
    )

    print(f"✅ Stored {len(documents)} chunks successfully!")
    return vector_store


# TEST RETRIEVAL
def test_retrieval(vector_store):
    """
    Tests that retrieval is working correctly
    by running sample travel queries.
    """
    print("\n" + "="*50)
    print("🔍 TESTING RETRIEVAL")
    print("="*50)

    test_queries = [
        "What should I do in Ella?",
        "Best food to try in Sri Lanka?",
        "How do I get from Colombo to Kandy by train?",
        "What wildlife can I see in Sri Lanka?",
        "How much does it cost to travel in Sri Lanka?",
        "What is the best time to visit Sri Lanka?",
        "Is Sri Lanka safe for solo travelers?",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)

        results = vector_store.similarity_search(
            query,
            k=2  # Return top 2 results
        )

        for i, doc in enumerate(results):
            print(f"Result {i+1} [{doc.metadata.get('topic')}]:")
            print(f"   {doc.page_content[:150]}...")
            print()

# MAIN — RUN FULL PIPELINE

def run_ingestion_pipeline():
    print("🚀 Starting Serendib AI Data Ingestion Pipeline")
    print("="*50)

    # Step 1 - Load all files
    print("\n📂 Loading files from:", RAW_DATA_PATH)
    all_files = load_all_files(RAW_DATA_PATH)
    print(f"\n✅ Total files loaded: {len(all_files)}")

    if not all_files:
        print("❌ No files found. Check your data path.")
        return

    # Step 2 - Build documents
    print("\n📄 Building document chunks...")
    documents = build_documents(all_files)
    print(f"✅ Total chunks created: {len(documents)}")

    # Step 3 - Load into ChromaDB
    vector_store = load_into_chromadb(documents)

    # Step 4 - Test retrieval
    test_retrieval(vector_store)

    # Final summary
    print("\n" + "="*50)
    print("✅ INGESTION PIPELINE COMPLETE!")
    print("="*50)
    print(f"   Files processed : {len(all_files)}")
    print(f"   Chunks created  : {len(documents)}")
    print(f"   Stored in       : {CHROMA_PATH}")
    print(f"   Collection      : {COLLECTION_NAME}")
    print("\n🚀 Your RAG knowledge base is ready!")


if __name__ == "__main__":
    run_ingestion_pipeline()