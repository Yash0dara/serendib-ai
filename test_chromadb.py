# test_chromadb.py

import chromadb
from sentence_transformers import SentenceTransformer

print("⏳ Setting up ChromaDB...")

# Create local ChromaDB (saves to your computer)
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection (like a table in database)
collection = client.get_or_create_collection(
    name="serendib_knowledge",
    metadata={"description": "Sri Lanka travel knowledge base"}
)

print("✅ ChromaDB collection created!\n")

# Load embedding model
print("⏳ Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Model loaded!\n")

# Sample Sri Lanka knowledge documents
documents = [
    {
        "id": "doc1",
        "text": "Sigiriya is an ancient rock fortress located in the Matale District of Sri Lanka. It rises 200 meters above the surrounding plains. Sigiriya is one of Sri Lanka's most visited tourist attractions and a UNESCO World Heritage Site.",
        "category": "destination"
    },
    {
        "id": "doc2",
        "text": "Ella is a small town in the Badulla District of Sri Lanka. It is surrounded by hills covered in tea plantations. Ella is famous for Nine Arch Bridge, Little Adam's Peak hiking, and Ravana Falls.",
        "category": "destination"
    },
    {
        "id": "doc3",
        "text": "Sri Lankan cuisine includes rice and curry, hoppers, kottu roti, and string hoppers. Rice and curry is the staple meal eaten for breakfast, lunch and dinner. Hoppers are bowl shaped pancakes made from fermented rice flour.",
        "category": "food"
    },
    {
        "id": "doc4",
        "text": "The Kandy to Ella train journey is considered one of the most scenic train rides in the world. The journey takes about 7 hours passing through tea plantations, waterfalls, and mountain tunnels. Book seats in advance especially observation cars.",
        "category": "transport"
    },
    {
        "id": "doc5",
        "text": "Yala National Park is the most visited national park in Sri Lanka. It has one of the highest leopard densities in the world. Best time to visit is February to July. Safari jeeps are available at the park entrance.",
        "category": "wildlife"
    },
]

# Convert documents to embeddings and store
print("⏳ Storing documents in ChromaDB...")

texts = [doc["text"] for doc in documents]
ids = [doc["id"] for doc in documents]
categories = [doc["category"] for doc in documents]

# Create embeddings
embeddings = model.encode(texts).tolist()

# Add to ChromaDB
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=texts,
    metadatas=[{"category": cat} for cat in categories]
)

print(f"✅ Stored {len(documents)} documents!\n")

# Now test searching
print("=" * 50)
print("🔍 SEARCH TEST")
print("=" * 50)

def search(query: str, n_results: int = 2):
    print(f"\nQuery: '{query}'")
    print("-" * 40)

    # Convert query to embedding
    query_embedding = model.encode([query]).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    for i, doc in enumerate(results["documents"][0]):
        print(f"Result {i+1}:")
        print(f"   {doc[:100]}...")
        print()

# Run searches
search("What should I see in the mountains?")
search("Tell me about Sri Lankan food")
search("How do I travel by train?")
search("Where can I see wildlife and animals?")

print("✅ ChromaDB search working perfectly!")