# test_embeddings.py

from sentence_transformers import SentenceTransformer
import numpy as np

print("⏳ Loading embedding model...")
print("   First time will download the model (~90MB)")
print("   This only happens once\n")

# Load free embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("✅ Model loaded!\n")

# Test sentences about Sri Lanka
sentences = [
    "What are the best beaches in Sri Lanka?",
    "Tell me about Sigiriya rock fortress",
    "What food should I try in Colombo?",
    "How do I get from Kandy to Ella by train?",
    "What is the weather like in Galle?",
]

print("⏳ Converting sentences to embeddings...")
embeddings = model.encode(sentences)

print(f"✅ Embeddings created!")
print(f"   Number of sentences : {len(embeddings)}")
print(f"   Embedding size      : {len(embeddings[0])} numbers per sentence\n")

# Show similarity between sentences
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"Similarity test:")
print(f"   '{sentences[0]}'")
print(f"   '{sentences[1]}'")
print(f"   Similarity score: {similarity:.2f}")
print(f"   (1.0 = identical, 0.0 = completely different)\n")

print("✅ Embeddings working perfectly!")