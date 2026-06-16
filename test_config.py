# test_config.py

from backend.config import config

config.validate()

print("\n=== Config Values ===")
print(f"App Name      : {config.APP_NAME}")
print(f"Groq Model    : {config.GROQ_MODEL}")
print(f"Embedding Model: {config.EMBEDDING_MODEL}")
print(f"Chroma Path   : {config.CHROMA_PATH}")
print(f"MongoDB DB    : {config.MONGODB_DB_NAME}")