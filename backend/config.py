# backend/config.py

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

class Config:
    # App Settings
    APP_NAME = os.getenv("APP_NAME", "Serendib AI")
    DEBUG = os.getenv("DEBUG", "False") == "True"

    # Groq LLM
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.1-8b-instant"

    # Embeddings (Free HuggingFace)
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # ChromaDB
    CHROMA_PATH = "./chroma_db"
    CHROMA_COLLECTION = "serendib_knowledge"

    # MongoDB
    MONGODB_URI = os.getenv("MONGODB_URI")
    MONGODB_DB_NAME = "serendib_ai"

    # Weather API
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    OPENWEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5"

    # Redis (skipped for now)
    REDIS_URL = os.getenv("REDIS_URL", None)

    # Validation
    @staticmethod
    def validate():
        required_keys = {
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
            "MONGODB_URI": os.getenv("MONGODB_URI"),
            "OPENWEATHER_API_KEY": os.getenv("OPENWEATHER_API_KEY"),
        }

        print("=== Serendib AI Config Check ===")
        all_good = True

        for name, value in required_keys.items():
            if value:
                print(f"✅ {name} is set")
            else:
                print(f"❌ {name} is missing")
                all_good = False

        if all_good:
            print("\n✅ All required keys found!")
        else:
            print("\n❌ Some keys are missing. Check your .env file.")

        return all_good


# Single instance used across project
config = Config()