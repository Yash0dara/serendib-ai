from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("MONGODB_URI")

if not uri:
    print("❌ MONGODB_URI not found")
    exit()

try:
    client = MongoClient(uri)
    db = client["serendib_ai"]
    
    # Insert test document
    result = db.test_collection.insert_one({"message": "MongoDB works!"})
    
    print("✅ MongoDB connected successfully!")
    print("Inserted ID:", result.inserted_id)

except Exception as e:
    print("❌ MongoDB Error:", e)