# backend/database/mongodb.py

from pymongo import MongoClient, ASCENDING
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class MongoDB:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI"))
        self.db = self.client["serendib_ai"]

        # Collections
        self.users = self.db["users"]
        self.conversations = self.db["conversations"]
        self.messages = self.db["messages"]
        self.itineraries = self.db["itineraries"]

    def setup_collections(self):
        """
        Creates collections and indexes.
        Indexes make searches faster.
        """

        # ── Users Collection ──
        self.users.create_index(
            [("session_id", ASCENDING)],
            unique=True
        )
        print("✅ Users collection ready")

        # ── Conversations Collection ──
        self.conversations.create_index(
            [("session_id", ASCENDING)]
        )
        print("✅ Conversations collection ready")

        # ── Messages Collection ──
        self.messages.create_index(
            [("conversation_id", ASCENDING)]
        )
        print("✅ Messages collection ready")

        # ── Itineraries Collection ──
        self.itineraries.create_index(
            [("session_id", ASCENDING)]
        )
        print("✅ Itineraries collection ready")

        print("\n✅ All collections set up successfully!")

    # ── USER OPERATIONS ──

    def create_user(self, session_id: str, language: str = "en"):
        """Creates a new user if not exists."""
        existing = self.users.find_one({"session_id": session_id})
        if existing:
            return existing

        user = {
            "session_id": session_id,
            "language": language,
            "traveler_type": None,
            "budget_level": None,
            "interests": [],
            "created_at": datetime.utcnow(),
            "last_active": datetime.utcnow()
        }
        self.users.insert_one(user)
        return user

    def update_user_profile(self, session_id: str, profile: dict):
        """Updates user travel preferences."""
        self.users.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    **profile,
                    "last_active": datetime.utcnow()
                }
            }
        )

    def get_user(self, session_id: str):
        """Gets user by session ID."""
        return self.users.find_one({"session_id": session_id})

    # ── CONVERSATION OPERATIONS ──

    def start_conversation(self, session_id: str):
        """Starts a new conversation."""
        conversation = {
            "session_id": session_id,
            "started_at": datetime.utcnow(),
            "ended_at": None,
            "status": "active",
            "message_count": 0
        }
        result = self.conversations.insert_one(conversation)
        return str(result.inserted_id)

    def end_conversation(self, conversation_id: str):
        """Marks conversation as ended."""
        from bson import ObjectId
        self.conversations.update_one(
            {"_id": ObjectId(conversation_id)},
            {
                "$set": {
                    "ended_at": datetime.utcnow(),
                    "status": "ended"
                }
            }
        )

    # ── MESSAGE OPERATIONS ──

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        intent: str = None
    ):
        """
        Saves a message to database.
        role = 'user' or 'assistant'
        """
        message = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "intent": intent,
            "timestamp": datetime.utcnow()
        }
        self.messages.insert_one(message)

        # Update message count
        from bson import ObjectId
        self.conversations.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$inc": {"message_count": 1}}
        )

    def get_messages(self, conversation_id: str) -> list:
        """Gets all messages for a conversation."""
        messages = self.messages.find(
            {"conversation_id": conversation_id},
            sort=[("timestamp", ASCENDING)]
        )
        return list(messages)

    # ── ITINERARY OPERATIONS ──

    def save_itinerary(
        self,
        session_id: str,
        itinerary: dict
    ):
        """Saves a generated itinerary."""
        doc = {
            "session_id": session_id,
            "created_at": datetime.utcnow(),
            "itinerary": itinerary,
            "duration_days": itinerary.get("duration_days"),
            "traveler_type": itinerary.get("traveler_type")
        }
        result = self.itineraries.insert_one(doc)
        return str(result.inserted_id)

    def get_itineraries(self, session_id: str) -> list:
        """Gets all itineraries for a user."""
        itineraries = self.itineraries.find(
            {"session_id": session_id},
            sort=[("created_at", ASCENDING)]
        )
        return list(itineraries)


# Single instance used across project
mongodb = MongoDB()