# test_mongodb_collections.py

from backend.database.mongodb import mongodb

print("=== MongoDB Collections Test ===\n")

# Step 1 - Setup collections
print("⏳ Setting up collections...")
mongodb.setup_collections()

# Step 2 - Create a test user
print("\n⏳ Testing user creation...")
user = mongodb.create_user(
    session_id="test_session_001",
    language="en"
)
print(f"✅ User created: {user['session_id']}")

# Step 3 - Update user profile
print("\n⏳ Testing profile update...")
mongodb.update_user_profile(
    session_id="test_session_001",
    profile={
        "traveler_type": "adventure",
        "budget_level": "mid",
        "interests": ["hiking", "wildlife", "food"]
    }
)
updated = mongodb.get_user("test_session_001")
print(f"✅ Profile updated!")
print(f"   Traveler type : {updated['traveler_type']}")
print(f"   Budget level  : {updated['budget_level']}")
print(f"   Interests     : {updated['interests']}")

# Step 4 - Start conversation
print("\n⏳ Testing conversation...")
conv_id = mongodb.start_conversation("test_session_001")
print(f"✅ Conversation started: {conv_id}")

# Step 5 - Save messages
print("\n⏳ Testing messages...")
mongodb.save_message(
    conversation_id=conv_id,
    role="user",
    content="I want to visit Sri Lanka for 7 days",
    intent="trip_planning"
)
mongodb.save_message(
    conversation_id=conv_id,
    role="assistant",
    content="Great! I can help you plan a 7-day Sri Lanka itinerary.",
    intent=None
)
messages = mongodb.get_messages(conv_id)
print(f"✅ Messages saved: {len(messages)} messages")
for msg in messages:
    print(f"   {msg['role']}: {msg['content'][:50]}...")

# Step 6 - Save itinerary
print("\n⏳ Testing itinerary save...")
sample_itinerary = {
    "duration_days": 7,
    "traveler_type": "adventure",
    "days": [
        {"day": 1, "location": "Colombo", "activities": ["City tour", "Galle Face"]},
        {"day": 2, "location": "Sigiriya", "activities": ["Rock climb", "Cave temples"]},
        {"day": 3, "location": "Kandy", "activities": ["Temple of Tooth", "Botanical Gardens"]},
    ]
}
itin_id = mongodb.save_itinerary("test_session_001", sample_itinerary)
print(f"✅ Itinerary saved: {itin_id}")

print("\n" + "=" * 40)
print("✅ ALL MONGODB TESTS PASSED!")
print("=" * 40)
print("✅ Users collection      — Working")
print("✅ Conversations         — Working")
print("✅ Messages              — Working")
print("✅ Itineraries           — Working")
print("\n🚀 MongoDB fully ready!")