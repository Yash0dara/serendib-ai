# test_smart_pipeline.py

import asyncio
from backend.core.smart_pipeline import smart_pipeline
from backend.database.mongodb import mongodb

async def test():
    print("=== Smart Pipeline Test ===\n")

    session_id = "test_smart_001"

    # Create test user
    mongodb.create_user(session_id, language="en")
    conv_id = mongodb.start_conversation(session_id)

    # Test messages covering all scenarios
    messages = [
        "Hello! Can you help me plan a trip?",
        "I want to visit Ella for 3 days on a budget",
        "What food should I try there?",
        "How do I get from Kandy to Ella by train?",
        "Is it safe to hike Adams Peak alone?",
        "My train was cancelled and I am so frustrated",
        "How much will this whole trip cost me?",
        "What was the first place I asked about?",
    ]

    for msg in messages:
        print(f"👤 Traveler : {msg}")
        print("-" * 60)

        result = await smart_pipeline.process(
            session_id=session_id,
            user_message=msg,
            conversation_id=conv_id
        )

        print(f"🤖 Serendib : {result['answer'][:200]}...")
        print(f"📊 Intent   : {result['intent']}")
        print(f"📍 Entities : {result['entities']}")
        print(f"💭 Sentiment: {result['sentiment']}")
        print(f"😤 State    : {result['user_state']}")
        print(f"📚 Sources  : {result['sources']}")
        print("\n" + "="*60 + "\n")

    print("✅ Smart Pipeline test complete!")

if __name__ == "__main__":
    asyncio.run(test())