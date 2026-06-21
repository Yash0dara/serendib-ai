import asyncio
from backend.core.smart_pipeline import smart_pipeline
from backend.database.mongodb import mongodb

async def test():
    print("=== Multi-Agent Smart Pipeline Test ===\n")

    session_id = "test_agent_001"
    mongodb.create_user(session_id, language="en")
    conv_id = mongodb.start_conversation(session_id)

    messages = [
        "Hello! Can you help me plan a trip?",
        "I want to visit Ella for 3 days on a budget",
        "What food should I try there?",
        "How do I get from Kandy to Ella by train?",
        "What is the weather like in Colombo right now?",
        "How much will a 3 day budget trip cost me?",
        "Is it safe to hike Adams Peak alone?",
        "My train was cancelled and I am frustrated",
        "What was the first place I asked about?",
    ]

    for msg in messages:
        print(f"👤 Traveler  : {msg}")
        print("-" * 60)

        result = await smart_pipeline.process(
            session_id=session_id,
            user_message=msg,
            conversation_id=conv_id
        )

        print(f"🤖 Agent     : {result['agent_used']}")
        print(f"📊 Intent    : {result['intent']}")
        print(f"💭 Sentiment : {result['sentiment']}")
        print(f"😤 State     : {result['user_state']}")
        print(f"✅ Validated : {result['validation'].get('passed')}")
        print(f"\n🗣 Response  :\n{result['answer'][:300]}...")
        print("\n" + "="*60 + "\n")

    print("✅ Multi-Agent Pipeline test complete!")

if __name__ == "__main__":
    asyncio.run(test())