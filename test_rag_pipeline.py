# test_rag_pipeline.py

import asyncio
from backend.rag.pipeline import rag_pipeline

async def test_pipeline():
    print("=== Testing Serendib RAG Pipeline ===\n")

    session_id = "test_session_001"

    # Test questions covering all features
    questions = [
        {
            "q": "What should I do in Ella?",
            "type": "Adventure Traveler",
            "category": "destination"
        },
        {
            "q": "What local food should I try?",
            "type": "Solo Traveler",
            "category": "food"
        },
        {
            "q": "How do I get from Colombo to Kandy?",
            "type": None,
            "category": "transport"
        },
        {
            "q": "What was the first place I asked about?",
            "type": None,
            "category": None
        },
    ]

    for i, item in enumerate(questions, 1):
        print(f"Question {i}: {item['q']}")
        print("-" * 50)

        result = await rag_pipeline.chat(
            session_id=session_id,
            user_message=item["q"],
            traveler_type=item["type"],
            category=item["category"]
        )

        print(f"Answer:\n{result['answer']}")
        print(f"\nSources used: {result['sources']}")
        print("\n" + "="*50 + "\n")

    print("✅ RAG Pipeline test complete!")

if __name__ == "__main__":
    asyncio.run(test_pipeline())