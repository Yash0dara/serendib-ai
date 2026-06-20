from backend.nlp.analyzer import nlp_analyzer

test_messages = [
    "I want to visit Ella for 3 days next month on a budget",
    "What is the best street food to try in Colombo?",
    "How do I get from Kandy to Galle by train?",
    "I am frustrated because my train was cancelled",
    "Hello! Can you help me plan a trip?",
    "Is it safe to hike Adam's Peak alone?"
]

print("=== NLP Layer Test ===\n")

for msg in test_messages:
    print(f"Message: '{msg}'")
    result = nlp_analyzer.analyze(msg)
    print(f"  Intent    : {result['intent']} ({result['intent_confidence']})")
    print(f"  Entities  : {result['entities']}")
    print(f"  Sentiment : {result['sentiment']} ({result['sentiment_confidence']})")
    print("-" * 50)

print("✅ NLP Layer test complete!")