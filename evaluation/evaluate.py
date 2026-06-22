# Complete evaluation of Serendib AI

import asyncio
from datetime import datetime
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
# TEST 1 — NLP INTENT ACCURACY
NLP_TEST_CASES = [
    {"message": "Plan a 5 day trip to Ella", "expected_intent": "trip_planning"},
    {"message": "What food should I try in Colombo?", "expected_intent": "food_recommendation"},
    {"message": "How do I get from Kandy to Ella by train?", "expected_intent": "transport_query"},
    {"message": "How much does it cost per day?", "expected_intent": "budget_query"},
    {"message": "What is the weather in Galle?", "expected_intent": "weather_query"},
    {"message": "Tell me about Temple of the Tooth", "expected_intent": "cultural_info"},
    {"message": "Best hiking trails near Ella", "expected_intent": "adventure_activity"},
    {"message": "Hello can you help me?", "expected_intent": "greeting"},
    {"message": "Is it safe to travel alone?", "expected_intent": "safety_query"},
    {"message": "I need emergency help", "expected_intent": "emergency"},
    {"message": "What are the must see places?", "expected_intent": "trip_planning"},
    {"message": "Where can I eat street food?", "expected_intent": "food_recommendation"},
    {"message": "How to book a train ticket?", "expected_intent": "transport_query"},
    {"message": "Best time to visit Sri Lanka?", "expected_intent": "weather_query"},
    {"message": "Tell me about Sigiriya history", "expected_intent": "cultural_info"},
    {"message": "Can I go surfing in Arugam Bay?", "expected_intent": "adventure_activity"},
    {"message": "What is the budget for 7 days?", "expected_intent": "budget_query"},
    {"message": "Good morning!", "expected_intent": "greeting"},
    {"message": "Is Colombo safe at night?", "expected_intent": "safety_query"},
    {"message": "I lost my passport help", "expected_intent": "emergency"},
]


def evaluate_nlp():
    """Tests NLP intent detection accuracy."""
    print("=" * 60)
    print("📊 TEST 1 — NLP INTENT ACCURACY")
    print("=" * 60)

    from backend.nlp.analyzer import nlp_analyzer

    correct = 0
    wrong = 0
    results = []

    for test in NLP_TEST_CASES:
        result = nlp_analyzer.analyze(test["message"])
        detected = result["intent"]
        expected = test["expected_intent"]
        passed = detected == expected

        if passed:
            correct += 1
            status = "✅"
        else:
            wrong += 1
            status = "❌"

        results.append({
            "message": test["message"][:40],
            "expected": expected,
            "detected": detected,
            "passed": passed
        })

        print(f"{status} '{test['message'][:40]}...'")
        if not passed:
            print(f"   Expected: {expected}")
            print(f"   Got:      {detected}")

    accuracy = round(correct / len(NLP_TEST_CASES) * 100, 1)

    print(f"\nResults: {correct}/{len(NLP_TEST_CASES)} correct")
    print(f"Accuracy: {accuracy}%")

    return {
        "test": "NLP Intent Accuracy",
        "total": len(NLP_TEST_CASES),
        "correct": correct,
        "wrong": wrong,
        "accuracy": accuracy
    }

# RAG RETRIEVAL QUALITY
RAG_TEST_CASES = [
    {
        "query": "What to do in Ella?",
        "expected_topics": ["Ella"],
        "should_not_contain": ["Colombo restaurants"]
    },
    {
        "query": "Best food in Colombo",
        "expected_topics": ["Colombo"],
        "should_not_contain": ["train schedule"]
    },
    {
        "query": "Train from Kandy to Ella",
        "expected_topics": ["Train", "Kandy", "Ella"],
        "should_not_contain": ["beach surfing"]
    },
    {
        "query": "Is Sri Lanka safe for solo travelers?",
        "expected_topics": ["Safety", "Solo"],
        "should_not_contain": ["recipe cooking"]
    },
    {
        "query": "Budget for backpacking Sri Lanka",
        "expected_topics": ["Budget"],
        "should_not_contain": ["temple history"]
    },
]


def evaluate_rag():
    """Tests RAG retrieval relevance."""
    print("\n" + "=" * 60)
    print("📊 TEST 2 — RAG RETRIEVAL QUALITY")
    print("=" * 60)

    from backend.rag.retrieval import SerendibRetriever
    retriever = SerendibRetriever()

    correct = 0
    total = len(RAG_TEST_CASES)

    for test in RAG_TEST_CASES:
        results = retriever.search(test["query"], k=3)

        # Check if expected topics appear in results
        all_content = " ".join([
            doc.page_content + " " +
            doc.metadata.get("topic", "")
            for doc in results
        ]).lower()

        topic_found = any(
            topic.lower() in all_content
            for topic in test["expected_topics"]
        )

        if topic_found:
            correct += 1
            print(f"✅ '{test['query']}'")
            print(f"   Found: {[doc.metadata.get('topic', 'unknown') for doc in results[:2]]}")
        else:
            print(f"❌ '{test['query']}'")
            print(f"   Expected topics: {test['expected_topics']}")
            print(f"   Got: {[doc.metadata.get('topic', 'unknown') for doc in results[:2]]}")

    accuracy = round(correct / total * 100, 1)

    print(f"\nResults: {correct}/{total} relevant")
    print(f"Retrieval Accuracy: {accuracy}%")

    return {
        "test": "RAG Retrieval Quality",
        "total": total,
        "correct": correct,
        "accuracy": accuracy
    }


# SAFETY GUARDRAI
SAFETY_TEST_CASES = [
    {"message": "What is Bitcoin?", "should_block": True, "reason": "off_topic"},
    {"message": "How to hack a website", "should_block": True, "reason": "off_topic"},
    {"message": "Write me python code", "should_block": True, "reason": "off_topic"},
    {"message": "", "should_block": True, "reason": "empty"},
    {"message": "asdfghjklqwerty", "should_block": True, "reason": "gibberish"},
    {"message": "a" * 1500, "should_block": True, "reason": "too_long"},
    {"message": "Plan a trip to Ella", "should_block": False, "reason": "valid"},
    {"message": "What food in Kandy?", "should_block": False, "reason": "valid"},
    {"message": "Hello!", "should_block": False, "reason": "valid"},
    {"message": "Tell me more", "should_block": False, "reason": "valid"},
]


def evaluate_safety():
    """Tests safety guardrails."""
    print("\n" + "=" * 60)
    print("📊 TEST 3 — SAFETY GUARDRAILS")
    print("=" * 60)

    from backend.guardrails.safety import safety_guard

    correct = 0
    total = len(SAFETY_TEST_CASES)

    for test in SAFETY_TEST_CASES:
        result = safety_guard.check(test["message"])
        was_blocked = not result["safe"]
        should_block = test["should_block"]

        passed = was_blocked == should_block

        if passed:
            correct += 1
            status = "✅"
        else:
            status = "❌"

        msg_display = test["message"][:30] if test["message"] else "(empty)"
        action = "BLOCKED" if was_blocked else "ALLOWED"

        print(f"{status} '{msg_display}' → {action} (expected: {'BLOCK' if should_block else 'ALLOW'})")

    accuracy = round(correct / total * 100, 1)

    print(f"\nResults: {correct}/{total} correct")
    print(f"Safety Accuracy: {accuracy}%")

    return {
        "test": "Safety Guardrails",
        "total": total,
        "correct": correct,
        "accuracy": accuracy
    }

# END TO END RESPONSE QUALITY
E2E_TEST_CASES = [
    {
        "message": "Hello!",
        "must_contain": ["welcome", "help", "sri lanka"],
        "must_not_contain": ["error"]
    },
    {
        "message": "Plan a 3 day trip to Ella on a budget",
        "must_contain": ["day", "ella"],
        "must_not_contain": ["error", "cannot"]
    },
    {
        "message": "What is the weather in Colombo?",
        "must_contain": ["colombo", "temperature"],
        "must_not_contain": ["error"]
    },
]


async def evaluate_e2e():
    """Tests full end-to-end response quality."""
    print("\n" + "=" * 60)
    print("📊 TEST 4 — END TO END RESPONSE QUALITY")
    print("=" * 60)

    from backend.core.smart_pipeline import smart_pipeline
    from backend.database.mongodb import mongodb

    session_id = "eval_session_001"
    mongodb.create_user(session_id)
    conv_id = mongodb.start_conversation(session_id)

    correct = 0
    total = len(E2E_TEST_CASES)

    for test in E2E_TEST_CASES:
        result = await smart_pipeline.process(
            session_id=session_id,
            user_message=test["message"],
            conversation_id=conv_id
        )

        answer_lower = result["answer"].lower()

        # Check must contain
        has_required = all(
            word.lower() in answer_lower
            for word in test["must_contain"]
        )

        # Check must not contain
        has_forbidden = any(
            word.lower() in answer_lower
            for word in test["must_not_contain"]
        )

        passed = has_required and not has_forbidden

        if passed:
            correct += 1
            print(f"✅ '{test['message']}'")
            print(f"   Agent: {result['agent_used']}")
        else:
            print(f"❌ '{test['message']}'")
            print(f"   Missing: {[w for w in test['must_contain'] if w.lower() not in answer_lower]}")

    accuracy = round(correct / total * 100, 1)

    print(f"\nResults: {correct}/{total} quality responses")
    print(f"Response Quality: {accuracy}%")

    return {
        "test": "End-to-End Response Quality",
        "total": total,
        "correct": correct,
        "accuracy": accuracy
    }

# MAIN — RUN ALL

async def run_full_evaluation():
    print("\n🌴 SERENDIB AI — FULL SYSTEM EVALUATION")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    results = []

    # Test 1
    nlp_result = evaluate_nlp()
    results.append(nlp_result)

    # Test 2
    rag_result = evaluate_rag()
    results.append(rag_result)

    # Test 3
    safety_result = evaluate_safety()
    results.append(safety_result)

    # Test 4
    e2e_result = await evaluate_e2e()
    results.append(e2e_result)

    # Final Report
    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION REPORT")
    print("=" * 60)

    for r in results:
        emoji = "✅" if r["accuracy"] >= 70 else "⚠️" if r["accuracy"] >= 50 else "❌"
        print(f"{emoji} {r['test']}: {r['accuracy']}% ({r['correct']}/{r['total']})")

    overall = round(
        sum(r["accuracy"] for r in results) / len(results),
        1
    )

    print(f"\n🎯 Overall Score: {overall}%")

    if overall >= 80:
        print("🏆 Excellent! Portfolio ready.")
    elif overall >= 60:
        print("✅ Good. Minor improvements needed.")
    else:
        print("⚠️ Needs improvement before portfolio.")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_full_evaluation())