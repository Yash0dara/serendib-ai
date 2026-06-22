import os
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from backend.nlp.analyzer import nlp_analyzer
from backend.agents.router import router_agent
from backend.agents.knowledge import knowledge_agent
from backend.agents.itinerary import itinerary_agent
from backend.agents.weather import weather_agent
from backend.agents.budget import budget_agent
from backend.agents.validator import validator_agent
from backend.database.mongodb import mongodb

load_dotenv()

# Intent → RAG category for knowledge agent
INTENT_TO_CATEGORY = {
    "food_recommendation": "food",
    "transport_query":     "transport",
    "cultural_info":       "culture",
    "adventure_activity":  "adventure",
    "safety_query":        "practical",
    "general_question":    None
}

# Greeting response
GREETING_RESPONSE = """
Hello! 👋 Welcome to Serendib AI.

I am your intelligent Sri Lanka travel assistant.

I can help you with:
🗺  **Trip Planning**     → Personalized itineraries
🍛  **Food Guide**        → Local cuisine and restaurants
🚂  **Transport**         → Trains, buses, tuk-tuks
💰  **Budget Planning**   → Daily cost estimates
🌤  **Weather**           → Real time conditions
🏛  **Culture**           → Temples, festivals, heritage
🏄  **Adventure**         → Hiking, surfing, wildlife
🛡  **Safety**            → Solo traveler tips

What would you like to know about Sri Lanka? 🌴
"""

# Emergency response
EMERGENCY_RESPONSE = """
🚨 EMERGENCY CONTACTS — SRI LANKA

Tourist Police    : 1912
General Emergency : 119
Ambulance         : 110
Fire Brigade      : 111

Tourist Police HQ : +94 11 2 421052
Colombo Hospital  : +94 11 2 691111
Kandy Hospital    : +94 81 2 222261

Stay calm. Help is available. 🙏
"""


class SmartPipeline:
    """
    Master pipeline connecting all agents.

    Flow:
    Message → NLP → Router → Specialized Agent
            → Validator → MongoDB → Response
    """

    def __init__(self):
        self.memories = {}
        print("✅ Smart Pipeline with Agents ready!")

    # ── Memory Management ──

    def get_memory(self, session_id: str) -> list:
        if session_id not in self.memories:
            self.memories[session_id] = []
        return self.memories[session_id]

    def update_memory(
        self,
        session_id: str,
        user_msg: str,
        ai_msg: str
    ):
        memory = self.get_memory(session_id)
        memory.append(HumanMessage(content=user_msg))
        memory.append(AIMessage(content=ai_msg))
        if len(memory) > 8:
            self.memories[session_id] = memory[-8:]

    def format_history(self, session_id: str) -> str:
        memory = self.get_memory(session_id)
        if not memory:
            return "No previous conversation."
        history = []
        for msg in memory:
            if isinstance(msg, HumanMessage):
                history.append(f"Traveler: {msg.content}")
            else:
                history.append(f"Serendib AI: {msg.content}")
        return "\n".join(history)

    # ── Main Process ──

    async def process(
        self,
        session_id: str,
        user_message: str,
        conversation_id: str = None
    ) -> dict:

        # Step 1 — NLP Analysis
        nlp_result = nlp_analyzer.analyze(user_message)
        intent = nlp_result["intent"]
        history = self.format_history(session_id)

        # Step 2 — Save user message
        if conversation_id:
            mongodb.save_message(
                conversation_id=conversation_id,
                role="user",
                content=user_message,
                intent=intent
            )

        # Step 3 — Update user profile
        entities = nlp_result.get("entities", {})
        profile_update = {}
        if entities.get("budget_level"):
            profile_update["budget_level"] = entities["budget_level"]
        if entities.get("locations"):
            profile_update["last_location"] = entities["locations"][0]
        if entities.get("duration"):
            profile_update["trip_duration"] = entities["duration"]
        if profile_update:
            mongodb.update_user_profile(session_id, profile_update)

        # Step 4 — Route to correct agent
        routing = router_agent.explain(nlp_result)
        agent_name = routing["agent"]

        print(f"\n🔀 Router → {agent_name} agent")
        print(f"   Intent : {intent}")
        print(f"   Reason : {routing['reason']}")

        # Step 5 — Handle with correct agent
        context_used = ""

        if agent_name == "greeting":
            answer = GREETING_RESPONSE
            sources = []

        elif agent_name == "emergency":
            answer = EMERGENCY_RESPONSE
            sources = []

        elif agent_name == "itinerary":
            result = await itinerary_agent.process(
                query=user_message,
                nlp_result=nlp_result,
                history=history,
                session_id=session_id
            )
            answer = result["answer"]
            sources = result["sources"]
            context_used = result.get("context_used", "")

        elif agent_name == "weather":
            result = await weather_agent.process(
                query=user_message,
                nlp_result=nlp_result,
                history=history
            )
            answer = result["answer"]
            sources = result["sources"]

        elif agent_name == "budget":
            result = await budget_agent.process(
                query=user_message,
                nlp_result=nlp_result,
                history=history
            )
            answer = result["answer"]
            sources = result["sources"]
            context_used = result.get("context_used", "")

        else:
            # Default → Knowledge Agent
            category = INTENT_TO_CATEGORY.get(intent)
            knowledge_agent._current_session = session_id
            result = await knowledge_agent.process(
                query=user_message,
                nlp_result=nlp_result,
                history=history,
                category=category
            )
            answer = result["answer"]
            sources = result["sources"]
            context_used = result.get("context_used", "")

        # Step 6 — Validate response
        if context_used and agent_name not in ["greeting", "emergency"]:
            final_answer, validation = validator_agent.apply(
                question=user_message,
                response=answer,
                context=context_used
            )
            print(f"   Validator: passed={validation.get('passed')} "
                  f"score={validation.get('confidence_score')}")
        else:
            final_answer = answer
            validation = {"passed": True, "confidence_score": 1.0}

        # Step 7 — Save response to MongoDB
        if conversation_id:
            mongodb.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content=final_answer
            )

        # Step 8 — Update memory
        self.update_memory(session_id, user_message, final_answer)

        return {
            "answer": final_answer,
            "intent": intent,
            "agent_used": agent_name,
            "entities": nlp_result.get("entities"),
            "sentiment": nlp_result.get("sentiment"),
            "user_state": nlp_result.get("user_state"),
            "sources": sources,
            "validation": validation
        }


smart_pipeline = SmartPipeline()