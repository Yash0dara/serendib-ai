# backend/core/smart_pipeline.py

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
from backend.guardrails.safety import safety_guard

load_dotenv()

# ── Sri Lanka Words To Protect ──
SL_WORDS = [
    "ella", "kandy", "galle", "colombo", "sigiriya",
    "kottu", "hopper", "hoppers", "tuk-tuk", "tuk", "negombo",
    "mirissa", "trincomalee", "arugam", "nuwara",
    "anuradhapura", "polonnaruwa", "dambulla",
    "serendib", "sinhala", "perahera", "poya",
    "yala", "minneriya", "wilpattu", "udawalawe",
    "hikkaduwa", "bentota", "unawatuna",
    "kiribath", "lamprais", "watalappan", "isso",
    "lkr", "rupee", "rupees"
]

# ── Sri Lanka Place Corrections ──
SL_CORRECTIONS = {
    "kandi": "kandy",
    "kandey": "kandy",
    "colmbo": "colombo",
    "ellla": "ella",
    "elle": "ella",
    "ell": "ella",
    "sigriya": "sigiriya",
    "sigirya": "sigiriya",
    "sigiria": "sigiriya",
    "galleh": "galle",
    "mirisa": "mirissa",
    "trinco": "trincomalee",
    "dambula": "dambulla",
    "anuradapura": "anuradhapura",
    "polanaruwa": "polonnaruwa",
    "negomba": "negombo",
    "hikadua": "hikkaduwa",
    "arugambay": "arugam bay",
    "nuwaraeliya": "nuwara eliya",
    "nuwara": "nuwara eliya",
    "yalla": "yala",
}

# ── Common English Corrections ──
COMMON_CORRECTIONS = {
    # Transport
    "trian": "train",
    "trane": "train",
    "tarin": "train",
    "trainn": "train",
    "trein": "train",
    "buss": "bus",
    "buus": "bus",
    "trnsport": "transport",
    "tranport": "transport",
    "taxxi": "taxi",
    "tukuk": "tuk tuk",
    "ttuk": "tuk tuk",

    # Activities
    "hikin": "hiking",
    "hikng": "hiking",
    "hikking": "hiking",
    "hikig": "hiking",
    "hik": "hike",
    "surfng": "surfing",
    "surffing": "surfing",
    "trekng": "trekking",
    "trkking": "trekking",
    "trking": "trekking",
    "trk": "trek",
    "safarri": "safari",
    "safri": "safari",
    "safrai": "safari",
    "safary": "safari",
    "swaiming": "swimming",
    "swimng": "swimming",
    "divng": "diving",
    "climing": "climbing",
    "clmbing": "climbing",
    "campng": "camping",

    # Places
    "beech": "beach",
    "bech": "beach",
    "mountan": "mountain",
    "monutain": "mountain",
    "forrest": "forest",
    "templ": "temple",
    "tempel": "temple",
    "templle": "temple",
    "musem": "museum",
    "musuem": "museum",

    # Food
    "foood": "food",
    "fod": "food",
    "restarant": "restaurant",
    "resturant": "restaurant",
    "restaurnt": "restaurant",
    "cuisne": "cuisine",
    "cuiisne": "cuisine",

    # Question words
    "wht": "what",
    "whats": "what is",
    "hw": "how",
    "hwo": "how",
    "wher": "where",
    "whre": "where",
    "wen": "when",
    "whne": "when",

    # Common words
    "iz": "is",
    "saff": "safe",
    "safty": "safety",
    "safley": "safely",
    "savety": "safety",
    "travl": "travel",
    "travle": "travel",
    "trvl": "travel",
    "travaling": "traveling",
    "travling": "traveling",
    "visti": "visit",
    "vsit": "visit",
    "visitting": "visiting",
    "palce": "place",
    "plaec": "place",
    "plce": "place",
    "alon": "alone",
    "alne": "alone",
    "alonne": "alone",
    "tri": "try",
    "shud": "should",
    "shoud": "should",
    "shold": "should",
    "coud": "could",
    "woud": "would",
    "cna": "can",
    "pleas": "please",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "thnks": "thanks",
    "recmmend": "recommend",
    "recomend": "recommend",
    "recomnd": "recommend",
    "recommnd": "recommend",
    "priice": "price",
    "priec": "price",
    "pric": "price",
    "budjet": "budget",
    "budge": "budget",
    "buget": "budget",
    "schedul": "schedule",
    "schdule": "schedule",
    "informtion": "information",
    "infomation": "information",
    "advce": "advice",
    "adviice": "advice",
    "helpfull": "helpful",
    "usefull": "useful",
    "beautifull": "beautiful",
    "wonderfull": "wonderful",
    "best": "best",
    "shud": "should",
}

# ── Intent → RAG Category ──
INTENT_TO_CATEGORY = {
    "food_recommendation": "food",
    "transport_query":     "transport",
    "cultural_info":       "culture",
    "adventure_activity":  "adventure",
    "safety_query":        "practical",
    "general_question":    None
}

# ── Static Responses ──
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
    Message → Spell Check → Safety → NLP → Router
            → Agent → Validator → MongoDB → Response
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

    # ── Dictionary Based Spell Correction ──

    def correct_spelling(self, text: str) -> str:
        """
        Dictionary based spell correction.
        No external libraries for word correction.
        Reliable and predictable results.

        Priority order per word:
        1. SL place corrections
        2. Common English corrections
        3. SL word protection
        4. Keep as is
        """
        try:
            if not text or len(text.strip()) < 2:
                return text

            words = text.split()
            corrected_words = []

            for word in words:
                # Preserve trailing punctuation
                stripped = word.strip(".,!?")
                punct_after = word[len(stripped):]
                word_lower = stripped.lower()
                final_word = stripped

                # Priority 1 — SL place corrections
                if word_lower in SL_CORRECTIONS:
                    final_word = SL_CORRECTIONS[word_lower]
                    if final_word != stripped:
                        print(
                            f"   SL fix: '{stripped}' "
                            f"→ '{final_word}'"
                        )

                # Priority 2 — Common English corrections
                elif word_lower in COMMON_CORRECTIONS:
                    final_word = COMMON_CORRECTIONS[word_lower]
                    if final_word != stripped:
                        print(
                            f"   Fix: '{stripped}' "
                            f"→ '{final_word}'"
                        )

                # Priority 3 — Protect SL words
                elif word_lower in SL_WORDS:
                    final_word = stripped

                # Priority 4 — Keep as is
                else:
                    final_word = stripped

                corrected_words.append(final_word + punct_after)

            corrected_text = " ".join(corrected_words)

            if corrected_text.lower() != text.lower():
                print(f"   Corrected: '{text}' → '{corrected_text}'")

            return corrected_text

        except Exception as e:
            print(f"   Spell check error: {e}")
            return text

    # ── Main Process ──

    async def process(
        self,
        session_id: str,
        user_message: str,
        conversation_id: str = None
    ) -> dict:

        # ── Step 0: Spell Correction ──
        corrected_message = self.correct_spelling(user_message)

        # ── Step 1: Safety Check ──
        safety_result = safety_guard.check(corrected_message)

        if not safety_result["safe"]:
            answer = safety_result["response"]

            if conversation_id:
                mongodb.save_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=corrected_message,
                    intent=safety_result["reason"]
                )
                mongodb.save_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=answer
                )

            self.update_memory(
                session_id,
                corrected_message,
                answer
            )

            return {
                "answer": answer,
                "intent": safety_result["reason"],
                "agent_used": "safety_guard",
                "entities": {},
                "sentiment": "neutral",
                "user_state": None,
                "sources": [],
                "validation": {
                    "passed": True,
                    "confidence_score": 1.0
                }
            }

        # ── Step 2: NLP Analysis ──
        nlp_result = nlp_analyzer.analyze(corrected_message)
        intent = nlp_result["intent"]
        history = self.format_history(session_id)

        # ── Step 3: Save User Message ──
        if conversation_id:
            mongodb.save_message(
                conversation_id=conversation_id,
                role="user",
                content=corrected_message,
                intent=intent
            )

        # ── Step 4: Update User Profile ──
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

        # ── Step 5: Route To Agent ──
        routing = router_agent.explain(nlp_result)
        agent_name = routing["agent"]

        print(f"\n🔀 Router → {agent_name} agent")
        print(f"   Intent : {intent}")
        print(f"   Reason : {routing['reason']}")

        # ── Step 6: Handle With Agent ──
        context_used = ""

        if agent_name == "greeting":
            answer = GREETING_RESPONSE
            sources = []

        elif agent_name == "emergency":
            answer = EMERGENCY_RESPONSE
            sources = []

        elif agent_name == "itinerary":
            result = await itinerary_agent.process(
                query=corrected_message,
                nlp_result=nlp_result,
                history=history,
                session_id=session_id
            )
            answer = result["answer"]
            sources = result["sources"]
            context_used = result.get("context_used", "")

        elif agent_name == "weather":
            result = await weather_agent.process(
                query=corrected_message,
                nlp_result=nlp_result,
                history=history
            )
            answer = result["answer"]
            sources = result["sources"]

        elif agent_name == "budget":
            result = await budget_agent.process(
                query=corrected_message,
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
                query=corrected_message,
                nlp_result=nlp_result,
                history=history,
                category=category
            )
            answer = result["answer"]
            sources = result["sources"]
            context_used = result.get("context_used", "")

        # ── Step 7: Validate Response ──
        if context_used and agent_name not in ["greeting", "emergency"]:
            final_answer, validation = validator_agent.apply(
                question=corrected_message,
                response=answer,
                context=context_used
            )
            print(
                f"   Validator: passed={validation.get('passed')} "
                f"score={validation.get('confidence_score')}"
            )
        else:
            final_answer = answer
            validation = {"passed": True, "confidence_score": 1.0}

        # ── Step 8: Save Response ──
        if conversation_id:
            mongodb.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content=final_answer
            )

        # ── Step 9: Update Memory ──
        self.update_memory(
            session_id,
            corrected_message,
            final_answer
        )

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


# Single instance
smart_pipeline = SmartPipeline()