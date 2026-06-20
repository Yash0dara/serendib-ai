# backend/core/smart_pipeline.py

import os
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from backend.nlp.analyzer import nlp_analyzer
from backend.rag.retrieval import SerendibRetriever
from backend.database.mongodb import mongodb

load_dotenv()

# ============================================
# INTENT → RAG CATEGORY MAPPING
# ============================================

INTENT_TO_CATEGORY = {
    "trip_planning":        "itinerary",
    "food_recommendation":  "food",
    "transport_query":      "transport",
    "budget_query":         "practical",
    "weather_query":        "practical",
    "cultural_info":        "culture",
    "adventure_activity":   "adventure",
    "safety_query":         "practical",
    "emergency":            "practical",
    "greeting":             None,
    "general_question":     None
}

# ============================================
# TONE INSTRUCTIONS BASED ON SENTIMENT
# ============================================

TONE_INSTRUCTIONS = {
    "positive": """
User is excited and enthusiastic.
Match their energy.
Be warm and encouraging.
Add interesting facts to excite them more.
""",
    "negative": """
User seems unhappy or disappointed.
Be empathetic and understanding first.
Acknowledge their frustration.
Then provide helpful solution.
""",
    "neutral": """
User is calm and informational.
Be clear, direct and practical.
Focus on facts and specifics.
"""
}

# ============================================
# USER STATE INSTRUCTIONS
# ============================================

USER_STATE_INSTRUCTIONS = {
    "frustrated": """
IMPORTANT: User is frustrated.
Start response by acknowledging this.
Example: "I understand that's frustrating..."
Be extra helpful and offer alternatives.
""",
    None: ""
}

# ============================================
# SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = """
You are Serendib AI, an expert Sri Lanka travel assistant.

Tone instruction:
{tone}

User state instruction:
{user_state_instruction}

Your rules:
- ONLY answer Sri Lanka travel questions
- Base answers on provided context
- Be specific with prices, times, locations
- Use clear formatting with line breaks
- If context missing say:
  "I don't have that specific detail.
   Please check srilanka.travel for accuracy."
- Never make up facts or prices

Traveler profile:
- Type     : {traveler_type}
- Budget   : {budget_level}
- Location : {locations}
- Duration : {duration}

Retrieved knowledge:
{context}
"""


# ============================================
# SMART PIPELINE CLASS
# ============================================

class SmartPipeline:
    """
    Connects NLP + RAG + LLM into one
    intelligent travel assistant pipeline.

    Every user message goes through:
    1. NLP analysis
    2. Smart RAG retrieval
    3. Personalized LLM response
    4. MongoDB storage
    5. Memory management
    """

    def __init__(self):
        self.retriever = SerendibRetriever()
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1024
        )
        # session_id → conversation history
        self.memories = {}
        print("✅ Smart Pipeline ready!")

    # ============================================
    # MEMORY MANAGEMENT
    # ============================================

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

        # Keep last 8 messages only
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

    # ============================================
    # USER PROFILE MANAGEMENT
    # ============================================

    def update_user_profile(
        self,
        session_id: str,
        nlp_result: dict
    ):
        """
        Updates MongoDB user profile
        based on NLP extracted entities.
        """
        entities = nlp_result["entities"]
        profile_update = {}

        if entities.get("budget_level"):
            profile_update["budget_level"] = entities["budget_level"]

        if entities.get("locations"):
            profile_update["last_location"] = entities["locations"][0]

        if entities.get("duration"):
            profile_update["trip_duration"] = entities["duration"]

        if profile_update:
            mongodb.update_user_profile(session_id, profile_update)

    # ============================================
    # SMART RAG SEARCH
    # ============================================

    def smart_retrieve(
        self,
        query: str,
        nlp_result: dict
    ) -> tuple[list, str]:
        """
        Uses NLP results to search RAG smartly.

        Returns:
        - results: list of relevant documents
        - context: formatted string for LLM
        """
        intent = nlp_result["intent"]
        entities = nlp_result["entities"]
        locations = entities.get("locations", [])

        # Get RAG category from intent
        category = INTENT_TO_CATEGORY.get(intent)

        # Build enhanced query using entities
        enhanced_query = query

        if locations:
            location_str = " ".join(locations)
            enhanced_query = f"{query} {location_str}"

        if entities.get("duration"):
            enhanced_query += f" {entities['duration']}"

        # Search with category filter
        results = self.retriever.smart_search(
            query=enhanced_query,
            category=category,
            k=4
        )

        # If no results with filter → search without filter
        if not results:
            results = self.retriever.search(query, k=4)

        context = self.retriever.format_context(results)
        return results, context

    # ============================================
    # GREETING HANDLER
    # ============================================

    def handle_greeting(self, session_id: str) -> str:
        """
        Special handler for greeting intent.
        No RAG needed.
        """
        user = mongodb.get_user(session_id)
        name = user.get("name", "Traveler") if user else "Traveler"

        return f"""
Hello {name}! 👋 Welcome to Serendib AI.

I am your intelligent Sri Lanka travel assistant.

I can help you with:
🗺  **Trip Planning**     → Personalized itineraries
🍛  **Food Guide**        → Local cuisine and restaurants
🚂  **Transport**         → Trains, buses, tuk-tuks
💰  **Budget Planning**   → Daily cost estimates
🌤  **Weather Guide**     → Best time to visit
🏛  **Culture & History** → Temples, festivals, heritage
🏄  **Adventure**         → Hiking, surfing, wildlife safari
🛡  **Safety**            → Solo traveler safety tips

What would you like to know about Sri Lanka? 🌴
"""

    # ============================================
    # EMERGENCY HANDLER
    # ============================================

    def handle_emergency(self) -> str:
        """
        Special handler for emergency intent.
        Returns immediate help information.
        """
        return """
🚨 EMERGENCY CONTACTS — SRI LANKA

Tourist Police    : 1912
General Emergency : 119
Ambulance         : 110
Fire              : 111

Tourist Police Headquarters:
📍 Colombo — +94 11 2 421052

Hospital:
📍 Colombo National Hospital — +94 11 2 691111
📍 Kandy General Hospital    — +94 81 2 222261

Embassy contacts depend on your nationality.
Please contact your country's embassy immediately
if you need consular assistance.

Stay calm. Help is available. 🙏
"""

    # ============================================
    # MAIN PROCESS FUNCTION
    # ============================================

    async def process(
        self,
        session_id: str,
        user_message: str,
        conversation_id: str = None
    ) -> dict:
        """
        Main function that processes every user message.

        Returns:
        {
            "answer": str,
            "intent": str,
            "entities": dict,
            "sentiment": str,
            "sources": list,
            "user_state": str
        }
        """

        # ── Step 1: NLP Analysis ──
        nlp_result = nlp_analyzer.analyze(user_message)
        intent = nlp_result["intent"]
        sentiment = nlp_result["sentiment"]
        user_state = nlp_result.get("user_state")
        entities = nlp_result["entities"]

        # ── Step 2: Save user message to MongoDB ──
        if conversation_id:
            mongodb.save_message(
                conversation_id=conversation_id,
                role="user",
                content=user_message,
                intent=intent
            )

        # ── Step 3: Update user profile ──
        self.update_user_profile(session_id, nlp_result)

        # ── Step 4: Handle special intents ──
        if intent == "greeting":
            answer = self.handle_greeting(session_id)
            if conversation_id:
                mongodb.save_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=answer
                )
            self.update_memory(session_id, user_message, answer)
            return {
                "answer": answer,
                "intent": intent,
                "entities": entities,
                "sentiment": sentiment,
                "sources": [],
                "user_state": user_state
            }

        if intent == "emergency":
            answer = self.handle_emergency()
            if conversation_id:
                mongodb.save_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=answer
                )
            self.update_memory(session_id, user_message, answer)
            return {
                "answer": answer,
                "intent": intent,
                "entities": entities,
                "sentiment": sentiment,
                "sources": [],
                "user_state": user_state
            }

        # ── Step 5: Smart RAG Retrieval ──
        results, context = self.smart_retrieve(
            user_message,
            nlp_result
        )

        # ── Step 6: Get conversation history ──
        history = self.format_history(session_id)

        # ── Step 7: Get user profile ──
        user = mongodb.get_user(session_id)
        profile = user or {}

        # ── Step 8: Build personalized prompt ──
        tone = TONE_INSTRUCTIONS.get(sentiment, TONE_INSTRUCTIONS["neutral"])
        user_state_instruction = USER_STATE_INSTRUCTIONS.get(
            user_state,
            ""
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", f"""
Conversation History:
{history}

Current Question:
{user_message}
""")
        ])

        # ── Step 9: Generate LLM response ──
        chain = prompt | self.llm

        response = chain.invoke({
            "tone": tone,
            "user_state_instruction": user_state_instruction,
            "traveler_type": profile.get("traveler_type", "Not specified"),
            "budget_level": entities.get("budget_level") or profile.get("budget_level", "Not specified"),
            "locations": ", ".join(entities.get("locations", [])) or "Not specified",
            "duration": entities.get("duration") or profile.get("trip_duration", "Not specified"),
            "context": context
        })

        answer = response.content

        # ── Step 10: Save response to MongoDB ──
        if conversation_id:
            mongodb.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer
            )

        # ── Step 11: Update memory ──
        self.update_memory(session_id, user_message, answer)

        # ── Step 12: Extract sources ──
        sources = list(set([
            doc.metadata.get("topic", "Sri Lanka")
            for doc in results
        ]))

        return {
            "answer": answer,
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "sources": sources,
            "user_state": user_state
        }


# Single instance
smart_pipeline = SmartPipeline()