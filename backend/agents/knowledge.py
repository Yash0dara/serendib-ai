# Handles general travel knowledge using RAG
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from backend.rag.retrieval import SerendibRetriever
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_PROMPT = """
You are Serendib AI, a Sri Lanka travel expert.

Tone: {tone}
User state: {user_state_note}

Rules:
- Answer ONLY from provided context
- Be specific with names, prices, times
- Use clear formatting
- If unsure say "Please verify at srilanka.travel"
- Never make up facts

Traveler profile:
- Budget : {budget}
- Locations of interest: {locations}

Context from knowledge base:
{context}

Conversation history:
{history}
"""


class KnowledgeAgent:
    """
    Answers general travel questions using RAG.
    Handles: food, transport, culture,
             adventure, safety queries.
    """

    def __init__(self):
        self.retriever = SerendibRetriever()
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1024
        )

    def get_tone(self, sentiment: str, user_state: str) -> tuple:
        """Returns tone instruction and user state note."""

        tones = {
            "positive": "Be warm enthusiastic and match their energy.",
            "negative": "Be empathetic acknowledge feelings then help.",
            "neutral":  "Be clear direct and practical."
        }

        state_notes = {
            "frustrated": "Start with empathy. Acknowledge frustration first.",
            None: ""
        }

        tone = tones.get(sentiment, tones["neutral"])
        state_note = state_notes.get(user_state, "")

        return tone, state_note

    async def process(
        self,
        query: str,
        nlp_result: dict,
        history: str,
        category: str = None
    ) -> dict:
        """
        Processes knowledge query through RAG + LLM.
        """
        entities = nlp_result.get("entities", {})
        sentiment = nlp_result.get("sentiment", "neutral")
        user_state = nlp_result.get("user_state")
        locations = entities.get("locations", [])

        # Build enhanced query
        enhanced_query = query
        if locations:
            enhanced_query += f" {' '.join(locations)}"

        # RAG retrieval
        results = self.retriever.smart_search(
            query=enhanced_query,
            category=category,
            k=4
        )

        if not results:
            results = self.retriever.search(query, k=4)

        context = self.retriever.format_context(results)

        # Get tone
        tone, state_note = self.get_tone(sentiment, user_state)

        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", KNOWLEDGE_PROMPT),
            ("human", "{query}")
        ])

        chain = prompt | self.llm

        response = chain.invoke({
            "tone": tone,
            "user_state_note": state_note,
            "budget": entities.get("budget_level", "not specified"),
            "locations": ", ".join(locations) or "not specified",
            "context": context,
            "history": history,
            "query": query
        })

        sources = list(set([
            doc.metadata.get("topic", "Sri Lanka")
            for doc in results
        ]))

        return {
            "answer": response.content,
            "sources": sources,
            "agent": "knowledge",
            "context_used": context
        }


knowledge_agent = KnowledgeAgent()