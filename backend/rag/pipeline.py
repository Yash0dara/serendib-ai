import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from backend.rag.retrieval import SerendibRetriever

load_dotenv()

# CONFIGURATION
GROQ_MODEL = "llama-3.1-8b-instant"
MAX_HISTORY = 6  # Keep last 6 messages in memory


# SYSTEM PROMPT
SYSTEM_PROMPT = """
You are Serendib AI, an expert Sri Lanka travel assistant.

Your personality:
- Friendly, warm and enthusiastic about Sri Lanka
- Knowledgeable but never condescending
- Practical and specific in your advice
- Honest when you are not sure about something

Your rules:
- ONLY answer questions related to Sri Lanka travel
- ALWAYS base your answers on the provided context
- If context does not have the answer say:
  "I don't have specific information about that.
   I recommend checking srilanka.travel for accurate details."
- NEVER make up prices, timings or facts
- Keep answers clear and well structured
- Use line breaks to make answers easy to read
- When giving itineraries be specific about times and costs

Context from knowledge base:
{context}

Conversation history is provided above each message.
Use it to maintain continuity in the conversation.
"""


# RAG PIPELINE CLASS
class SerendibRAGPipeline:
    """
    Full RAG pipeline for Serendib AI.

    Flow:
    User Question
         ↓
    Search ChromaDB (Retriever)
         ↓
    Get Relevant Chunks (Context)
         ↓
    Send to Groq LLM with Context + History
         ↓
    Get Grounded Answer
         ↓
    Save to Memory
         ↓
    Return Answer to User
    """

    def __init__(self):
        # Initialize retriever
        self.retriever = SerendibRetriever()

        # Initialize LLM
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=GROQ_MODEL,
            temperature=0.7,
            max_tokens=1024
        )

        # Conversation memories per session
        # session_id → list of messages
        self.memories = {}

        print("✅ Serendib RAG Pipeline ready!")

    # MEMORY MANAGEMENT
    def get_memory(self, session_id: str) -> list:
        """Gets conversation history for a session."""
        if session_id not in self.memories:
            self.memories[session_id] = []
        return self.memories[session_id]

    def save_to_memory(
        self,
        session_id: str,
        user_message: str,
        ai_response: str
    ):
        """Saves message pair to memory."""
        memory = self.get_memory(session_id)

        memory.append(HumanMessage(content=user_message))
        memory.append(AIMessage(content=ai_response))

        # Keep only last MAX_HISTORY messages
        if len(memory) > MAX_HISTORY:
            self.memories[session_id] = memory[-MAX_HISTORY:]

    def format_history(self, session_id: str) -> str:
        """Formats conversation history as readable text."""
        memory = self.get_memory(session_id)

        if not memory:
            return "No previous conversation."

        history_text = []
        for msg in memory:
            if isinstance(msg, HumanMessage):
                history_text.append(f"Traveler: {msg.content}")
            else:
                history_text.append(f"Serendib AI: {msg.content}")

        return "\n".join(history_text)

    # MAIN CHAT FUNCTION
    async def chat(
        self,
        session_id: str,
        user_message: str,
        traveler_type: str = None,
        category: str = None
    ) -> dict:
        """
        Main function that processes user message
        and returns AI response.

        Returns:
        {
            "answer": "The actual response",
            "sources": ["source1", "source2"],
            "context_used": "What context was retrieved"
        }
        """

        # Step 1 — Search knowledge base
        results = self.retriever.smart_search(
            query=user_message,
            traveler_type=traveler_type,
            category=category,
            k=4
        )

        # Step 2 — Format context from results
        context = self.retriever.format_context(results)

        # Step 3 — Get conversation history
        history = self.format_history(session_id)

        # Step 4 — Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", f"""
Conversation History:
{history}

Current Question:
{user_message}
""")
        ])

        # Step 5 — Send to LLM
        chain = prompt | self.llm
        response = chain.invoke({"context": context})
        answer = response.content

        # Step 6 — Save to memory
        self.save_to_memory(session_id, user_message, answer)

        # Step 7 — Extract sources used
        sources = list(set([
            doc.metadata.get("topic", "Sri Lanka")
            for doc in results
        ]))

        return {
            "answer": answer,
            "sources": sources,
            "context_used": context
        }


    # CLEAR SESSION
    def clear_session(self, session_id: str):
        """Clears conversation memory for a session."""
        if session_id in self.memories:
            del self.memories[session_id]
            print(f"✅ Session {session_id} cleared.")



# SINGLE INSTANCE
rag_pipeline = SerendibRAGPipeline()