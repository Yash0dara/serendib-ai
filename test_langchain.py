# test_langchain.py

from dotenv import load_dotenv
import os

load_dotenv()

print("=== LangChain Connection Test ===\n")

# ================================
# TEST 1 - Groq LLM via LangChain
# ================================

print("⏳ Test 1 - Groq LLM via LangChain...")

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.7
)

messages = [
    SystemMessage(content="You are a Sri Lanka travel assistant."),
    HumanMessage(content="Name one famous place in Sri Lanka in one sentence.")
]

response = llm.invoke(messages)
print(f"✅ Groq via LangChain works!")
print(f"   Response: {response.content}\n")


# ================================
# TEST 2 - Embeddings via LangChain
# ================================

print("⏳ Test 2 - HuggingFace Embeddings via LangChain...")

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

test_text = "Best beaches in Sri Lanka"
embedding_result = embeddings.embed_query(test_text)

print(f"✅ Embeddings via LangChain works!")
print(f"   Text     : {test_text}")
print(f"   Size     : {len(embedding_result)} dimensions\n")


# ================================
# TEST 3 - ChromaDB via LangChain
# ================================

print("⏳ Test 3 - ChromaDB via LangChain...")

from langchain_chroma import Chroma

# Create a test collection
vector_store = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add sample documents
from langchain_core.documents import Document

sample_docs = [
    Document(
        page_content="Sigiriya is an ancient rock fortress in Sri Lanka rising 200 meters high.",
        metadata={"category": "destination", "location": "Sigiriya"}
    ),
    Document(
        page_content="Ella is famous for Nine Arch Bridge and tea plantations in Sri Lanka.",
        metadata={"category": "destination", "location": "Ella"}
    ),
    Document(
        page_content="Sri Lankan food includes rice and curry, hoppers and kottu roti.",
        metadata={"category": "food", "location": "Sri Lanka"}
    ),
]

vector_store.add_documents(sample_docs)

# Search
results = vector_store.similarity_search(
    "What is there to see in the hills?",
    k=2
)

print(f"✅ ChromaDB via LangChain works!")
print(f"   Query  : 'What is there to see in the hills?'")
print(f"   Results:")
for i, doc in enumerate(results):
    print(f"   {i+1}. {doc.page_content[:80]}...")
print()

# ================================
# TEST 4 - Conversation Memory
# ================================

print("⏳ Test 4 - Conversation Memory via LangChain...")

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Store for session histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Build prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Sri Lanka travel assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Build chain
chain = prompt | llm

# Wrap with memory
with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# First message
response1 = with_memory.invoke(
    {"input": "I am planning to visit Sri Lanka for 5 days."},
    config={"configurable": {"session_id": "test123"}}
)
print(f"   User : I am planning to visit Sri Lanka for 5 days.")
print(f"   Bot  : {response1.content[:100]}...")

# Second message (tests memory)
response2 = with_memory.invoke(
    {"input": "How many days did I say I was visiting for?"},
    config={"configurable": {"session_id": "test123"}}
)
print(f"\n   User : How many days did I say I was visiting for?")
print(f"   Bot  : {response2.content[:100]}...")
print(f"\n✅ Memory via LangChain works!\n")


# ================================
# FINAL SUMMARY
# ================================

print("=" * 40)
print("✅ ALL LANGCHAIN TESTS PASSED!")
print("=" * 40)
print("✅ Groq LLM      — Connected")
print("✅ Embeddings    — Connected")
print("✅ ChromaDB      — Connected")
print("✅ Memory        — Connected")
print("\n🚀 Ready to build RAG system!")