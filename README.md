# 🌴 Serendib AI — Sri Lanka Travel Assistant

A conversational AI assistant that helps travelers plan and experience Sri Lanka through natural conversation. Built to learn and apply LLM, RAG, NLP, and multi-agent system design in a real-world project.

---

## What You Can Ask

- "Plan a 5 day trip to Ella on a budget"
- "What food should I try in Kandy?"
- "How do I get from Colombo to Ella by train?"
- "Is it safe to travel alone in Sri Lanka?"
- "What is the weather like in Galle right now?"

---

## Tech Stack

- Python, FastAPI
- Groq LLaMA 3.1 (LLM)
- LangChain + ChromaDB (RAG)
- HuggingFace Transformers (NLP)
- MongoDB Atlas
- OpenWeatherMap API
- HTML, CSS, JavaScript

---

## Getting Started

Clone the repo

    git clone https://github.com/YOUR_USERNAME/serendib-ai.git
    cd serendib-ai

Create virtual environment

    python -m venv venv
    venv\Scripts\activate

Install dependencies

    pip install -r requirements.txt

Set up your keys — copy .env.example to .env and fill in

    GROQ_API_KEY=your_key
    OPENWEATHER_API_KEY=your_key
    MONGODB_URI=your_mongodb_string

Free keys from console.groq.com, openweathermap.org, mongodb.com/atlas

Load knowledge base (run once)

    python backend/rag/ingestion.py

Start the app

    uvicorn backend.main:app --reload --port 8000

Open http://localhost:8000/ui to chat
Open http://localhost:8000/docs for API docs

---

## Run Tests

    python evaluation/evaluate.py

---

## Evaluation

- Intent Detection — 95%
- RAG Retrieval — 100%
- Safety Guardrails — 100%
- Overall Score — 98.8%

---

Built as a self-learning project. All tools used are free.