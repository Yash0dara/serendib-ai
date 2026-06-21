# Specialized agent for building travel itineraries

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from backend.rag.retrieval import SerendibRetriever
from backend.database.mongodb import mongodb
from dotenv import load_dotenv

load_dotenv()

ITINERARY_PROMPT = """
You are Serendib AI, a specialist Sri Lanka trip planner.

Rules:
- Build realistic day by day itineraries
- Consider travel time between locations
- Group nearby attractions together
- Include morning afternoon evening structure
- Add estimated costs for each day
- Consider traveler type and budget
- Mention best transport between locations
- Add practical tips for each location
- Never recommend unreachable combinations in one day

Traveler profile:
- Type     : {traveler_type}
- Budget   : {budget}
- Duration : {duration}
- Interests: {interests}
- Locations: {locations}

Knowledge base context:
{context}

Conversation history:
{history}

Format each day like this:
DAY X — [Location]
Morning   : Activity (time, cost)
Afternoon : Activity (time, cost)
Evening   : Activity + dinner (cost)
Stay      : Area + accommodation type (cost range)
Transport : How to get to next location
Daily Cost: Estimated total
"""


class ItineraryAgent:
    """
    Specialized agent for building
    detailed day by day travel plans.
    """

    def __init__(self):
        self.retriever = SerendibRetriever()
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.8,
            max_tokens=2048
        )

    def get_traveler_interests(
        self,
        traveler_type: str
    ) -> str:
        """Maps traveler type to interests."""
        interests = {
            "solo": "flexible schedule, budget conscious, authentic experiences",
            "cultural": "temples, history, festivals, local traditions",
            "adventure": "hiking, wildlife, surfing, outdoor activities",
        }
        return interests.get(
            traveler_type,
            "general sightseeing and local experiences"
        )

    def build_search_queries(
        self,
        entities: dict,
        traveler_type: str
    ) -> list:
        """
        Builds multiple search queries to
        retrieve comprehensive itinerary context.
        """
        queries = []
        locations = entities.get("locations", [])
        duration = entities.get("duration", "")

        # Base query
        queries.append(
            f"Sri Lanka itinerary {duration} {traveler_type} traveler"
        )

        # Location specific queries
        for location in locations:
            queries.append(
                f"things to do in {location} Sri Lanka"
            )

        # Budget specific
        if entities.get("budget_level") == "low":
            queries.append("budget travel tips Sri Lanka")
        elif entities.get("budget_level") == "high":
            queries.append("luxury experiences Sri Lanka")

        return queries

    async def process(
        self,
        query: str,
        nlp_result: dict,
        history: str,
        session_id: str = None
    ) -> dict:
        """
        Builds a personalized itinerary.
        """
        entities = nlp_result.get("entities", {})
        locations = entities.get("locations", [])
        duration = entities.get("duration", "7 days")
        budget = entities.get("budget_level", "mid")

        # Get user profile for traveler type
        traveler_type = "solo"
        if session_id:
            user = mongodb.get_user(session_id)
            if user:
                traveler_type = user.get(
                    "traveler_type",
                    "solo"
                )

        interests = self.get_traveler_interests(traveler_type)

        # Build multiple search queries for rich context
        search_queries = self.build_search_queries(
            entities,
            traveler_type
        )

        # Retrieve context from multiple queries
        all_results = []
        for sq in search_queries[:3]:
            results = self.retriever.search(sq, k=3)
            all_results.extend(results)

        # Remove duplicates by content
        seen = set()
        unique_results = []
        for doc in all_results:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(doc)

        context = self.retriever.format_context(
            unique_results[:6]
        )

        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", ITINERARY_PROMPT),
            ("human", "{query}")
        ])

        chain = prompt | self.llm

        response = chain.invoke({
            "traveler_type": traveler_type,
            "budget": budget,
            "duration": duration,
            "interests": interests,
            "locations": ", ".join(locations) or "flexible",
            "context": context,
            "history": history,
            "query": query
        })

        # Save itinerary to MongoDB
        if session_id:
            mongodb.save_itinerary(
                session_id=session_id,
                itinerary={
                    "raw_plan": response.content,
                    "duration_days": duration,
                    "traveler_type": traveler_type,
                    "locations": locations,
                    "budget_level": budget
                }
            )

        sources = list(set([
            doc.metadata.get("topic", "Sri Lanka")
            for doc in unique_results
        ]))

        return {
            "answer": response.content,
            "sources": sources,
            "agent": "itinerary",
            "itinerary_saved": True
        }


itinerary_agent = ItineraryAgent()