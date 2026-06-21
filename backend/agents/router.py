# Decides which agent handles each user request

# Intent → Agent mapping
INTENT_AGENT_MAP = {
    "trip_planning":       "itinerary",
    "food_recommendation": "knowledge",
    "transport_query":     "knowledge",
    "budget_query":        "budget",
    "weather_query":       "weather",
    "cultural_info":       "knowledge",
    "adventure_activity":  "knowledge",
    "safety_query":        "knowledge",
    "emergency":           "emergency",
    "greeting":            "greeting",
    "general_question":    "knowledge"
}


class RouterAgent:
    """
    Reads NLP result and decides which
    specialized agent should handle the request.

    Think of this as the receptionist
    who directs patients to the right doctor.
    """

    def route(self, nlp_result: dict) -> str:
        """
        Returns agent name based on intent.
        """
        intent = nlp_result.get("intent", "general_question")
        agent = INTENT_AGENT_MAP.get(intent, "knowledge")

        # Override based on secondary intents
        secondary = nlp_result.get("secondary_intents", [])

        # If primary is trip_planning but has weather keywords
        # → still use itinerary but flag weather needed
        if intent == "trip_planning" and "weather_query" in secondary:
            return "itinerary"

        # If frustrated user → knowledge agent with empathy
        if nlp_result.get("user_state") == "frustrated":
            if agent == "knowledge":
                return "knowledge"

        return agent

    def explain(self, nlp_result: dict) -> dict:
        """
        Returns full routing decision with explanation.
        Useful for debugging and logging.
        """
        intent = nlp_result.get("intent")
        agent = self.route(nlp_result)
        entities = nlp_result.get("entities", {})

        return {
            "agent": agent,
            "reason": f"Intent '{intent}' routed to '{agent}' agent",
            "locations": entities.get("locations", []),
            "duration": entities.get("duration"),
            "budget": entities.get("budget_level"),
            "user_state": nlp_result.get("user_state")
        }


router_agent = RouterAgent()