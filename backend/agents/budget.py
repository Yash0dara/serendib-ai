# Calculates and explains Sri Lanka travel costs

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from backend.rag.retrieval import SerendibRetriever
from dotenv import load_dotenv

load_dotenv()

# Daily cost estimates in LKR (Sri Lankan Rupee)
BUDGET_DATA = {
    "low": {
        "accommodation": {"min": 2000, "max": 4500,
                          "type": "hostels and budget guesthouses"},
        "food":          {"min": 800,  "max": 1500,
                          "type": "local restaurants and street food"},
        "transport":     {"min": 500,  "max": 1200,
                          "type": "public buses and 3rd class trains"},
        "activities":    {"min": 500,  "max": 1500,
                          "type": "free attractions and budget entry fees"},
        "total_per_day": {"min": 3800, "max": 8700}
    },
    "mid": {
        "accommodation": {"min": 5000,  "max": 12000,
                          "type": "mid-range hotels and guesthouses"},
        "food":          {"min": 1500,  "max": 3500,
                          "type": "mix of local and tourist restaurants"},
        "transport":     {"min": 1500,  "max": 4000,
                          "type": "private buses and 2nd class trains"},
        "activities":    {"min": 1500,  "max": 4000,
                          "type": "most attractions and guided tours"},
        "total_per_day": {"min": 9500,  "max": 23500}
    },
    "high": {
        "accommodation": {"min": 15000, "max": 50000,
                          "type": "boutique hotels and luxury resorts"},
        "food":          {"min": 4000,  "max": 10000,
                          "type": "upscale restaurants and fine dining"},
        "transport":     {"min": 5000,  "max": 15000,
                          "type": "private cars and taxis"},
        "activities":    {"min": 5000,  "max": 15000,
                          "type": "premium experiences and private tours"},
        "total_per_day": {"min": 29000, "max": 90000}
    }
}

# Entry fees for popular attractions (LKR)
ENTRY_FEES = {
    "Sigiriya Rock":              4500,
    "Temple of the Tooth":        1500,
    "Royal Botanical Gardens":    3000,
    "Yala National Park":         6000,
    "Anuradhapura Sacred City":   4000,
    "Dambulla Cave Temple":       1500,
    "Minneriya National Park":    5000,
    "Udawalawe National Park":    4500,
    "Galle Dutch Fort":           0,
    "Colombo National Museum":    500,
}

BUDGET_PROMPT = """
You are Serendib AI, a Sri Lanka budget planning expert.

Rules:
- Give specific cost breakdowns
- Use LKR (Sri Lankan Rupee) as primary currency
- Also mention USD equivalent (1 USD ≈ 300 LKR)
- Be honest about budget ranges
- Include money saving tips
- Mention which entry fees to expect
- Consider season pricing differences

Traveler budget level: {budget_level}

Calculated estimates:
{budget_breakdown}

Entry fees reference:
{entry_fees}

Knowledge base context:
{context}

Conversation history:
{history}
"""


class BudgetAgent:
    """
    Specialized agent for travel cost calculations
    and budget planning advice.
    """

    def __init__(self):
        self.retriever = SerendibRetriever()
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.4,
            max_tokens=1024
        )

    def calculate_budget(
        self,
        budget_level: str,
        duration_days: int
    ) -> dict:
        """
        Calculates total trip cost estimate.
        """
        level = BUDGET_DATA.get(budget_level, BUDGET_DATA["mid"])

        total_min = level["total_per_day"]["min"] * duration_days
        total_max = level["total_per_day"]["max"] * duration_days

        return {
            "budget_level": budget_level,
            "duration_days": duration_days,
            "daily": level,
            "total": {
                "min_lkr": total_min,
                "max_lkr": total_max,
                "min_usd": round(total_min / 300),
                "max_usd": round(total_max / 300)
            }
        }

    def format_budget_breakdown(
        self,
        calculation: dict
    ) -> str:
        """Formats budget data for LLM."""
        daily = calculation["daily"]
        total = calculation["total"]
        days = calculation["duration_days"]
        level = calculation["budget_level"].upper()

        breakdown = f"""
BUDGET LEVEL: {level}
TRIP DURATION: {days} days

DAILY BREAKDOWN:
Accommodation : {daily['accommodation']['min']:,} - {daily['accommodation']['max']:,} LKR ({daily['accommodation']['type']})
Food          : {daily['food']['min']:,} - {daily['food']['max']:,} LKR ({daily['food']['type']})
Transport     : {daily['transport']['min']:,} - {daily['transport']['max']:,} LKR ({daily['transport']['type']})
Activities    : {daily['activities']['min']:,} - {daily['activities']['max']:,} LKR ({daily['activities']['type']})
DAILY TOTAL   : {daily['total_per_day']['min']:,} - {daily['total_per_day']['max']:,} LKR

TOTAL TRIP ESTIMATE:
LKR : {total['min_lkr']:,} - {total['max_lkr']:,} LKR
USD : ${total['min_usd']} - ${total['max_usd']} USD
"""
        return breakdown

    def format_entry_fees(self) -> str:
        """Formats entry fees reference."""
        fees_text = "COMMON ENTRY FEES (LKR):\n"
        for attraction, fee in ENTRY_FEES.items():
            if fee == 0:
                fees_text += f"{attraction}: Free\n"
            else:
                fees_text += f"{attraction}: {fee:,} LKR\n"
        return fees_text

    def parse_duration(self, duration_str: str) -> int:
        """Converts duration string to days integer."""
        if not duration_str:
            return 7

        import re
        match = re.search(r'(\d+)', duration_str)
        if not match:
            return 7

        value = int(match.group(1))
        if "week" in duration_str.lower():
            return value * 7

        return value

    async def process(
        self,
        query: str,
        nlp_result: dict,
        history: str
    ) -> dict:
        """
        Calculates budget and provides
        detailed cost breakdown.
        """
        entities = nlp_result.get("entities", {})
        budget_level = entities.get("budget_level", "mid")
        duration_str = entities.get("duration", "7 days")
        duration_days = self.parse_duration(duration_str)

        # Calculate budget
        calculation = self.calculate_budget(
            budget_level,
            duration_days
        )

        budget_breakdown = self.format_budget_breakdown(calculation)
        entry_fees = self.format_entry_fees()

        # RAG context for budget tips
        results = self.retriever.search(
            "Sri Lanka budget travel costs tips",
            k=3
        )
        context = self.retriever.format_context(results)

        # Generate response
        prompt = ChatPromptTemplate.from_messages([
            ("system", BUDGET_PROMPT),
            ("human", "{query}")
        ])

        chain = prompt | self.llm

        response = chain.invoke({
            "budget_level": budget_level,
            "budget_breakdown": budget_breakdown,
            "entry_fees": entry_fees,
            "context": context,
            "history": history,
            "query": query
        })

        return {
            "answer": response.content,
            "sources": ["Budget Travel Guide", "Sri Lanka Travel Guide"],
            "agent": "budget",
            "calculation": calculation
        }


budget_agent = BudgetAgent()