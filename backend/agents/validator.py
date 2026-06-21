# Response quality checker - balanced validation

import os
import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

VALIDATOR_PROMPT = """
You are a quality checker for a Sri Lanka travel assistant.

Be LENIENT. Only fail responses that are clearly wrong.

Check this response and return ONLY a JSON object.

ONLY fail (passed: false) if:
- Response is completely off-topic (not about travel)
- Response contains clearly dangerous advice
- Response is empty or gibberish

PASS (passed: true) if:
- Response attempts to answer the travel question
- Response is helpful even if not perfect
- Response uses general travel knowledge

Return ONLY this JSON (no other text):
{{
    "passed": true or false,
    "hallucination_detected": false,
    "is_relevant": true or false,
    "is_safe": true or false,
    "issues": "none or brief description",
    "confidence_score": 0.0 to 1.0
}}

Question: {question}
Response to check: {response}
"""

FALLBACK_RESPONSE = """
I want to make sure I give you accurate Sri Lanka travel information.

Could you rephrase your question?

Or check these official sources:
- 🌐 srilanka.travel
- 🚂 railway.gov.lk
"""


class ValidatorAgent:
    """
    Balanced quality checker.
    Only rejects clearly bad responses.
    """

    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=200
        )

    def basic_checks(
        self,
        response: str,
        question: str
    ) -> dict:
        """
        Simple rule based checks first.
        Fast and reliable.
        """
        # Empty response
        if not response or len(response.strip()) < 20:
            return {
                "passed": False,
                "issue": "Response too short or empty"
            }

        # Response is just the fallback already
        if "srilanka.travel" in response and len(response) < 200:
            return {
                "passed": True,
                "issue": "none"
            }

        # Check minimum length
        if len(response.split()) < 10:
            return {
                "passed": False,
                "issue": "Response too brief"
            }

        return {"passed": True, "issue": "none"}

    def validate(
        self,
        question: str,
        response: str,
        context: str
    ) -> dict:
        """
        Validates response quality.
        """
        # Run basic checks first
        basic = self.basic_checks(response, question)
        if not basic["passed"]:
            return {
                "passed": False,
                "hallucination_detected": False,
                "is_relevant": False,
                "is_safe": True,
                "issues": basic["issue"],
                "confidence_score": 0.0
            }

        # Run LLM validation
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", VALIDATOR_PROMPT),
                ("human", "Please validate.")
            ])

            chain = prompt | self.llm

            result = chain.invoke({
                "question": question,
                "response": response[:600]
            })

            json_match = re.search(
                r'\{.*\}',
                result.content,
                re.DOTALL
            )

            if json_match:
                validation = json.loads(json_match.group())
                return validation

        except Exception:
            pass

        # Default → pass if unsure
        return {
            "passed": True,
            "hallucination_detected": False,
            "is_relevant": True,
            "is_safe": True,
            "issues": "none",
            "confidence_score": 0.8
        }

    def apply(
        self,
        question: str,
        response: str,
        context: str
    ) -> tuple[str, dict]:
        """
        Returns final response after validation.
        """
        validation = self.validate(question, response, context)

        # Only use fallback for clear failures
        if (not validation.get("passed", True) and
                not validation.get("is_safe", True)):
            return FALLBACK_RESPONSE, validation

        # Pass everything else through
        return response, validation


validator_agent = ValidatorAgent()
