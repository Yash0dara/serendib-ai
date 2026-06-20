# backend/nlp/analyzer.py

from backend.nlp.intent import intent_detector
from backend.nlp.entities import entity_extractor
from backend.nlp.sentiment import sentiment_analyzer

class NLPAnalyzer:
    def analyze(self, text: str) -> dict:
        intent = intent_detector.detect(text)
        entities = entity_extractor.extract(text)
        sentiment = sentiment_analyzer.analyze(text)

        return {
            "intent": intent["intent"],
            "intent_confidence": intent["confidence"],
            "secondary_intents": intent.get(
                "secondary_intents",
                []
            ),
            "intent_method": intent.get(
                "method",
                "unknown"
            ),
            "entities": entities,
            "sentiment": sentiment["mood"],
            "sentiment_confidence": sentiment["confidence"],
            "user_state": sentiment.get(
                "user_state"
            )
        }

nlp_analyzer = NLPAnalyzer()