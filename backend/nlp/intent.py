# backend/nlp/intent.py
# Rule-First Hybrid Intent Detector
# Rules handle clear cases → ML handles ambiguous ones

from transformers import pipeline

# ============================================
# KEYWORD RULES (High Priority)
# More specific = higher confidence
# ============================================

KEYWORD_RULES = {
    "transport_query": {
        "keywords": [
            "train", "bus", "tuk-tuk", "tuk tuk", "taxi",
            "how to get to", "how to reach", "get from",
            "travel from", "travel to", "pickme", "uber",
            "from colombo to", "from kandy to", "from galle to",
            "from ella to", "ticket", "station", "departure",
            "arrival", "journey from", "route from"
        ],
        "weight": 0.85
    },
    "food_recommendation": {
        "keywords": [
            "food", "eat", "restaurant", "cuisine", "dish",
            "meal", "hungry", "taste", "try", "hopper",
            "kottu", "curry", "cafe", "street food",
            "where to eat", "best food", "local food",
            "what to eat", "dining", "breakfast", "lunch", "dinner"
        ],
        "weight": 0.85
    },
    "budget_query": {
        "keywords": [
            "how much", "cost", "price", "budget", "cheap",
            "expensive", "afford", "lkr", "rupee", "spend",
            "money", "entry fee", "ticket price", "daily cost",
            "per day", "total cost", "fee"
        ],
        "weight": 0.85
    },
    "weather_query": {
        "keywords": [
            "weather", "rain", "sunny", "monsoon", "season",
            "climate", "temperature", "hot", "cold",
            "best time to visit", "when to visit",
            "best month", "rainy season", "dry season"
        ],
        "weight": 0.85
    },
    "cultural_info": {
        "keywords": [
            "temple", "culture", "history", "festival",
            "heritage", "religion", "ancient", "relic",
            "museum", "perahera", "buddhist", "hindu",
            "colonial", "ruins", "sigiriya", "anuradhapura",
            "polonnaruwa", "dambulla", "tooth relic"
        ],
        "weight": 0.85
    },
    "adventure_activity": {
        "keywords": [
            "hike", "hiking", "surf", "surfing", "safari",
            "wildlife", "trek", "trekking", "climb", "climbing",
            "waterfall", "dive", "diving", "camp", "camping",
            "adams peak", "ella rock", "nine arch",
            "little adams peak", "knuckles", "leopard",
            "elephant", "whale watching", "zip line"
        ],
        "weight": 0.85
    },
    "emergency": {
        "keywords": [
            "emergency", "danger", "accident",
            "lost my passport", "i am sick", "hospital",
            "police", "stolen", "hurt", "injured",
            "robbed", "attacked", "urgent help"  # ← more specific
        ],
        "weight": 0.95
    },"safety_query": {
    "keywords": [
        "safe", "is it safe", "dangerous",
        "alone", "risk", "security",
        "crime", "scam", "unsafe"
    ],
    "weight": 0.90
    },
    "greeting": {
        "keywords": [
            "hello", "hi", "hey", "good morning",
            "good evening", "good afternoon",
            "what can you do", "how can you help",
            "who are you", "what are you",
            "can you help", "help me plan",    # ← add these
            "i need help", "assist me"          # ← add these
        ],
        "weight": 0.90
    },
"trip_planning": {
    "keywords": [
        "itinerary", "plan my trip","plan a trip", "travel plan", "trip","traveling",
        "visit","visiting","going to","stay in","days in sri lanka","visit sri lanka",
        "schedule","places to visit","what to do in","recommend", "day trip","week in sri lanka",
        "best places","must see","should i visit","worth visiting"
    ],
    "weight": 0.85
}
}

# ML Labels for ambiguous cases
ML_LABELS = [
    "planning a multi day trip itinerary",
    "asking about local food or restaurants",
    "asking about transport trains or buses",
    "asking about travel costs or budget",
    "asking about weather or best season",
    "asking about culture history or temples",
    "asking about adventure hiking or wildlife",
    "saying hello or greeting",
    "reporting emergency or needing help",
    "general question about Sri Lanka",
    "asking about travel safety or risks",
]

ML_LABEL_MAP = {
    "planning a multi day trip itinerary": "trip_planning",
    "asking about local food or restaurants": "food_recommendation",
    "asking about transport trains or buses": "transport_query",
    "asking about travel costs or budget": "budget_query",
    "asking about weather or best season": "weather_query",
    "asking about culture history or temples": "cultural_info",
    "asking about adventure hiking or wildlife": "adventure_activity",
    "saying hello or greeting": "greeting",
    "reporting emergency or needing help": "emergency",
    "general question about Sri Lanka": "general_question",
    "asking about travel safety or risks": "safety_query",
}

# Confidence threshold
# Below this → use ML to help decide
CONFIDENCE_THRESHOLD = 0.50


class IntentDetector:
    """
    Rule-First Hybrid Intent Detector.

    Priority order:
    1. Emergency keywords → always highest priority
    2. Strong keyword matches → high confidence
    3. ML model → for ambiguous messages
    4. General question → fallback
    """

    def __init__(self):
        print("⏳ Loading intent model...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
        print("✅ Intent model loaded!")

    def rule_check(self, text: str) -> dict:
        """
        Checks all keyword rules.
        Returns intent scores based on keyword matches.
        """
        text_lower = text.lower()
        scores = {}

        for intent, config in KEYWORD_RULES.items():
            keywords = config["keywords"]
            weight = config["weight"]

            # Count keyword matches
            matches = [kw for kw in keywords if kw in text_lower]

            if matches:
                # More matches = higher confidence
                match_score = min(len(matches) * 0.20, 1.0)
                scores[intent] = match_score * weight

        return scores

    def ml_check(self, text: str) -> dict:
        """
        Uses ML model for ambiguous cases.
        Returns intent scores from ML model.
        """
        result = self.classifier(
            text,
            candidate_labels=ML_LABELS
        )

        scores = {}
        for label, score in zip(
            result["labels"],
            result["scores"]
        ):
            intent = ML_LABEL_MAP.get(label, "general_question")
            scores[intent] = score

        return scores
    def detect(self, text: str) -> dict:

        # Rule scores
        rule_scores = self.rule_check(text)

        # Emergency always wins
        if "emergency" in rule_scores:
            return {
                "intent": "emergency",
                "confidence": round(rule_scores["emergency"], 3),
                "secondary_intents": [],
                "method": "rule"
            }

        # ML scores
        ml_scores = self.ml_check(text)

        # Combine
        all_intents = set(
            list(rule_scores.keys()) +
            list(ml_scores.keys())
        )

        combined = {}

        for intent in all_intents:
            rule = rule_scores.get(intent, 0)
            ml = ml_scores.get(intent, 0)

            combined[intent] = (
                rule * 0.65 +
                ml * 0.35
            )

        if not combined:
            return {
                "intent": "general_question",
                "confidence": 0.50,
                "secondary_intents": [],
                "method": "fallback"
            }

        # Sort by score
        sorted_intents = sorted(
            combined.items(),
            key=lambda x: x[1],
            reverse=True
        )

        primary_intent = sorted_intents[0][0]
        primary_score = sorted_intents[0][1]

        secondary_intents = [
            intent
            for intent, score in sorted_intents[1:]
            if score > 0.25
        ]

        return {
            "intent": primary_intent,
            "confidence": round(primary_score, 3),
            "secondary_intents": secondary_intents[:2],
            "method": "combined"
        }

intent_detector = IntentDetector()