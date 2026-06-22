# Protects the bot from misuse and bad inputs

import re


class SafetyGuard:
    """
    Checks every message before processing.

    Catches:
    - Empty or too short messages
    - Too long messages (spam)
    - Off-topic questions
    - Harmful or inappropriate content
    - Gibberish or random characters
    """

    # ── Off Topic Keywords ──
    OFF_TOPIC_KEYWORDS = [
        "bitcoin", "crypto", "stock market", "forex",
        "programming", "python code", "javascript",
        "write me a code", "build me a website",
        "who is the president", "politics",
        "football", "cricket score", "movie review",
        "dating", "relationship advice",
        "homework", "exam answers", "assignment",
        "hack", "crack", "pirate",
        "generate image", "draw me", "create art"
    ]

    # ── Harmful Keywords ──
    HARMFUL_KEYWORDS = [
        "kill", "murder", "weapon", "bomb",
        "drugs", "illegal", "smuggle",
        "abuse", "assault", "threaten",
        "suicide", "self harm", "self-harm"
    ]

    # ── Sri Lanka Travel Keywords ──
    # If message contains any of these → likely on topic
    TRAVEL_KEYWORDS = [
        "sri lanka", "colombo", "kandy", "ella", "galle",
        "sigiriya", "train", "bus", "hotel", "hostel",
        "beach", "temple", "food", "trip", "travel",
        "visit", "tour", "itinerary", "budget", "cost",
        "weather", "monsoon", "safari", "hike", "surf",
        "visa", "airport", "tuk-tuk", "currency", "rupee",
        "plan", "recommend", "best time", "safe",
        "backpack", "solo", "adventure", "culture",
        "heritage", "festival", "museum", "national park",
        "mirissa", "trincomalee", "arugam", "nuwara",
        "anuradhapura", "polonnaruwa", "dambulla",
        "yala", "udawalawe", "wilpattu", "minneriya",
        "hopper", "kottu", "curry", "rice",
        "hello", "hi", "hey", "help", "thank"
    ]

    # ── Responses ──
    OFF_TOPIC_RESPONSE = (
        "I am Serendib AI, specialized only in Sri Lanka travel. 🌴\n\n"
        "I can help you with:\n"
        "- Trip planning and itineraries\n"
        "- Food recommendations\n"
        "- Transport (trains, buses, tuk-tuks)\n"
        "- Budget planning\n"
        "- Weather information\n"
        "- Culture and heritage\n"
        "- Adventure activities\n"
        "- Safety tips\n\n"
        "Please ask me something about Sri Lanka travel!"
    )

    HARMFUL_RESPONSE = (
        "I cannot help with that request. 🚫\n\n"
        "I am a travel assistant for Sri Lanka.\n"
        "If you need emergency help in Sri Lanka:\n\n"
        "🚨 Tourist Police: 1912\n"
        "🚨 Emergency: 119\n"
        "🚨 Ambulance: 110"
    )

    EMPTY_RESPONSE = (
        "It looks like your message was empty.\n"
        "Try asking something like:\n\n"
        "- 'Plan a 3 day trip to Ella'\n"
        "- 'What food should I try in Colombo?'\n"
        "- 'How do I get from Kandy to Ella by train?'"
    )

    TOO_LONG_RESPONSE = (
        "Your message is quite long. 📝\n"
        "Could you please ask a shorter, more specific question?\n\n"
        "For example:\n"
        "- 'What should I do in Kandy?'\n"
        "- 'Best beaches in Sri Lanka?'"
    )

    GIBBERISH_RESPONSE = (
        "I could not understand your message. 🤔\n"
        "Could you please rephrase?\n\n"
        "Try asking in English about Sri Lanka travel."
    )

    def check(self, message: str) -> dict:
        """
        Main safety check function.

        Returns:
        {
            "safe": True/False,
            "reason": "why blocked",
            "response": "response to show user"
        }
        """
        # Check 1 — Empty or too short
        if not message or len(message.strip()) < 2:
            return {
                "safe": False,
                "reason": "empty_message",
                "response": self.EMPTY_RESPONSE
            }

        # Check 2 — Too long (spam protection)
        if len(message) > 1000:
            return {
                "safe": False,
                "reason": "too_long",
                "response": self.TOO_LONG_RESPONSE
            }

        # Check 3 — Gibberish detection
        if self.is_gibberish(message):
            return {
                "safe": False,
                "reason": "gibberish",
                "response": self.GIBBERISH_RESPONSE
            }

        # Check 4 — Harmful content
        if self.is_harmful(message):
            return {
                "safe": False,
                "reason": "harmful_content",
                "response": self.HARMFUL_RESPONSE
            }

        # Check 5 — Off topic detection
        if self.is_off_topic(message):
            return {
                "safe": False,
                "reason": "off_topic",
                "response": self.OFF_TOPIC_RESPONSE
            }

        # All checks passed
        return {
            "safe": True,
            "reason": "passed",
            "response": None
        }

    def is_gibberish(self, text: str) -> bool:
        """Detects gibberish or random character input."""
        # Too many special characters
        special_count = len(re.findall(r'[^a-zA-Z0-9\s\.\,\!\?\'\-]', text))
        if special_count > len(text) * 0.4:
            return True

        # Too many consonants in a row (not real words)
        if re.search(r'[bcdfghjklmnpqrstvwxyz]{6,}', text.lower()):
            return True

        # Repeated characters
        if re.search(r'(.)\1{5,}', text):
            return True

        # Very short with no real words
        words = text.strip().split()
        if len(words) <= 2:
            real_words = [w for w in words if len(w) > 1]
            if not real_words:
                return True

        return False

    def is_harmful(self, text: str) -> bool:
        """Checks for harmful or dangerous content."""
        text_lower = text.lower()

        for keyword in self.HARMFUL_KEYWORDS:
            if keyword in text_lower:
                return True

        return False

    def is_off_topic(self, text: str) -> bool:
        """
        Checks if message is about Sri Lanka travel.

        Logic:
        1. If travel keywords found → on topic
        2. If off-topic keywords found → off topic
        3. If neither → allow it (might be follow-up question)
        """
        text_lower = text.lower()

        # Check for travel keywords first
        has_travel = any(
            kw in text_lower
            for kw in self.TRAVEL_KEYWORDS
        )

        if has_travel:
            return False

        # Check for off-topic keywords
        has_off_topic = any(
            kw in text_lower
            for kw in self.OFF_TOPIC_KEYWORDS
        )

        if has_off_topic:
            return True

        # No clear match either way
        # Allow it — might be a follow-up question
        # like "tell me more" or "what else?"
        return False


# Single instance
safety_guard = SafetyGuard()