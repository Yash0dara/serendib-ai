from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        print("⏳ Loading sentiment model...")
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1
        )
        print("✅ Sentiment model loaded!")

    def analyze(self, text: str) -> dict:

        result = self.analyzer(text)[0]

        label = result["label"].lower()
        score = round(result["score"], 3)

        mood = (
            "positive"
            if "positive" in label
            else "negative"
            if "negative" in label
            else "neutral"
        )

        frustration_words = [
            "cancelled",
            "delayed",
            "late",
            "missed",
            "angry",
            "frustrated",
            "annoyed",
            "upset"
        ]

        text_lower = text.lower()

        user_state = None

        if any(
            word in text_lower
            for word in frustration_words
        ):
            user_state = "frustrated"

        return {
            "mood": mood,
            "confidence": score,
            "user_state": user_state
        }

sentiment_analyzer = SentimentAnalyzer()