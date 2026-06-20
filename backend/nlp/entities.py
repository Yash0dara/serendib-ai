from transformers import pipeline
import re

class EntityExtractor:
    def __init__(self):
        print("⏳ Loading NER model...")
        self.ner = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=-1
        )
        print("✅ NER model loaded!")
    def extract(self, text: str) -> dict:

        ner_results = self.ner(text)

        entities = {
            "locations": [],
            "duration": None,
            "budget_level": None,
            "dates": []
        }

        # NER locations
        for ent in ner_results:

            if ent["entity_group"] == "LOC":

                location = ent["word"]

                location = location.replace(" ' ", "'")
                location = location.replace(" ##", "")

                entities["locations"].append(location)

        # Sri Lanka location dictionary
        SL_LOCATIONS = [
            "colombo",
            "kandy",
            "galle",
            "ella",
            "nuwara eliya",
            "sigiriya",
            "anuradhapura",
            "polonnaruwa",
            "dambulla",
            "mirissa",
            "trincomalee",
            "arugam bay",
            "hikkaduwa",
            "bentota",
            "adam's peak",
            "little adam's peak",
            "ella rock",
            "yala",
            "udawalawe",
            "negombo"
        ]

        text_lower = text.lower()

        for location in SL_LOCATIONS:

            if location in text_lower:
                entities["locations"].append(
                    location.title()
                )

        # Duration
        dur_match = re.search(
            r'(\d+)\s*(day|days|week|weeks|hour|hours)',
            text_lower
        )

        if dur_match:
            entities["duration"] = (
                f"{dur_match.group(1)} "
                f"{dur_match.group(2)}"
            )

        # Budget
        budget_keywords = {
            "budget": "low",
            "cheap": "low",
            "affordable": "low",
            "mid": "mid",
            "moderate": "mid",
            "average": "mid",
            "luxury": "high",
            "expensive": "high",
            "premium": "high"
        }

        for word, level in budget_keywords.items():

            if word in text_lower:
                entities["budget_level"] = level
                break

        # Dates
        months = (
            r'(january|february|march|april|may|june|'
            r'july|august|september|october|november|'
            r'december|next month|this week)'
        )

        date_matches = re.findall(
            months,
            text_lower
        )

        if date_matches:
            entities["dates"] = date_matches

        entities["locations"] = list(
            set(entities["locations"])
        )

        return entities

entity_extractor = EntityExtractor()