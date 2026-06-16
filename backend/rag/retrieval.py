from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


# CONFIGURATION
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "serendib_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



# RETRIEVER CLASS
class SerendibRetriever:
    """
    Handles all search operations against
    the Sri Lanka knowledge base in ChromaDB.
    """

    def __init__(self):
        print("⏳ Connecting to knowledge base...")

        # Load embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        # Connect to existing ChromaDB
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PATH
        )

        print("✅ Knowledge base connected!")


    # BASIC SEARCH

    def search(self, query: str, k: int = 3) -> list:
        """
        Basic semantic search.
        Returns top k most relevant chunks.

        k = how many results to return
        """
        results = self.vector_store.similarity_search(
            query=query,
            k=k
        )
        return results


    # SEARCH WITH SCORE

    def search_with_score(self, query: str, k: int = 3) -> list:
        """
        Same as search but also returns
        relevance score for each result.

        Score closer to 0 = more relevant
        Score closer to 1 = less relevant
        """
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k
        )
        return results


    # FILTERED SEARCH
    def search_by_category(
        self,
        query: str,
        category: str,
        k: int = 3
    ) -> list:
        """
        Search within a specific category only.

        Categories:
        - destination
        - food
        - transport
        - culture
        - adventure
        - practical
        - itinerary
        """
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter={"category": category}
        )
        return results

    def search_by_traveler_type(
        self,
        query: str,
        traveler_type: str,
        k: int = 3
    ) -> list:
        """
        Search content relevant to a specific traveler type.

        Types:
        - Solo Traveler
        - Cultural Traveler
        - Adventure Traveler
        """
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter={"best_for": {"$contains": traveler_type}}
        )
        return results


    # SMART SEARCH
    def smart_search(
        self,
        query: str,
        traveler_type: str = None,
        category: str = None,
        k: int = 4
    ) -> list:
        """
        Intelligent search that combines:
        - Semantic similarity
        - Optional category filter
        - Optional traveler type filter
        - Falls back to basic search if filtered search fails
        """
        try:
            # Build filter if provided
            if category and traveler_type:
                results = self.search_by_category(
                    query, category, k
                )
            elif category:
                results = self.search_by_category(
                    query, category, k
                )
            else:
                results = self.search(query, k)

            # If no results found fall back to basic search
            if not results:
                print("   ⚠️ No filtered results. Using basic search.")
                results = self.search(query, k)

            return results

        except Exception as e:
            print(f"   ⚠️ Smart search error: {e}. Using basic search.")
            return self.search(query, k)


    # FORMAT RESULTS
    def format_context(self, results: list) -> str:
        """
        Formats retrieved chunks into a single
        context string to send to the LLM.

        LLM reads this context to generate answer.
        """
        if not results:
            return "No relevant information found."

        context_parts = []

        for i, doc in enumerate(results, 1):
            topic = doc.metadata.get("topic", "Sri Lanka")
            category = doc.metadata.get("category", "general")

            context_parts.append(
                f"[Source {i} - {topic} ({category})]:\n"
                f"{doc.page_content}\n"
            )

        return "\n".join(context_parts)



# SINGLE INSTANCE
retriever = SerendibRetriever()