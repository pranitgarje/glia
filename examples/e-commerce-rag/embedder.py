from sentence_transformers import SentenceTransformer
from typing import List

class RealEmbedder:
    """
    Wraps sentence-transformers to match Glia's expected interface.
    
    GliaManager expects the injected vectorizer to expose:
    - embed(text: str) -> List[float]
    - embed_many(texts: List[str]) -> List[List[float]]
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # "all-MiniLM-L6-v2" is a small, fast model perfect for local testing.
        # Note: This model outputs vectors with 384 dimensions.
        self.model = SentenceTransformer(model_name)
    
    def embed(self, text: str) -> List[float]:
        """Embeds a single string into a list of floats for Glia's check/store methods."""
        # .encode() returns a numpy array, but Glia requires a standard Python list
        return self.model.encode(text).tolist()
    
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of strings (optional, useful for batch ingestion)."""
        return self.model.encode(texts).tolist()