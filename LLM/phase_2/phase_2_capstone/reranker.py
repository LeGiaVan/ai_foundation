from sentence_transformers import CrossEncoder
from config import settings

class Reranker:
    def __init__(self):
        print("Loading Reranker Model (Cross-Encoder)...")
        self.model = CrossEncoder(settings.reranker_model)
        
    def rerank(self, query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
        """
        Rerank a list of chunks based on a query using CrossEncoder.
        chunks: list of dicts, each must contain at least a 'text' key.
        Returns the top_k chunks with updated scores.
        """
        if not chunks:
            return []
            
        # Create sentence pairs: [[query, chunk1], [query, chunk2], ...]
        pairs = [[query, chunk["text"]] for chunk in chunks]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Update chunks with new scores
        for i, chunk in enumerate(chunks):
            chunk["score"] = float(scores[i])
            
        # Sort chunks by score in descending order
        reranked_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
        
        # Return top_k
        return reranked_chunks[:top_k]

# Singleton instance
reranker_instance = Reranker()
