from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from config import settings

class Embedder:
    def __init__(self):
        print("Loading Dense Model...")
        self.dense_model = SentenceTransformer(settings.dense_embedding_model)
        
        print("Loading Sparse Model (BM25)...")
        self.sparse_model = SparseTextEmbedding(settings.sparse_embedding_model)
        
    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Return a list of dense embeddings (list of float lists) for the given texts."""
        embeddings = self.dense_model.encode(texts)
        return [emb.tolist() for emb in embeddings]

    def embed_sparse(self, texts: list[str]) -> list[dict]:
        """Return a list of sparse vectors (dicts with indices & values) for the given texts."""
        embeddings = list(self.sparse_model.embed(texts))
        return embeddings

# Singleton instance used throughout the project
embedder_instance = Embedder()