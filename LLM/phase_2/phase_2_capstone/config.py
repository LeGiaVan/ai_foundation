import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    groq_api_key: str = "sk-fake-key" # Sẽ bị ghi đè bởi .env
    llm_model: str = "groq/compound-mini"
    
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "rag_docs"
    qdrant_meta_collection_name: str = "rag_docs_meta"
    
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    dense_embedding_model: str = "all-MiniLM-L6-v2"
    sparse_embedding_model: str = "Qdrant/bm25"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
