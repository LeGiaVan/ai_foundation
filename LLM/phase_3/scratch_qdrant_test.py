import os
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="rag_docs",
    embedding=embeddings,
    vector_name="dense",
    content_payload_key="text"
)

docs = vector_store.similarity_search("test", k=1)
if docs:
    print("Metadata from default:", docs[0].metadata)

vector_store_flat = QdrantVectorStore(
    client=client,
    collection_name="rag_docs",
    embedding=embeddings,
    vector_name="dense",
    content_payload_key="text",
    metadata_payload_key=None
)
docs_flat = vector_store_flat.similarity_search("test", k=1)
if docs_flat:
    print("Metadata from metadata_payload_key=None:", docs_flat[0].metadata)
