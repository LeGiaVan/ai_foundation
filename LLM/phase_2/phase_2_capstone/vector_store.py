from qdrant_client import QdrantClient
from qdrant_client import models as q_models
import uuid
from datetime import datetime
from config import settings
from embedder import embedder_instance
from models import DocumentMetadata

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.collection_name = settings.qdrant_collection_name
        self.meta_collection_name = settings.qdrant_meta_collection_name
        
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        # Tạo collection chính cho RAG (Hybrid Search)
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": q_models.VectorParams(
                        size=384, # all-MiniLM-L6-v2 size
                        distance=q_models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "bm25": q_models.SparseVectorParams()
                }
            )
            print(f"Created collection {self.collection_name}")
            
        # Tạo collection phụ để lưu metadata của document
        if not self.client.collection_exists(self.meta_collection_name):
            self.client.create_collection(
                collection_name=self.meta_collection_name,
                vectors_config=q_models.VectorParams(
                    size=1, # Dummy size
                    distance=q_models.Distance.COSINE
                )
            )
            print(f"Created metadata collection {self.meta_collection_name}")

    def upsert_document(self, doc_id: str, filename: str, chunks_data: list[dict]) -> int:
        texts = [chunk["text"] for chunk in chunks_data]
        
        # Tạo Embeddings
        dense_vecs = embedder_instance.embed_dense(texts)
        sparse_vecs = embedder_instance.embed_sparse(texts)
        
        points = []
        for i in range(len(chunks_data)):
            chunk_id = str(uuid.uuid4())
            sparse = sparse_vecs[i]
            
            payload = {
                "doc_id": doc_id,
                "filename": filename,
                "text": chunks_data[i]["text"],
                "page": chunks_data[i]["page"]
            }
            
            points.append(q_models.PointStruct(
                id=chunk_id,
                vector={
                    "dense": dense_vecs[i],
                    "bm25": q_models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist()
                    )
                },
                payload=payload
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        # Lưu vào collection metadata
        self.client.upsert(
            collection_name=self.meta_collection_name,
            points=[q_models.PointStruct(
                id=doc_id,
                vector=[0.0], # Dummy vector
                payload={
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_count": len(chunks_data),
                    "upload_time": datetime.now().isoformat()
                }
            )]
        )
        return len(chunks_data)

    def delete_document(self, doc_id: str):
        # Xóa chunks trong collection chính
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=q_models.Filter(
                must=[
                    q_models.FieldCondition(
                        key="doc_id",
                        match=q_models.MatchValue(value=doc_id)
                    )
                ]
            )
        )
        # Xóa metadata
        self.client.delete(
            collection_name=self.meta_collection_name,
            points_selector=[doc_id]
        )

    def get_all_documents(self) -> list[DocumentMetadata]:
        # Cuộn qua collection metadata để lấy tất cả docs
        docs = []
        scroll_result = self.client.scroll(
            collection_name=self.meta_collection_name,
            limit=1000
        )[0]
        
        for point in scroll_result:
            p = point.payload
            docs.append(DocumentMetadata(
                doc_id=p["doc_id"],
                filename=p["filename"],
                chunk_count=p["chunk_count"],
                upload_time=p["upload_time"]
            ))
        return docs

    def search_hybrid(self, query: str, doc_id: str = None, top_k: int = 5) -> list[dict]:
        dense_q = embedder_instance.embed_dense([query])[0]
        sparse_q_list = embedder_instance.embed_sparse([query])
        sparse_q = sparse_q_list[0]
        
        filter_cond = None
        if doc_id:
            filter_cond = q_models.Filter(
                must=[q_models.FieldCondition(key="doc_id", match=q_models.MatchValue(value=doc_id))]
            )
            
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                q_models.Prefetch(
                    query=dense_q, 
                    using="dense", 
                    limit=20,
                    filter=filter_cond
                ),
                q_models.Prefetch(
                    query=q_models.SparseVector(
                        indices=sparse_q.indices.tolist(),
                        values=sparse_q.values.tolist()
                    ),
                    using="bm25",
                    limit=20,
                    filter=filter_cond
                )
            ],
            query=q_models.FusionQuery(fusion=q_models.Fusion.RRF),
            limit=top_k
        )
        
        output = []
        for res in results.points:
            # results.points chứa payload và score
            output.append({
                "score": res.score,
                "text": res.payload["text"],
                "filename": res.payload["filename"],
                "page": res.payload["page"]
            })
        return output

vector_store_instance = VectorStore()
