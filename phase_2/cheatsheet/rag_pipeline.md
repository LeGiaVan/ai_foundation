# Tuần 5 – Vector Search + RAG Integration Cheatsheet

---

## 1️⃣  Retrieval Quality — Đánh giá chất lượng tìm kiếm

### Tại sao retrieval là nút cổ chai?

```
User hỏi: "Bệnh tiểu đường type 2 điều trị thế nào?"
                    │
        ┌───────────┴───────────┐
        │  Retrieval (tìm chunk) │  ← SAI ở đây thì mọi thứ phía sau đều sai
        └───────────┬───────────┘
                    │
            ┌───────┴───────┐
            │  LLM trả lời  │  ← Dù LLM giỏi cỡ nào cũng không cứu được context sai
            └───────────────┘

→ Garbage In, Garbage Out
```

### 4 kiểu thất bại retrieval

| Kiểu lỗi | Nghĩa là | Ví dụ |
|---|---|---|
| **False Positive** | Tìm về chunk **không liên quan** | Query "Apple stock price" → trả về chunk về "apple fruit nutrition" |
| **False Negative** | **Bỏ sót** chunk liên quan | Query "tiểu đường" → bỏ sót chunk chứa "diabetes mellitus" (từ đồng nghĩa tiếng Anh) |
| **Chunk quá lớn** | Vector bị "trung bình hóa" nhiều chủ đề | Chunk 2000 tokens chứa cả chẩn đoán + điều trị + tiên lượng → vector không đại diện tốt cho topic nào |
| **Chunk quá nhỏ** | Thiếu ngữ cảnh | Chunk "có thể dùng insulin" → không biết dùng cho bệnh gì, ai, lúc nào |

### Precision@k — Metric đơn giản nhất

```
Precision@k = số chunk relevant trong top-k / k

Ví dụ: Query "Cách điều trị tiểu đường type 2?"
Top-3 trả về:
  1. "Insulin là phương pháp điều trị chính cho tiểu đường type 2..."  ✅ relevant
  2. "Bệnh nhân cần theo dõi đường huyết hàng ngày..."              ✅ relevant
  3. "Bệnh viện Bạch Mai được thành lập năm 1911..."                ❌ irrelevant

Precision@3 = 2/3 = 0.67 (67%)
```

### Code đánh giá Retrieval

```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

model  = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

# ── Bước 1: Tạo bộ test (câu hỏi + chunk_id mong đợi) ──────────
test_cases = [
    {
        "query": "Bệnh tiểu đường type 2 điều trị thế nào?",
        "expected_chunk_ids": [1, 5, 12],  # Bạn tự xác định trước
    },
    {
        "query": "Tác dụng phụ của insulin là gì?",
        "expected_chunk_ids": [3, 7],
    },
    # ... thêm 8-10 câu nữa
]

# ── Bước 2: Chạy retrieval + tính Precision@k ────────────────────
def evaluate_retrieval(test_cases, collection_name, top_k=3):
    total_precision = 0
    
    for tc in test_cases:
        q_vec = model.encode(tc["query"]).tolist()
        results = client.query_points(
            collection_name=collection_name, query=q_vec, limit=top_k
        )
        
        retrieved_ids = [r.id for r in results.points]
        relevant_count = sum(1 for rid in retrieved_ids if rid in tc["expected_chunk_ids"])
        precision = relevant_count / top_k
        
        status = "✅" if precision >= 0.67 else "⚠️" if precision > 0 else "❌"
        print(f"{status} P@{top_k}={precision:.2f} | Query: {tc['query'][:60]}")
        print(f"   Retrieved: {retrieved_ids} | Expected: {tc['expected_chunk_ids']}")
        
        total_precision += precision
    
    avg = total_precision / len(test_cases)
    print(f"\n📊 Average Precision@{top_k} = {avg:.2f}")
    return avg

evaluate_retrieval(test_cases, "my_docs", top_k=3)
```

### So sánh chunk_size ảnh hưởng đến Retrieval

```python
# Test 3 chunk_size khác nhau trên cùng bộ câu hỏi
for size in [200, 500, 1000]:
    print(f"\n{'='*50}")
    print(f"CHUNK_SIZE = {size}")
    # 1. Chunk lại toàn bộ text
    # 2. Embed + upsert vào collection riêng (vd: "docs_200", "docs_500", "docs_1000")
    # 3. Chạy evaluate_retrieval → ghi nhận Precision@3
```

```
Kết quả kỳ vọng (ví dụ):

chunk_size=200:  P@3 = 0.47  ← Chunk nhỏ quá, thiếu context
chunk_size=500:  P@3 = 0.73  ← Sweet spot ✅
chunk_size=1000: P@3 = 0.60  ← Chunk to quá, vector bị "trung bình"

→ Kết luận: chunk_size=500 tốt nhất cho dataset này
```

---

## 2️⃣  RAG Pipeline — Tích hợp LLM

### Luồng dữ liệu hoàn chỉnh

```
POST /ask {"question": "Bệnh tiểu đường điều trị thế nào?"}
  │
  ├─ 1. Embed question → query_vector (384 dims)
  │
  ├─ 2. Qdrant.query_points(query_vector, top_k=3) → 3 chunks + payload
  │
  ├─ 3. Format prompt:
  │     ┌──────────────────────────────────────────────────┐
  │     │ System: Chỉ trả lời dựa trên CONTEXT.           │
  │     │         Nếu không có thông tin → nói rõ.         │
  │     │                                                  │
  │     │ User:                                            │
  │     │   CONTEXT:                                       │
  │     │   [1] Insulin là phương pháp điều trị chính...   │
  │     │   [2] Bệnh nhân cần theo dõi đường huyết...     │
  │     │   [3] Chế độ ăn ít đường giúp kiểm soát...      │
  │     │                                                  │
  │     │   CÂU HỎI: Bệnh tiểu đường điều trị thế nào?   │
  │     └──────────────────────────────────────────────────┘
  │
  ├─ 4. LLM.chat(prompt) → answer
  │
  └─ 5. Return {"answer": "...", "sources": [...]}
```

### RAG Prompt Template

```python
def build_prompt(question: str, chunks: list[dict]) -> str:
    """Ghép câu hỏi + context chunks thành prompt cho LLM."""
    context = "\n\n".join(
        f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(chunks)
    )
    
    return f"""Bạn là trợ lý Q&A chuyên nghiệp. Hãy trả lời câu hỏi CHỈ DỰA TRÊN context được cung cấp.
Nếu không tìm thấy thông tin trong context, hãy nói rõ "Thông tin này không có trong tài liệu".

CONTEXT:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""
```

> ⚠️ **Tại sao phải nói "chỉ trả lời dựa trên context"?**
> Nếu không có ràng buộc này, LLM sẽ trộn kiến thức training data của nó với tài liệu của bạn → câu trả lời *có vẻ đúng* nhưng thực ra là **hallucination**.

### FastAPI RAG Endpoint

```python
# rag_api.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import openai  # hoặc bất kỳ LLM client nào

app = FastAPI()
model  = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

# ── Pydantic Models ───────────────────────────────────
class AskRequest(BaseModel):
    question: str
    top_k: int = 3

class Source(BaseModel):
    text: str
    source_file: str
    chunk_index: int
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

# ── Endpoints ─────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    # 1. Embed query
    q_vec = model.encode(request.question).tolist()
    
    # 2. Retrieve từ Qdrant
    results = client.query_points(
        collection_name="my_docs", query=q_vec, limit=request.top_k
    )
    
    # 3. Chuẩn bị sources và context
    sources = []
    chunks_for_prompt = []
    for r in results.points:
        sources.append(Source(
            text=r.payload["text"],
            source_file=r.payload.get("source", "unknown"),
            chunk_index=r.payload.get("chunk_id", -1),
            score=r.score,
        ))
        chunks_for_prompt.append(r.payload)
    
    # 4. Build prompt + gọi LLM
    prompt = build_prompt(request.question, chunks_for_prompt)
    
    # Gọi LLM (ví dụ OpenAI, thay bằng LLM bạn dùng)
    llm_response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = llm_response.choices[0].message.content
    
    return AskResponse(answer=answer, sources=sources)
```

---

## 3️⃣  Upload File qua API

```python
from fastapi import UploadFile, File, HTTPException
from io import BytesIO
import pymupdf
import docx

@app.post("/index")
async def index_file(file: UploadFile = File(...)):
    # 1. Validate file type
    allowed = [".pdf", ".docx"]
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Chỉ hỗ trợ {allowed}. Nhận được: {ext}")
    
    # 2. Đọc file vào bộ nhớ (không cần ghi ra disk)
    content = await file.read()
    buffer = BytesIO(content)
    
    # 3. Extract text
    if ext == ".pdf":
        with pymupdf.open(stream=buffer, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
    elif ext == ".docx":
        doc = docx.Document(buffer)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    
    # 4. Chunk → Embed → Upsert (dùng lại hàm đã viết)
    chunks  = split_text(text)
    vectors = model.encode(chunks, batch_size=32)
    # ... upsert vào Qdrant
    
    return {"status": "ok", "chunks_indexed": len(chunks), "filename": file.filename}
```

**Điểm quan trọng về `BytesIO`:**
```
BytesIO = file ảo trong RAM, không cần ghi ra ổ cứng

file.read() → bytes (dữ liệu thô)
BytesIO(bytes) → file-like object (pymupdf/docx đọc được trực tiếp)

Lợi ích:
  ✅ Nhanh hơn (không I/O disk)
  ✅ An toàn hơn (không tạo file tạm trên server)
  ✅ Sạch hơn (không cần dọn file sau khi xử lý)
```

---

## 4️⃣  Hybrid Search — Dense + Sparse (BM25)

### Tại sao dense-only đôi khi không đủ?

```
Query: "lỗi 404 NOT FOUND khi gọi API"

Dense search:
  → Tìm chunk về "lỗi HTTP" (hiểu ngữ nghĩa)          ✅
  → BỎ SÓT chunk có mã "404" chính xác                 ❌

BM25 (sparse):
  → Tìm chunk chứa đúng từ "404" và "NOT FOUND"        ✅
  → Bỏ sót chunk "HTTP error responses" (không khớp từ) ❌

Hybrid = Dense + BM25:
  → Tìm được CẢ HAI loại chunk                         ✅✅
```

### Reciprocal Rank Fusion (RRF) — Hợp nhất 2 danh sách

```
Dense results:           BM25 results:
  1. Chunk A (rank 1)      1. Chunk C (rank 1)
  2. Chunk B (rank 2)      2. Chunk A (rank 2)
  3. Chunk C (rank 3)      3. Chunk D (rank 3)

RRF score = Σ 1/(k + rank)    (k thường = 60)

Chunk A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325  ← Cao nhất
Chunk C: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
Chunk B: 1/(60+2) + 0        = 0.0161
Chunk D: 0        + 1/(60+3) = 0.0159

Final ranking: A → C → B → D
→ Chunk A cao nhất vì nó xuất hiện trong CẢ HAI danh sách
```

### Hybrid Search với Qdrant + Fastembed

```python
# Cài đặt: pip install fastembed

from qdrant_client import QdrantClient, models

client = QdrantClient(host="localhost", port=6333)

# Tạo collection hỗ trợ cả dense + sparse
client.create_collection(
    collection_name="hybrid_docs",
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE,
    ),
    sparse_vectors_config={
        "bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF,  # Dùng IDF weighting
        )
    },
)

# Upsert với cả 2 loại vector
from fastembed import SparseTextEmbedding, TextEmbedding

dense_model  = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding("Qdrant/bm25")

chunks = ["your chunks here..."]

# Encode dense + sparse
dense_vecs  = list(dense_model.embed(chunks))
sparse_vecs = list(sparse_model.embed(chunks))

# Upsert
points = [
    models.PointStruct(
        id=i,
        vector={
            "": dense_vecs[i].tolist(),           # dense vector (default)
            "bm25": models.SparseVector(          # sparse vector
                indices=sparse_vecs[i].indices.tolist(),
                values=sparse_vecs[i].values.tolist(),
            ),
        },
        payload={"text": chunks[i]},
    )
    for i in range(len(chunks))
]
client.upsert("hybrid_docs", points=points)

# Query hybrid (dùng prefetch + fusion)
results = client.query_points(
    collection_name="hybrid_docs",
    prefetch=[
        models.Prefetch(query=dense_query, using="", limit=20),
        models.Prefetch(
            query=models.SparseVector(indices=..., values=...),
            using="bm25",
            limit=20,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),  # RRF fusion
    limit=3,
)
```

---

## 5️⃣  Reranking — Nâng chất lượng top-k

### Vì sao cần rerank?

```
Bi-encoder (embedding model):
  → Encode query và chunk RIÊNG BIỆT → so sánh vector
  → Nhanh nhưng "xấp xỉ"

Cross-encoder (reranker):
  → Đọc query + chunk CÙNG LÚC → đánh giá trực tiếp mức độ liên quan
  → Chậm nhưng CHÍNH XÁC hơn nhiều

Pipeline tối ưu:
  Qdrant → top-20 (bi-encoder, nhanh)
    → Reranker → top-3 (cross-encoder, chính xác)
      → LLM với 3 chunks chất lượng cao
```

### Code Reranking

```python
from sentence_transformers import CrossEncoder

# Load reranker model (nhỏ, chạy local được)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def search_with_rerank(query: str, collection_name: str,
                       retrieve_k: int = 20, final_k: int = 3):
    """Retrieve top-20, rerank, trả về top-3 chất lượng cao."""
    
    # 1. Retrieve rộng (top-20)
    q_vec = model.encode(query).tolist()
    results = client.query_points(
        collection_name=collection_name, query=q_vec, limit=retrieve_k
    )
    candidates = results.points
    
    # 2. Rerank bằng cross-encoder
    pairs = [(query, c.payload["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)
    
    # 3. Sắp xếp lại theo rerank score
    reranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    
    # 4. Lấy top-3 sau rerank
    print(f"\n🔍 Query: '{query}'")
    print(f"   Retrieve {retrieve_k} → Rerank → Top {final_k}:\n")
    
    for i, (point, score) in enumerate(reranked[:final_k], 1):
        original_rank = [c.id for c in candidates].index(point.id) + 1
        rank_change = original_rank - i
        arrow = f"↑{rank_change}" if rank_change > 0 else f"↓{-rank_change}" if rank_change < 0 else "="
        
        print(f"  {i}. [rerank={score:.4f}] (was rank {original_rank} {arrow})")
        print(f"     {point.payload['text'][:120]}...")
    
    return [point for point, _ in reranked[:final_k]]


# Sử dụng:
top3 = search_with_rerank("Cách điều trị tiểu đường type 2?", "my_docs")
```

**Output ví dụ:**
```
🔍 Query: 'Cách điều trị tiểu đường type 2?'
   Retrieve 20 → Rerank → Top 3:

  1. [rerank=9.2341] (was rank 3 ↑2)
     Insulin là phương pháp điều trị chính cho tiểu đường type 2. Bệnh nhân cần kết hợp...

  2. [rerank=8.7812] (was rank 1 ↓1)
     Bệnh tiểu đường type 2 là tình trạng cơ thể không sử dụng insulin hiệu quả...

  3. [rerank=7.1203] (was rank 7 ↑4)
     Chế độ ăn ít đường, tập thể dục thường xuyên giúp kiểm soát đường huyết...

→ Chunk rank 7 (bị "chôn" ở vector search) nhảy lên top-3 nhờ reranker!
```

### Trade-off: Rerank có đáng không?

```
Benchmark (trên 20 queries, 1000 chunks):

                    P@3     Latency
Vector-only:        0.60    12ms
Vector + Rerank:    0.83    45ms  (+33ms)
                    ↑ +38%  ↑ +275%

→ Precision tăng 38% chỉ tốn thêm 33ms
→ Hoàn toàn đáng với hầu hết use case RAG
```

---

## ✅  Checklist hoàn thành Tuần 5

- [ ] Tạo được bộ test 10 câu hỏi, tính Precision@3
- [ ] So sánh chunk_size (200 vs 500 vs 1000) → kết luận chunk_size tối ưu
- [ ] Endpoint `/ask` trả về answer + sources
- [ ] Upload file qua `/index` bằng `UploadFile` + `BytesIO`
- [ ] Hybrid search chạy được, so sánh với dense-only
- [ ] Reranking tích hợp vào pipeline, đo được cải thiện P@3
- [ ] Giải thích được: **"Tại sao prompt RAG phải nói 'chỉ trả lời dựa trên context'?"**

> Vì nếu không có ràng buộc này, LLM sẽ trộn kiến thức training data với tài liệu → **hallucination** trông rất thuyết phục nhưng hoàn toàn sai.
