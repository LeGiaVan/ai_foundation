# Qdrant – Vector Database Cheatsheet (Ngày 5)

---

## 1️⃣  Qdrant (Vector DB) sinh ra để làm gì?

**Hiểu một cách đơn giản nhất:**
1. **Lưu trữ lâu dài (Persistence):** Thay vì mỗi lần chạy script Python phải embed lại từ đầu và lưu trên RAM (mất data khi tắt script), Qdrant lưu vector an toàn xuống ổ cứng để dùng lại vĩnh viễn.
2. **Tìm kiếm siêu tốc (HNSW):** Thay vì dùng vòng lặp for tính cosine similarity với từng vector một (vừa chậm vừa tốn tài nguyên), Qdrant dùng thuật toán HNSW để search cực kỳ tối ưu.

```
Bài toán: có 100,000 chunks đã embed → tìm 3 chunks gần nhất

Cách ngây thơ (Python list):
  for chunk in 100_000_chunks:
      score = cosine_similarity(query, chunk)  ← Quét toàn bộ, O(N), ngốn CPU

Cách của Qdrant (HNSW index):
  → Đi đường tắt trong đồ thị (graph) để tìm điểm gần nhất
  → Chỉ mất O(log N), nhanh hơn hàng trăm lần
```

**So sánh với SQL:**

| | SQL Database | Vector Database (Qdrant) |
|---|---|---|
| **Lưu** | Rows (text, number, date...) | Vectors (float arrays) + metadata |
| **Tìm kiếm** | `WHERE name = 'John'` (exact) | Nearest neighbor (approximate) |
| **Metric** | Exact match, range | Cosine / Dot product / Euclidean |
| **Dùng cho** | Structured data | Semantic search, RAG |

---

## 2️⃣  Các khái niệm cốt lõi

```
Qdrant
  └── Collection  (≈ Table trong SQL)
        └── Point  (≈ Row trong SQL)
              ├── id      : int hoặc UUID (unique)
              ├── vector  : [0.12, -0.45, 0.78, ...]  (384 chiều)
              └── payload : {"text": "...", "source": "cv.pdf", "page": 3}
```

### Collection
- Chứa tất cả Points có cùng **vector size** và **distance metric**
- Tạo 1 lần, insert nhiều lần

### Point
- Đơn vị lưu trữ cơ bản = 1 chunk đã embed
- `id` → dùng để update/delete
- `vector` → dùng để search
- `payload` → trả về khi tìm thấy, giúp lấy text gốc về

### Payload
```python
# Thiết kế payload tốt:
payload = {
    "text": "The patient has Type 2 diabetes...",   # ← text gốc để trả về cho LLM
    "source": "medical_record.pdf",                  # ← file nguồn
    "page": 3,                                       # ← số trang
    "chunk_id": 15,                                  # ← vị trí chunk
    "section": "diagnosis",                          # ← metadata tùy chọn
}
```

### HNSW (Hierarchical Navigable Small World)
```
Index mặc định của Qdrant – thuật toán Approximate Nearest Neighbor:

Build time: tạo một graph đa tầng kết nối các vector gần nhau
Query time: đi từ tầng cao (coarse) xuống tầng thấp (fine) → nhanh

Tham số quan trọng:
  m         = số cạnh mỗi node (mặc định 16) → tăng → chậm hơn nhưng chính xác hơn
  ef_construct = độ rộng tìm kiếm lúc build (mặc định 100)
  → Với RAG thông thường: dùng default là đủ
```

### Distance Metric
| Metric | Khi nào dùng |
|---|---|
| `COSINE` | Sentence-transformers (đã normalize) ✅ phổ biến nhất |
| `DOT` | OpenAI embeddings, hoặc khi đã normalize |
| `EUCLID` | Ít dùng trong NLP |

---

## 3️⃣  Cài đặt

```bash
# Option 1: Docker Compose (recommended – persist data, dễ quản lý)
# Tại thư mục phase_2 (nơi chứa file docker-compose.yml), chạy:
docker-compose up -d

# Khi cần tắt:
docker-compose down

# Option 2: In-memory (chỉ dùng để test, mất data khi tắt)
# → dùng QdrantClient(":memory:") trong Python, không cần Docker

# Cài Python client
pip install qdrant-client sentence-transformers
```

**Verify Qdrant đang chạy:**
```bash
curl http://localhost:6333/collections
# → {"result":{"collections":[]},"status":"ok"}
```
hoặc mở browser: `http://localhost:6333/dashboard`

---

## 4️⃣  Kết nối client

```python
from qdrant_client import QdrantClient

# Kết nối đến Qdrant đang chạy qua Docker
client = QdrantClient(host="localhost", port=6333)

# In-memory (không cần Docker, dùng khi test nhanh)
client = QdrantClient(":memory:")

# Qdrant Cloud
client = QdrantClient(
    url="https://xxxx.us-east.aws.cloud.qdrant.io",
    api_key="your-api-key",
)
```

---

## 5️⃣  Tạo Collection

```python
from qdrant_client.models import Distance, VectorParams

client.create_collection(
    collection_name="medical_docs",
    vectors_config=VectorParams(
        size=384,             # phải khớp với embedding model dimension
        distance=Distance.COSINE,
    ),
)

# Kiểm tra collection đã tạo:
print(client.get_collections())
# → CollectionsResponse(collections=[CollectionDescription(name='medical_docs')])
```

> ⚠️ `size` phải khớp chính xác với model:
> - `all-MiniLM-L6-v2` → 384
> - `all-mpnet-base-v2` → 768
> - `text-embedding-3-small` → 1536

---

## 6️⃣  Upsert Points (insert/update)

```python
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

model  = SentenceTransformer("all-MiniLM-L6-v2")

chunks = [
    "The patient has Type 2 diabetes and hypertension.",
    "Insulin therapy is the standard treatment for diabetes.",
    "Blood pressure should be monitored daily.",
    "Machine learning can predict patient readmission risk.",
    "The Eiffel Tower is located in Paris, France.",
]

# Embed tất cả (batch)
vectors = model.encode(chunks, batch_size=32, show_progress_bar=False)

# Tạo danh sách Points
points = [
    PointStruct(
        id=i,
        vector=vectors[i].tolist(),   # numpy array → list[float]
        payload={
            "text": chunks[i],
            "source": "example.pdf",
            "chunk_id": i,
        }
    )
    for i in range(len(chunks))
]

# Upsert vào collection
client.upsert(
    collection_name="medical_docs",
    points=points,
)

print(f"Upserted {len(points)} points")
```

---

## 7️⃣  Vector Search (query)

```python
query = "How is diabetes treated?"
query_vec = model.encode(query).tolist()

results = client.query_points(
    collection_name="medical_docs",
    query=query_vec,
    limit=3,              # top-k = 3
)

for i, result in enumerate(results.points, 1):
    print(f"\n{i}. [score={result.score:.4f}]")
    print(f"   text: {result.payload['text']}")
    print(f"   id  : {result.id}")
```

**Output:**
```
1. [score=0.8912]
   text: Insulin therapy is the standard treatment for diabetes.
   id  : 1

2. [score=0.8134]
   text: The patient has Type 2 diabetes and hypertension.
   id  : 0

3. [score=0.7203]
   text: Blood pressure should be monitored daily.
   id  : 2
```

---

## 8️⃣  Filter + Search (kết hợp metadata)

> Qdrant cho phép filter theo payload **trước khi** search vector.

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Chỉ tìm trong documents từ file "medical_record.pdf"
results = client.query_points(
    collection_name="medical_docs",
    query=query_vec,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value="medical_record.pdf"),
            )
        ]
    ),
    limit=3,
)

# Filter theo range (vd: chỉ lấy page >= 5)
from qdrant_client.models import Range
results = client.query_points(
    collection_name="medical_docs",
    query=query_vec,
    query_filter=Filter(
        must=[FieldCondition(key="page", range=Range(gte=5))]
    ),
    limit=3,
)
```

---

## 9️⃣  CRUD Operations

```python
# ── Xem thông tin collection ──────────────────────────
info = client.get_collection("medical_docs")
print(info.points_count)          # số points đang có
print(info.config.params.vectors) # vector size, distance

# ── Lấy point theo id ────────────────────────────────
points = client.retrieve(
    collection_name="medical_docs",
    ids=[0, 1, 2],
    with_payload=True,
    with_vectors=False,   # thường không cần lấy vector về
)

# ── Update payload ────────────────────────────────────
client.set_payload(
    collection_name="medical_docs",
    payload={"reviewed": True, "section": "diagnosis"},
    points=[0, 1],        # update point id 0 và 1
)

# ── Xóa points ────────────────────────────────────────
client.delete(
    collection_name="medical_docs",
    points_selector=[3, 4],   # xóa point id 3 và 4
)

# ── Xóa toàn bộ collection ────────────────────────────
client.delete_collection("medical_docs")
```

---

## 🔟  Full Pipeline – Text → Qdrant → Query

```python
# qdrant_pipeline.py
import fitz
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Config ────────────────────────────────────────────
PDF_PATH        = "your_document.pdf"
COLLECTION_NAME = "my_docs"
MODEL_NAME      = "all-MiniLM-L6-v2"
VECTOR_SIZE     = 384
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50

# ── 1. Load PDF ───────────────────────────────────────
with fitz.open(PDF_PATH) as doc:
    raw_text = "\n".join(page.get_text("text") for page in doc)

# ── 2. Chunk ──────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_text(raw_text)
print(f"Chunks: {len(chunks)}")

# ── 3. Embed ──────────────────────────────────────────
model   = SentenceTransformer(MODEL_NAME)
vectors = model.encode(chunks, batch_size=32, show_progress_bar=True)

# ── 4. Upsert vào Qdrant ─────────────────────────────
client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(            # xóa nếu đã tồn tại, tạo mới
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)

points = [
    PointStruct(
        id=i,
        vector=vectors[i].tolist(),
        payload={"text": chunks[i], "source": PDF_PATH, "chunk_id": i},
    )
    for i in range(len(chunks))
]
client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Indexed {len(points)} chunks into '{COLLECTION_NAME}'")

# ── 5. Query ──────────────────────────────────────────
query     = "your question here"
q_vec     = model.encode(query).tolist()
results   = client.query_points(collection_name=COLLECTION_NAME, query=q_vec, limit=3)

print(f"\nQuery: '{query}'")
for i, r in enumerate(results.points, 1):
    print(f"\n{i}. [score={r.score:.4f}]")
    print(f"   {r.payload['text'][:200]}...")
```

---

## ✅  Checklist hoàn thành

- [ ] Qdrant chạy được qua Docker, dashboard mở được tại `localhost:6333/dashboard`
- [ ] Tạo collection với đúng `size` và `distance`
- [ ] Upsert ít nhất 10 points với payload chứa `text`
- [ ] Query trả về top-3 kết quả có score hợp lý
- [ ] Giải thích được: **payload dùng để làm gì?**

> **Payload** = nơi lưu text gốc + metadata. Vector chỉ dùng để *tìm*, payload mới là thứ *trả về* cho LLM khi RAG cần context.
