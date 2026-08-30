# Embedding Model – Cheatsheet (Ngày 4)

---

## 1️⃣  Embedding là gì?

**Embedding** = chuyển đổi text → vector số (mảng float) có chiều cố định.

```
"The patient has diabetes"  →  [0.12, -0.45, 0.78, ..., 0.33]  (384 chiều)
"Patient diagnosed with DM" →  [0.11, -0.43, 0.81, ..., 0.31]  (384 chiều)
"I love playing football"   →  [-0.55, 0.22, -0.10, ..., 0.67]  (384 chiều)
```

**Ý tưởng cốt lõi:**
- Văn bản **cùng nghĩa** → vector **gần nhau** trong không gian n chiều
- Văn bản **khác nghĩa** → vector **xa nhau**
- Không cần trùng từng chữ → tìm được câu trả lời dù query dùng từ khác

```
Query:  "How to treat high blood sugar?"
Chunk:  "Insulin therapy is the standard treatment for diabetes."
→ similarity = 0.87  ✅  (tìm được dù không chung từ nào)
```

---

## 2️⃣  Dense vs Sparse Vector

| | Dense vector | Sparse vector (BM25/TF-IDF) |
|---|---|---|
| **Cấu trúc** | Mỗi chiều đều có giá trị | Phần lớn = 0, vài chiều khác 0 |
| **Chiều** | 384 – 3072 | Bằng vocabulary size (hàng chục nghìn) |
| **Tìm theo** | Ý nghĩa (semantic) | Keyword chính xác |
| **Ví dụ** | sentence-transformers, OpenAI | Elasticsearch, BM25 |
| **Dùng khi** | RAG, semantic search | Full-text search, keyword match |

```python
# Dense: mỗi chiều đều mang giá trị
dense = [0.12, -0.45, 0.78, 0.33, -0.21, ...]   # 384 số, gần như không có 0

# Sparse: hầu hết = 0, chỉ các từ xuất hiện mới ≠ 0
sparse = {1024: 0.8, 5372: 0.3, 12891: 0.6}    # index: tf-idf score
```

---

## 3️⃣  Cosine Similarity

> Đo **góc** giữa 2 vector, không phải khoảng cách Euclidean.
> Giá trị từ **-1 đến 1**: `1` = giống hệt, `0` = không liên quan, `-1` = đối lập.

### Công thức:

```
                  A · B
cos(θ) = ─────────────────────
              ||A|| × ||B||
```

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Ví dụ
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

vec_a = model.encode("The patient has Type 2 diabetes.")
vec_b = model.encode("Patient was diagnosed with high blood sugar.")
vec_c = model.encode("I enjoy hiking in the mountains.")

print(cosine_similarity(vec_a, vec_b))  # → 0.87  (cùng chủ đề)
print(cosine_similarity(vec_a, vec_c))  # → 0.11  (khác chủ đề)
```

### Tại sao dùng cosine thay vì Euclidean distance?

```
Vector A = [1, 0]     (chiều dài = 1)
Vector B = [10, 0]    (chiều dài = 10, cùng hướng với A)

Euclidean(A, B) = 9.0    ← "khác nhau nhiều"
Cosine(A, B)    = 1.0    ← "giống nhau hoàn toàn"  ✅

→ Cosine quan tâm đến HƯỚNG, không phải ĐỘ LỚN của vector
→ Cùng ý nghĩa dù câu dài hay ngắn
```

---

## 4️⃣  Chọn Embedding Model

| Model | Chiều | Local/API | Ngôn ngữ | Tốc độ | Chất lượng |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Local | 🇬🇧 EN | ⚡⚡⚡ | ★★★ |
| `all-mpnet-base-v2` | 768 | Local | 🇬🇧 EN | ⚡⚡ | ★★★★ |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Local | 🌏 50+ langs | ⚡⚡ | ★★★ |
| `nomic-embed-text` | 768 | Local (Ollama) | 🌏 Multi | ⚡⚡ | ★★★★ |
| `text-embedding-3-small` (OpenAI) | 1536 | API | 🌏 Multi | ⚡ (network) | ★★★★ |
| `text-embedding-3-large` (OpenAI) | 3072 | API | 🌏 Multi | ⚡ (network) | ★★★★★ |

### Khi nào chọn model nào:

```
Prototype / học / local dev     → all-MiniLM-L6-v2  (nhẹ, nhanh)
Cần chất lượng cao hơn, vẫn local → all-mpnet-base-v2
Text tiếng Việt / đa ngôn ngữ  → paraphrase-multilingual-MiniLM-L12-v2
Production, budget API          → text-embedding-3-small (OpenAI)
Production, cần tốt nhất        → text-embedding-3-large
```

---

## 5️⃣  Cài đặt & sử dụng

```bash
pip install sentence-transformers scikit-learn numpy
```

### 5.1  Embed một câu

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

vec = model.encode("Hello world")
print(vec.shape)   # (384,)
print(vec[:5])     # [-0.067,  0.042, -0.023,  0.089, -0.031]
```

### 5.2  Embed nhiều câu (batch – hiệu quả hơn)

```python
sentences = [
    "The patient has Type 2 diabetes.",
    "Insulin is used to manage blood sugar.",
    "The cat sat on the mat.",
    "I enjoy hiking in the mountains.",
    "Blood glucose levels must be monitored daily.",
]

# Batch encode – nhanh hơn nhiều lần so với encode từng câu
vectors = model.encode(sentences, batch_size=32, show_progress_bar=False)
print(vectors.shape)  # (5, 384)
```

### 5.3  Tính similarity toàn bộ cặp (similarity matrix)

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sim_matrix = cosine_similarity(vectors)

# In dạng bảng
print(f"{'':30}", end="")
for s in sentences:
    print(f"{s[:12]:>14}", end="")
print()

for i, row in enumerate(sim_matrix):
    print(f"{sentences[i][:30]:30}", end="")
    for val in row:
        print(f"{val:>14.3f}", end="")
    print()
```

**Output mẫu:**
```
                               The patient  Insulin is  The cat sat  I enjoy hik  Blood gluco
The patient has Type 2 diabet        1.000       0.821        0.102        0.089        0.873
Insulin is used to manage blo        0.821       1.000        0.091        0.077        0.841
The cat sat on the mat.              0.102       0.091        1.000        0.312        0.098
I enjoy hiking in the mounta         0.089       0.077        0.312        1.000        0.081
Blood glucose levels must be         0.873       0.841        0.098        0.081        1.000
```

> Nhận xét: 3 câu về y tế (hàng 1, 2, 5) có similarity 0.82–0.87 với nhau,
> trong khi với 2 câu không liên quan chỉ 0.08–0.10.

---

## 6️⃣  Cơ chế RAG Retrieval – ví dụ tự tay

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

# "Database" (chunks từ document)
chunks = [
    "Phở is a Vietnamese noodle soup consisting of broth, rice noodles, herbs.",
    "To make phở broth, simmer beef bones for 6-8 hours with spices.",
    "The Eiffel Tower is located in Paris, France.",
    "Machine learning models require large amounts of training data.",
    "Phở originated in northern Vietnam in the early 20th century.",
]

query = "Làm thế nào để nấu phở?"   # query tiếng Việt (model có thể xử lý phần nào)

# Embed tất cả
chunk_vecs = model.encode(chunks)
query_vec  = model.encode([query])

# Tính similarity
scores = cosine_similarity(query_vec, chunk_vecs)[0]

# Sắp xếp và in top-k
top_k = 3
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

print(f"Query: '{query}'\n")
print(f"Top {top_k} chunks retrieved:")
for rank, (idx, score) in enumerate(ranked, 1):
    print(f"  {rank}. [score={score:.3f}] {chunks[idx]}")
```

**Output:**
```
Query: 'Làm thế nào để nấu phở?'

Top 3 chunks retrieved:
  1. [score=0.612] To make phở broth, simmer beef bones for 6-8 hours with spices.
  2. [score=0.587] Phở is a Vietnamese noodle soup consisting of broth, rice noodles, herbs.
  3. [score=0.521] Phở originated in northern Vietnam in the early 20th century.
```

> ✅ Tìm được đúng 3 chunk liên quan dù query tiếng Việt, chunks tiếng Anh.
> ❌ "Eiffel Tower" và "ML models" không được retrieve.

---

## 7️⃣  Đo tốc độ embed

```python
import time
from sentence_transformers import SentenceTransformer

model  = SentenceTransformer("all-MiniLM-L6-v2")
chunks = ["Sample chunk text " * 20] * 100   # 100 chunks giả lập

# Single encode (vòng lặp)
start = time.time()
for c in chunks:
    model.encode(c)
t_single = time.time() - start

# Batch encode
start = time.time()
model.encode(chunks, batch_size=32)
t_batch = time.time() - start

print(f"Single loop : {t_single:.2f}s  ({100/t_single:.0f} chunks/s)")
print(f"Batch encode: {t_batch:.2f}s  ({100/t_batch:.0f} chunks/s)")
print(f"Speedup     : {t_single/t_batch:.1f}x")
```

**Output mẫu (CPU):**
```
Single loop : 8.43s  (12 chunks/s)
Batch encode: 1.21s  (83 chunks/s)
Speedup     : 7.0x
```

> 💡 **Luôn dùng batch encode** khi xử lý nhiều chunks. Nhanh hơn 5–10x.

---

## 8️⃣  Gotchas & Best Practices

### ❗ Max sequence length

```python
# all-MiniLM-L6-v2 chỉ xử lý tối đa 256 tokens
# Nếu chunk dài hơn → BỊ CẮT NGẦM, không báo lỗi!

model.max_seq_length  # → 256

# Kiểm tra chunk có bị cắt không:
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

for chunk in chunks:
    tokens = tokenizer.encode(chunk)
    if len(tokens) > 256:
        print(f"⚠️  Chunk quá dài: {len(tokens)} tokens → sẽ bị cắt!")
```

| Model | Max tokens |
|---|---|
| `all-MiniLM-L6-v2` | 256 |
| `all-mpnet-base-v2` | 514 |
| `nomic-embed-text` | 8192 |
| `text-embedding-3-small` | 8191 |

### ❗ Normalize vector trước khi dùng cosine

```python
# sentence-transformers mặc định đã normalize (L2 norm = 1)
# Nếu dùng raw output từ transformers → phải normalize thủ công:
import numpy as np

def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

# Sau khi normalize: cosine_similarity = dot product (nhanh hơn)
score = np.dot(normalize(vec_a), normalize(vec_b))
```

### ❗ Không embed query và chunk bằng 2 model khác nhau

```
✅ Query: model A   |  Chunks: model A
❌ Query: model A   |  Chunks: model B  → similarity vô nghĩa
```

---

## 9️⃣  Quick Reference

```python
# ─── Minimal working example ───────────────────────────────────────
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model  = SentenceTransformer("all-MiniLM-L6-v2")
chunks = ["chunk 1 text...", "chunk 2 text...", "chunk 3 text..."]
query  = "your query here"

# Embed
chunk_vecs = model.encode(chunks, batch_size=32)   # (n, 384)
query_vec  = model.encode([query])                 # (1, 384)

# Retrieve top-3
scores  = cosine_similarity(query_vec, chunk_vecs)[0]
top3    = np.argsort(scores)[::-1][:3]

for rank, idx in enumerate(top3, 1):
    print(f"{rank}. [score={scores[idx]:.3f}] {chunks[idx]}")
# ───────────────────────────────────────────────────────────────────
```

---

## ✅  Kiểm tra hiểu bài

Trả lời được 3 câu này → bạn đã nắm vững:

1. **"Dense vector là gì?"**
   > Mảng số thực có chiều cố định (384/768...), hầu hết các chiều ≠ 0, đại diện cho ý nghĩa của text trong không gian nhiều chiều.

2. **"Cosine similarity đo cái gì?"**
   > Đo góc giữa 2 vector – quan tâm đến hướng (ý nghĩa), không phải độ lớn (độ dài câu). Giá trị 1 = cùng hướng = cùng nghĩa.

3. **"Tại sao query và answer không cần trùng từ mà vẫn tìm được?"**
   > Vì embedding model học được semantic relationship – "diabetes" và "high blood sugar" được encode vào vùng gần nhau trong không gian vector, nên cosine similarity của chúng cao dù không chung từ nào.
