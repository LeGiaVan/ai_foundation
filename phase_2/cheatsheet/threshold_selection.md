# Chọn `similarity_threshold` cho Semantic Chunking

---

## 1️⃣  Nguyên tắc cơ bản

```
similarity_score = cosine_similarity(embedding[câu_i], embedding[câu_i+1])

score < threshold  →  TÁCH  (đặt breakpoint giữa 2 câu)
score >= threshold →  GIỮ   (2 câu thuộc cùng chunk)
```

**Hệ quả trực tiếp:**

| | Threshold thấp | Threshold cao |
|---|---|---|
| **Breakpoint xảy ra khi** | Chỉ khi cực kỳ khác nhau | Chỉ cần hơi khác nhau |
| **Số chunks** | Ít, mỗi chunk lớn | Nhiều, mỗi chunk nhỏ |
| **Rủi ro** | Mất ranh giới topic | Mất context ở boundary, tốn tài nguyên |
| **Thích hợp** | Narrative dài, cần chunk lớn | Ít khi cần, chỉ dùng khi granularity cao |

> **Sweet spot thực tế: 0.68 – 0.78** cho hầu hết use-case.

---

## 2️⃣  Tại sao không đoán threshold?

Cùng một giá trị threshold nhưng cho kết quả **hoàn toàn khác nhau** tùy document:

```
Document A (technical manual) – các câu liên kết chặt:
  sim scores: [0.88, 0.91, 0.85, 0.29, 0.87, 0.90]
  threshold = 0.75 → chỉ tách 1 chỗ (score 0.29) ← ổn

Document B (mixed blog) – chủ đề nhảy liên tục:
  sim scores: [0.71, 0.68, 0.45, 0.72, 0.51, 0.69]
  threshold = 0.75 → tách 5/6 cặp → mỗi câu 1 chunk ← quá nhiều!
```

→ **Phải phân tích distribution của từng document** trước khi chọn.

---

## 3️⃣  Quy trình 4 bước

```
BƯỚC 1 │ Phân tích similarity distribution  (bắt buộc)
        │ → lấy P25, P50, P75 làm điểm tham chiếu
        ↓
BƯỚC 2 │ Sweep threshold trong vùng P25 → P75
        │ → quan sát số chunks và kích thước trung bình
        ↓
BƯỚC 3 │ Visual inspection
        │ → xem boundary chunk có tự nhiên không
        ↓
BƯỚC 4 │ Đo Recall@k  (chỉ cần nếu có ground truth queries)
        │ → metric khách quan duy nhất
```

---

## 4️⃣  BƯỚC 1 – Phân tích similarity distribution

> **Mục tiêu:** hiểu vùng similarity của document để xác định ngưỡng sweep hợp lý.

```python
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def analyze_distribution(text: str, model_name: str = "all-MiniLM-L6-v2") -> dict:
    model     = SentenceTransformer(model_name)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]

    if len(sentences) < 3:
        raise ValueError("Cần ít nhất 3 câu.")

    embs   = model.encode(sentences, show_progress_bar=False)
    scores = [float(cosine_similarity([embs[i]], [embs[i+1]])[0][0])
              for i in range(len(embs) - 1)]

    p25, p50, p75 = np.percentile(scores, [25, 50, 75])

    print("=" * 52)
    print("📊  Similarity Distribution")
    print("=" * 52)
    print(f"  Sentences  : {len(sentences)}")
    print(f"  Min        : {min(scores):.3f}  ← topic shift mạnh nhất")
    print(f"  P25        : {p25:.3f}  ← mốc aggressive")
    print(f"  P50 median : {p50:.3f}  ← mốc balanced")
    print(f"  P75        : {p75:.3f}  ← mốc conservative")
    print(f"  Max        : {max(scores):.3f}  ← 2 câu giống nhau nhất")
    print()
    print(f"  → Sweep từ {p25:.2f} đến {p75:.2f}  |  bắt đầu xem từ {p50:.2f}")
    print("=" * 52)

    return {"p25": p25, "p50": p50, "p75": p75, "scores": scores}
```

### Ý nghĩa của từng percentile

Giả sử document có **11 cặp câu liên tiếp** với scores:

```
Scores sắp xếp tăng dần:
0.28  0.31  0.63  0.71  0.75  0.78  0.82  0.85  0.88  0.91  0.93
              ↑P25        ↑P50              ↑P75
```

| Percentile | Giá trị | Nghĩa khi dùng làm threshold |
|---|---|---|
| **P25 = 0.63** | Aggressive | threshold < 25% scores → chỉ **2–3 chỗ** bị tách |
| **P50 = 0.78** | Balanced | threshold < 50% scores → **~5–6 chỗ** bị tách |
| **P75 = 0.85** | Conservative | threshold < 75% scores → **~8 chỗ** bị tách → quá nhiều |

> ⚠️ **Sweet spot thực tế thường gần P25–P40**, không phải P75.
> P75 là biên trên của vùng sweep, không phải giá trị nên chọn.

### Ví dụ output thực tế

**Input:** 12 câu medical record (9 câu clinical + 3 câu về ML):

```
==================================================
📊  Similarity Distribution
==================================================
  Sentences  : 12
  Min        : 0.281  ← "Post-procedure..." ↔ "ML models..."  (topic shift mạnh)
  P25        : 0.634  ← mốc aggressive
  P50 median : 0.781  ← mốc balanced
  P75        : 0.843  ← mốc conservative
  Max        : 0.932  ← "hypertension" ↔ "blood pressure"  (cùng topic)

  → Sweep từ 0.63 đến 0.84  |  bắt đầu xem từ 0.78
==================================================
```

**Đọc kết quả:**
- `Min = 0.281` → chỗ này **chắc chắn bị tách** dù threshold có thấp
- `Max = 0.932` → chỗ này **không bao giờ bị tách** dù threshold cao
- `P50 = 0.781` → dùng làm điểm bắt đầu sweep, **không phải điểm chọn cuối**

---

## 5️⃣  BƯỚC 2 – Sweep threshold

> **Mục tiêu:** tìm vùng threshold cho số chunks hợp lý.

```python
import numpy as np
from langchain_community.text_splitters import SemanticChunker
from sentence_transformers import SentenceTransformer

def sweep_threshold(text: str, p25: float, p75: float, step: float = 0.05,
                    model_name: str = "all-MiniLM-L6-v2") -> dict:
    model      = SentenceTransformer(model_name)
    thresholds = list(np.arange(round(p25, 2), round(p75 + step, 2), step))

    print(f"{'Threshold':>10} | {'# Chunks':>8} | {'Avg chars':>10} | {'Min':>5} | {'Max':>5}")
    print("-" * 52)

    results = {}
    for thresh in thresholds:
        splitter = SemanticChunker(
            embedding_function=model.encode,
            similarity_threshold=float(thresh),
        )
        chunks = splitter.split_text(text)
        sizes  = [len(c) for c in chunks]
        avg    = int(np.mean(sizes))
        flag   = " ✅" if 4 <= len(chunks) <= 15 else (" ⚠️ few" if len(chunks) < 4 else " ⚠️ many")
        print(f"{thresh:>10.2f} | {len(chunks):>8d} | {avg:>10d} | {min(sizes):>5d} | {max(sizes):>5d}{flag}")
        results[float(thresh)] = {"n": len(chunks), "avg": avg, "chunks": chunks}

    return results
```

### Ví dụ output sweep (tiếp theo ví dụ medical ở trên):

```
Threshold | # Chunks | Avg chars |  Min |  Max
----------------------------------------------------
     0.63 |        2 |      2600 | 1800 | 3400  ⚠️ few
     0.68 |        4 |      1300 |  600 | 2100  ✅
     0.73 |        6 |       870 |  300 | 1500  ✅
     0.78 |        8 |       650 |  200 | 1200  ✅
     0.83 |       14 |       370 |   80 |  900  ⚠️ many
```

**Đọc kết quả:**
- `0.68` → 4 chunks: clinical summary / ECG / procedure / ML section → **tốt nếu muốn chunk lớn**
- `0.73` → 6 chunks: granularity vừa → **thường là sweet spot**
- `0.78` → 8 chunks: chi tiết hơn, vẫn ok
- `0.83` → 14 chunks: quá nhỏ, mất context

→ **Đưa 0.68, 0.73, 0.78 vào Bước 3** để visual inspect.

---

## 6️⃣  BƯỚC 3 – Visual Inspection

> **Mục tiêu:** xác nhận boundary chunk tự nhiên, không bị cắt giữa ý.

```python
def visual_inspect(results: dict, thresh: float, n: int = 5):
    chunks = results[thresh]["chunks"]
    print(f"\n{'='*60}")
    print(f"🔍  threshold={thresh}  |  {len(chunks)} chunks")
    print(f"{'='*60}")
    for i, c in enumerate(chunks[:n]):
        c = c.strip()
        print(f"\n┌─ Chunk {i+1} ({len(c)} chars) ─────────────────")
        print(f"│ START: {c[:90]}...")
        print(f"│ END  : ...{c[-90:]}")
        print(f"└{'─'*55}")
```

### Checklist boundary tốt:

- [ ] **START** không bắt đầu bằng "This / It / They / However" không có antecedent
- [ ] **END** kết thúc ở cuối câu (`.` hoặc `?` hoặc `!`), không giữa chừng
- [ ] Chunk không cắt ngang danh sách, bảng biểu, hay definition
- [ ] Kích thước nằm trong khoảng **200–1200 chars** (tùy use-case)

### Ví dụ boundary tốt vs xấu:

```
✅ Boundary tốt (threshold=0.73):
   Chunk 1 END  : "...A diagnosis of inferior STEMI was confirmed."
   Chunk 2 START: "The patient was taken to the cath lab within 90 minutes..."

❌ Boundary xấu (threshold=0.83):
   Chunk 4 END  : "...The patient was taken to the cath lab within"
   Chunk 5 START: "90 minutes. PCI was performed on..."  ← cắt giữa câu!
```

---

## 7️⃣  BƯỚC 4 – Đo Recall@k *(chỉ cần nếu có ground truth)*

> **Mục tiêu:** chọn threshold bằng metric khách quan thay vì cảm tính.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from langchain_community.text_splitters import SemanticChunker
from sentence_transformers import SentenceTransformer

def evaluate_recall(
    text: str,
    test_queries: list[dict],  # [{"query": "...", "answer": "..."}]
    thresholds: list[float],
    k: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
):
    """
    Recall@k: câu trả lời có xuất hiện trong top-k chunks được retrieve không?
    """
    model  = SentenceTransformer(model_name)
    encode = model.encode

    print(f"{'Threshold':>10} | {'Recall@'+str(k):>10} | Note")
    print("-" * 40)

    for thresh in thresholds:
        splitter   = SemanticChunker(embedding_function=encode, similarity_threshold=thresh)
        chunks     = splitter.split_text(text)
        chunk_embs = encode(chunks)
        hits = 0

        for item in test_queries:
            scores    = cos_sim(encode([item["query"]]), chunk_embs)[0]
            top_k     = [chunks[i] for i in np.argsort(scores)[::-1][:k]]
            if any(item["answer"][:50] in c for c in top_k):
                hits += 1

        recall = hits / len(test_queries)
        note   = "✅ BEST" if recall >= 0.9 else ("⚠️" if recall < 0.7 else "")
        print(f"{thresh:>10.2f} | {recall:>10.2%} | {note}")
```

### Tạo test queries nhanh:
```python
# Lấy một đoạn ngắn quan trọng trong document làm "answer"
test_queries = [
    {"query": "when was patient taken to cath lab",  "answer": "within 90 minutes"},
    {"query": "what procedure was performed",         "answer": "PCI was performed"},
    {"query": "what does ML model predict",           "answer": "patient readmission risk"},
]
```

---

## 8️⃣  Quick Reference – Theo loại document

| Document type | Gợi ý threshold | Lý do |
|---|---|---|
| Legal / Contract | **0.68–0.73** | Câu liên kết chặt, tránh over-split |
| Medical record | **0.70–0.75** | Cùng lý do, event-based chunking |
| Research paper | **0.67–0.73** | Section dài, cần chunk đủ lớn |
| News / Blog | **0.72–0.78** | Topic thay đổi rõ ràng giữa đoạn |
| FAQ / Q&A | **0.60–0.68** | Mỗi Q&A là unit riêng biệt |
| Code documentation | **0.65–0.72** | Function/class là ranh giới tự nhiên |
| **Unknown / Mixed** | **0.72** | Điểm khởi đầu an toàn nhất |

---

## 9️⃣  Template – Copy-paste & chạy ngay

```python
# threshold_finder.py
import numpy as np, re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.text_splitters import SemanticChunker

TEXT = """Paste your document text here..."""

model     = SentenceTransformer("all-MiniLM-L6-v2")
sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", TEXT) if len(s.strip()) > 20]
embs      = model.encode(sentences)
scores    = [float(cosine_similarity([embs[i]], [embs[i+1]])[0][0]) for i in range(len(embs)-1)]

p25, p50, p75 = np.percentile(scores, [25, 50, 75])
print(f"P25={p25:.2f}  P50={p50:.2f}  P75={p75:.2f}  → sweep {p25:.2f} đến {p75:.2f}")

# Sweep
print(f"\n{'Threshold':>10} | {'# Chunks':>8} | {'Avg chars':>10}")
for thresh in np.arange(round(p25, 2), round(p75 + 0.05, 2), 0.05):
    splitter = SemanticChunker(embedding_function=model.encode, similarity_threshold=float(thresh))
    chunks   = splitter.split_text(TEXT)
    sizes    = [len(c) for c in chunks]
    print(f"{thresh:>10.2f} | {len(chunks):>8d} | {int(np.mean(sizes)):>10d}")

# Visual inspect – thay CHOSEN bằng giá trị bạn chọn từ sweep
CHOSEN   = p25 + (p50 - p25) * 0.3   # ≈ P35, thường là sweet spot
splitter = SemanticChunker(embedding_function=model.encode, similarity_threshold=CHOSEN)
chunks   = splitter.split_text(TEXT)
print(f"\n🔍  CHOSEN threshold={CHOSEN:.2f}  →  {len(chunks)} chunks")
for i, c in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ({len(c)} chars) ---")
    print(c[:200].replace("\n", " ") + "...")
```

---

## ✅  Bottom-line

| # | Rule |
|---|---|
| 1 | **Không đoán** – phân tích distribution trước |
| 2 | **Sweep từ P25 → P75** để bao quát đủ vùng |
| 3 | **Sweet spot thường gần P25–P40**, không phải P50 hay P75 |
| 4 | **Visual inspect** là bước không thể bỏ – dù có metric |
| 5 | **Recall@k** là metric duy nhất đáng tin nếu có ground truth |
| 6 | **Validate trên ≥ 3 documents** trước khi dùng production |
