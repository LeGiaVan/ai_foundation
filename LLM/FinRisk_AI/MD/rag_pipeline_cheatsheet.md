# 📚 RAG Pipeline Cheatsheet — Phase 2

> **Mục đích:** Giải thích các khái niệm và kỹ thuật RAG được dùng trong FinRisk AI Phase 2.
> Đọc file này khi cần hiểu "tại sao làm vậy" thay vì chỉ biết "làm gì".

---

## 1. Dense vs Sparse Embedding — Tại sao cần cả 2?

| | Dense (BGE-M3) | Sparse (BM25) |
|---|---|---|
| **Cách hoạt động** | Encode văn bản thành vector 1024 chiều qua neural network | Tính trọng số từ khoá dựa trên tần suất xuất hiện (TF-IDF nâng cao) |
| **Điểm mạnh** | Hiểu ngữ nghĩa: "doanh thu" ≈ "revenue" → match được | Khớp từ khoá chính xác: "DSCR 1.2" phải có chữ "DSCR" |
| **Điểm yếu** | Bỏ sót số liệu cụ thể (2.8 ≈ 3.1 trong vector space) | Không hiểu đồng nghĩa: "lợi nhuận" ≠ "profit" |
| **Model** | `BAAI/bge-m3` via `fastembed` (~600MB) | `Qdrant/bm25` built-in fastembed |
| **Trong FinRisk AI** | Tìm đoạn văn liên quan đến câu hỏi | Tìm số liệu cụ thể (DSCR, Z-Score, năm 2023) |

**Hybrid = Dense + Sparse** → Vừa hiểu ngữ nghĩa, vừa khớp số liệu. Đây là standard trong RAG tài chính.

---

## 2. RRF — Reciprocal Rank Fusion

Khi có 2 danh sách kết quả (Dense list + Sparse list), cần gộp lại thành 1 danh sách duy nhất. RRF là thuật toán làm điều đó.

**Công thức RRF score:**
```
RRF_score(doc) = Σ 1 / (k + rank_i)
```
Trong đó `k = 60` (hằng số), `rank_i` là thứ hạng của doc trong list thứ i.

**Tại sao không đơn giản cộng score?**
- Dense score và Sparse score có đơn vị khác nhau (cosine similarity vs. BM25 score).
- RRF chỉ dùng thứ hạng (rank) → không phụ thuộc đơn vị → robust hơn.

**Trong code:** Qdrant xử lý RRF tự động khi dùng `FusionQuery(fusion="rrf")`.

---

## 3. Parent-Document Retriever — Tại sao cần?

```
Vấn đề: Embed chunk dài → vector "loãng", kém chính xác.
         Embed chunk ngắn → vector "tập trung", chính xác hơn.
         Nhưng chunk ngắn → LLM không đủ context để trả lời.
```

**Giải pháp: Child embed, Parent serve**

```
Parent (1000 tokens) — Lưu trong Qdrant payload
  ├── Child 1 (150 tokens) — Embed để search
  ├── Child 2 (150 tokens) — Embed để search
  └── Child 3 (150 tokens) — Embed để search

Search → tìm Child chính xác
         → trả về Parent đầy đủ cho LLM
```

**Kết quả:**
- Search **chính xác** (embed trên child ngắn, focused).
- LLM nhận **context đầy đủ** (parent chứa toàn bộ thông tin xung quanh).

---

## 4. Cross-Encoder Reranker

### Bi-Encoder (Dense Search) vs. Cross-Encoder (Reranker)

| | Bi-Encoder | Cross-Encoder |
|---|---|---|
| **Cách encode** | Query và Doc encode **riêng biệt** | Query và Doc encode **cùng nhau** |
| **Tốc độ** | Nhanh (pre-compute doc vectors) | Chậm hơn (phải chạy inference mỗi lần) |
| **Độ chính xác** | Tốt | Rất tốt (attention giữa query-doc) |
| **Dùng khi nào** | First-stage retrieval (k=20) | Second-stage rerank (n=4) |

**Pipeline trong FinRisk AI:**
```
Query → Bi-Encoder (BGE-M3) → Top-20 candidates → Cross-Encoder → Top-4 → LLM
```
Cross-Encoder đọc từng cặp `(query, child_text)` → cho điểm 0-1 → chọn Top-4.

**Model:** `BAAI/bge-reranker-v2-m3` — hỗ trợ tiếng Việt, chuẩn cho tài liệu tài chính.

---

## 5. Luồng dữ liệu đầy đủ trong FinRisk AI

```
[Người dùng] "DSCR của Vinamilk 2023 là bao nhiêu?"
      │
      ▼
[rag_agent_node]
      │
      ├─ embed_dense(query)  → dense_vec [1024]
      ├─ embed_sparse(query) → sparse_vec {indices, values}
      │
      ▼
[Qdrant Hybrid Search — server-side RRF]
      ├─ Prefetch Dense: cosine similarity → Top-20
      ├─ Prefetch Sparse: BM25 dot product → Top-20
      └─ RRF Fusion → Unified Top-20 candidates
      │
      ▼
[Cross-Encoder Reranker]
      ├─ Chấm điểm 20 cặp (query, child_text)
      ├─ Lọc threshold 0.3
      └─ Top-4 RetrievedDoc (mang theo parent_text)
      │
      ▼
[math_agent / risk_agent]
      ├─ state["retrieved_context"] = [parent_text_1, parent_text_2, ...]
      └─ LLM dùng context này để phân tích + tính toán
```

---

## 6. Lệnh Docker cho Qdrant Local

```bash
# Khởi động Qdrant (data được lưu persistent qua Docker volume)
docker run -d \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  --name qdrant \
  --restart unless-stopped \
  qdrant/qdrant

# Kiểm tra health
curl http://localhost:6333/healthz

# Xem collections qua Dashboard
# Mở browser: http://localhost:6333/dashboard

# Dừng/khởi động lại
docker stop qdrant
docker start qdrant

# Xem logs
docker logs qdrant -f
```

---

## 7. Thứ tự làm việc khi thêm tài liệu mới

```bash
# 1. Đảm bảo Qdrant đang chạy
docker ps | grep qdrant

# 2. Index PDF mẫu (demo mode không cần PDF thật)
python scripts/index_sample.py --demo

# 3. Index PDF thật
python scripts/index_sample.py \
  --pdf "data/vinamilk_bctc_2023.pdf" \
  --company "Vinamilk" \
  --year 2023 \
  --doc-type bctc

# 4. Verify qua Dashboard: http://localhost:6333/dashboard
# 5. Chạy pipeline
python main.py
```

---

## 8. Troubleshooting

| Lỗi | Nguyên nhân | Cách sửa |
|-----|------------|---------|
| `Connection refused localhost:6333` | Qdrant chưa chạy | `docker start qdrant` |
| `Collection 'financial_docs' not found` | Chưa index | Chạy `scripts/index_sample.py --demo` |
| `Model download stuck` | Network chậm | BGE-M3 ~600MB, Reranker ~1.1GB — đợi thêm |
| `OSError: [Errno 28] No space left` | Ổ đĩa đầy | Models cache tại `~/.cache/fastembed` — xoá models cũ |
| `0 results returned` | threshold quá cao hoặc chưa có data | Giảm `RERANKER_THRESHOLD` trong `.env` hoặc index data trước |

---

## 9. Cấu hình trong `.env`

```dotenv
# Qdrant
QDRANT_HOST=localhost          # Đổi thành "qdrant" khi dùng Docker Compose
QDRANT_PORT=6333
QDRANT_API_KEY=                # Để trống khi local; điền khi dùng Qdrant Cloud
QDRANT_COLLECTION=financial_docs

# Embedding & Reranker
DENSE_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_THRESHOLD=0.3         # Giảm nếu bị "0 results"
RETRIEVAL_TOP_K=20             # Số candidates trước rerank
RETRIEVAL_TOP_N=4              # Số docs gửi LLM

# Chunking
CHUNK_CHILD_TOKENS=150
CHUNK_PARENT_TOKENS=1000
```

---

## 10. Phase tiếp theo (Phase 3)

| Cải tiến | Mô tả |
|---------|-------|
| **Contextual Retrieval** | Dùng LLM gắn prefix ngữ cảnh trước khi embed child chunk → tăng recall |
| **Ragas Evaluation** | Đo Faithfulness, Context Recall trên Golden Testset 50 câu |
| **Langfuse Traces** | Ghi lại mỗi search query, số docs retrieved, reranker scores → debug dễ hơn |
| **LLMLingua** | Nén parent_text từ 1000 tokens → 300 tokens → giảm 30% chi phí LLM |
