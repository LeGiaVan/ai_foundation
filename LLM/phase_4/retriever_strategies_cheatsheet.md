# 🧭 CẨM NANG CHIẾN THUẬT RETRIEVER TOÀN DIỆN (MODERN RETRIEVER CHEATSHEET)
## Bản đồ công nghệ từ Cơ bản đến Tối tân (State-of-the-Art)

---

## 1. MENTAL MODEL: VÒNG ĐỜI 3 GIAI ĐOẠN CỦA RETRIEVER PIPELINE

Trong RAG nâng cao (Advanced RAG) và Agentic RAG, **Retriever không phải là 1 bước đơn lẻ**, mà là một dây chuyền gồm 3 giai đoạn độc lập:

```
[User Query]
     │
     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1: PRE-RETRIEVAL (Tiền xử lý & Biến đổi câu hỏi)            │
│ Mục tiêu: Biến câu hỏi thô/dở của User thành các "chìa khóa vàng".    │
│ • Query Rewriting / Expansion   • HyDE (Văn bản giả định)              │
│ • Sub-Question Decomposition     • Step-Back Prompting                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Bộ truy vấn chuẩn)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2: IN-RETRIEVAL (Chiến lược Tìm kiếm & Cấu trúc Index)       │
│ Mục tiêu: Quét kho dữ liệu, tối ưu hóa sự cân bằng giữa Precision/Recall│
│ • Dense Retrieval (HNSW)         • Sparse Search (BM25 / SPLADE)       │
│ • Hybrid Search + RRF            • Small-to-Big / Parent-Document      │
│ • RAPTOR (Cây phân cấp)          • Anthropic Contextual Retrieval      │
│ • ColBERT (Late Interaction)     • GraphRAG (Knowledge Graph)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Tập ứng viên thô: 15 - 30 chunks)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3: POST-RETRIEVAL (Hậu xử lý & Tinh lọc trước khi nạp LLM)   │
│ Mục tiêu: Đóng gói context sạch nhất, đúng thứ tự, chống ảo giác       │
│ • Cross-Encoder Reranker         • MMR (Maximal Marginal Relevance)    │
│ • Score Thresholding             • Context Compression (LLMLingua)     │
│ • Lost-in-the-middle Reordering                                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Top 3 - 5 chunks tinh hoa)
                                    ▼
                             [LLM Generator]
```

---

## 1.1. SƠ ĐỒ HỆ THỐNG HÓA TẤT CẢ CÁC CHIẾN THUẬT (MERMAID ARCHITECTURE MAP)

<p align="center">
  <img src="rag_optimize.png" alt="Rag Optimizing Stragegy" width="800" />
</p>

---

## 2. CHI TIẾT CÁC CHIẾN THUẬT RETRIEVER PHỔ BIẾN & HIỆN ĐẠI

---

### GIAI ĐOẠN 1: PRE-RETRIEVAL (TỐI ƯU ĐẦU VÀO)

#### 1. Query Rewriting (Viết lại truy vấn)
* **Bản chất:** Dùng LLM sửa lỗi chính tả, bỏ tiếng lóng, bổ sung từ khóa ngữ cảnh vào câu hỏi thô của người dùng.
* **Ví dụ:** `"Lương tháng 13 cty mình tính sao ta?"` $\rightarrow$ `"Quy định và cách tính tiền thưởng lương tháng 13 theo chính sách nhân sự công ty"`.
* **Khi nào dùng:** Mọi hệ thống RAG tiếp xúc trực tiếp với người dùng đại chúng (Customer Support Chatbot).

#### 2. Multi-Query Expansion (Mở rộng đa truy vấn)
* **Bản chất:** LLM sinh ra 3 - 5 biến thể của cùng một câu hỏi từ nhiều góc nhìn khác nhau. Chạy song song cả 5 biến thể xuống Vector DB rồi gộp kết quả lại (De-duplicate).
* **Ưu điểm:** Khắc phục triệt để việc người dùng dùng từ không khớp với tài liệu (Vocabulary Mismatch). Kéo Recall tăng vọt.
* **Nhược điểm:** Tăng số lượng truy vấn xuống Vector DB gấp 3-5 lần.

#### 3. Sub-Question Decomposition (Bẻ nhỏ câu hỏi phức tạp)
* **Bản chất:** Tách 1 câu hỏi đa ý (Multi-hop Query) thành 2-3 câu hỏi đơn lẻ độc lập.
* **Ví dụ:** `"So sánh doanh thu quý 2 của Vinamilk và TH True Milk"` $\rightarrow$ Tách thành:
  * Câu 1: `"Doanh thu quý 2 của Vinamilk là bao nhiêu?"`
  * Câu 2: `"Doanh thu quý 2 của TH True Milk là bao nhiêu?"`
* **Khi nào dùng:** Các bài toán phân tích tài chính, đối chiếu hợp đồng, so sánh chính sách.

#### 4. HyDE (Hypothetical Document Embeddings - Tài liệu giả định)
* **Bản chất:** Cho LLM "bịa" ra một đoạn văn bản giả định trả lời câu hỏi trước. Dùng vector của đoạn văn giả định này để đi tìm kiếm tài liệu thật trong DB.
* **Cơ chế:** Đổi bài toán từ so khớp `(Câu hỏi ngắn <-> Đoạn văn dài)` thành `(Đoạn văn <-> Đoạn văn)`.
* **Khi nào dùng:** Khi câu hỏi quá ngắn, trừu tượng hoặc tài liệu chuyên ngành rất dài.

#### 5. Step-Back Prompting (Lùi một bước để thấy toàn cảnh)
* **Bản chất:** LLM đặt một câu hỏi tổng quát hơn, ở mức độ nguyên lý nền tảng (High-level concept) trước khi đi vào tiểu tiết.
* **Ví dụ:** Hỏi `"Tại sao pin iPhone 15 bị tụt nhanh khi chơi game X?"` $\rightarrow$ Câu hỏi Step-back: `"Các yếu tố ảnh hưởng đến mức tiêu hao năng lượng của chip A16 khi xử lý đồ họa cao"`.

---

### GIAI ĐOẠN 2: IN-RETRIEVAL (CHIẾN LƯỢC TÌM KIẾM & INDEXING)

#### 1. Standard Dense Retrieval (Vector Search cơ bản)
* **Cơ chế:** Dùng Embedding Model (như `text-embedding-3-small`, `bge-m3`) biến Text thành Vector, lưu trên chỉ mục đồ thị HNSW.
* **Đặc điểm:** Tốc độ siêu nhanh (< 10ms), hiểu ngữ nghĩa trừu tượng tốt, nhưng **mù từ khóa cứng** (mã số, tên riêng, ngày tháng).

#### 2. Sparse Retrieval (BM25 / SPLADE)
* **Cơ chế:** Tìm kiếm dựa trên tần suất xuất hiện của từ (Term Frequency - Inverted Index).
  * **BM25:** Khớp chính xác 100% từ khóa cứng (như Ctrl+F nâng cao).
  * **SPLADE (Sparse + Neural):** Tạo ra vector thưa có hiểu biết từ đồng nghĩa từ mô hình ngôn ngữ BERT.
* **Đặc điểm:** Khắc tinh của các lỗi chính xác về số liệu, mã linh kiện, số hiệu văn bản pháp luật.

#### 3. Hybrid Search + RRF (Tiêu chuẩn vàng Production)
* **Cơ chế:** Chạy song song cả Dense (Vector) và Sparse (BM25), sau đó dùng thuật toán **RRF (Reciprocal Rank Fusion)** để xếp hạng lại:
  ```text
  RRF_Score(Doc) = Tổng [ 1 / (60 + Thứ_hạng) ]
  ```
* **Đặc điểm:** Bắt trọn vẹn cả ý niệm trừu tượng lẫn từ khóa cứng. Là kiến trúc **bắt buộc** trong các hệ thống RAG doanh nghiệp hiện đại.

#### 4. Parent-Document Retrieval (Small-to-Big)
* **Cơ chế:** Lưu 2 cấp độ:
  * Cấp con (Child Chunk - 100 tokens): Đem embed vào Vector DB để tìm kiếm cho nhạy.
  * Cấp cha (Parent Chunk - 1000 tokens): Lưu trong DocStore (Redis/SQLite).
* **Lúc tìm:** Search trúng Child Chunk nhưng lại **bốc trọn vẹn Parent Chunk gửi cho LLM**.
* **Đặc điểm:** Giải quyết triệt để vấn đề mất ngữ cảnh do cắt vụn chunk.

#### 5. Sentence-Window Retrieval
* **Cơ chế:** Chia nhỏ tài liệu thành từng **câu đơn lẻ** để vector search chính xác 100%. Khi tìm thấy câu đúng, hệ thống tự động bung cửa sổ lấy thêm $K$ câu phía trước và $K$ câu phía sau nó để gửi cho LLM.
* **Đặc điểm:** Cực kỳ hiệu quả cho tài liệu có mật độ thông tin dày đặc (Sách y khoa, văn bản luật).

#### 6. RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
* **Cơ chế:** 
  * Cắt tài liệu thành các chunk nhỏ ở tầng đáy.
  * Dùng thuật toán phân cụm (Clustering) gom các chunk cùng chủ đề lại, cho LLM **tóm tắt (Summarize)** thành một node cha.
  * Lặp lại đệ quy để tạo ra một **Cây phân cấp (Tree Hierarchy)** gồm nhiều tầng từ chi tiết đến vĩ mô.
* **Đặc điểm:** Cho phép trả lời cả câu hỏi tiểu tiết (lá cây) lẫn câu hỏi bao quát toàn bộ tài liệu như *"Ý chính của cả cuốn sách này là gì?"* (gốc cây).

#### 7. Anthropic Contextual Retrieval (Đột phá cuối năm 2024)
* **Vấn đề nó giải:** Một chunk độc lập: `"Doanh thu quý này tăng 5%"` $\rightarrow$ Không ai biết là quý nào, của công ty nào.
* **Cơ chế:** Lúc chuẩn bị nạp dữ liệu, Anthropic dùng LLM (Claude Haiku) sinh ra một đoạn tiền tố giải thích ngữ cảnh (50-100 tokens) gắn vào đầu mỗi chunk:
  * *Ngữ cảnh bổ sung:* `"Đoạn này trích từ Báo cáo tài chính Q2/2023 của Vinamilk, bàn về mảng sữa đặc..."`
* **Đặc điểm:** Anthropic công bố kỹ thuật này kết hợp với BM25 giúp **giảm 49% tỷ lệ truy xuất thất bại**.

#### 8. ColBERT / Late Interaction (Multi-Vector per Document)
* **Cơ chế:** Thay vì nén cả đoạn văn 500 chữ thành 1 vector duy nhất (làm mất mát thông tin), ColBERT tạo ra **1 vector cho mỗi từ (Token)**.
* **Lúc tìm kiếm:** So khớp ma trận tất cả các token của Query với tất cả các token của Document (Toán tử MaxSim).
* **Đặc điểm:** Độ chính xác tương đương Cross-Encoder nhưng tốc độ nhanh gần bằng Vector Search thông thường.

#### 9. GraphRAG (Knowledge Graph + Vector - Microsoft)
* **Cơ chế:** Dùng LLM trích xuất toàn bộ Thực thể (Entities: người, địa điểm, sự kiện) và Quan hệ (Relationships: làm việc tại, sở hữu bởi) thành một **Đồ thị tri thức (Knowledge Graph)**.
* **Đặc điểm:** Vô địch trong các bài toán điều tra, liên kết mắt xích thông tin nằm rải rác ở hàng trăm tài liệu khác nhau. Nhược điểm: Chi phí index cực kỳ đắt đỏ.

---

### GIAI ĐOẠN 3: POST-RETRIEVAL (TINH CHẾ CONTEXT)

#### 1. Cross-Encoder Reranker (BẮT BUỘC TRONG PRODUCTION)
* **Cơ chế:** Sau khi Vector DB trả về 20 ứng viên, dùng một mô hình chuyên dụng (như `cohere-rerank-v3` hoặc `bge-reranker-large`) đọc đồng thời cả cặp `(Query, Document)` để chấm điểm lại từ 0 đến 1.
* **Đặc điểm:** Đẩy chunk đúng 100% lên vị trí Top-1 và Top-2, triệt tiêu hiện tượng "Bội thực rác" (Bệnh 2).

#### 2. MMR (Maximal Marginal Relevance - Chống trùng lặp)
* **Cơ chế:** Phạt các chunk có nội dung quá giống với chunk đã được chọn trước đó:
  ```text
  MMR = λ × Sim(Query, d) - (1 - λ) × Max_Sim(d, Đã_chọn)
  ```
* **Đặc điểm:** Tăng tối đa độ phủ của các luận điểm khác nhau (Tăng Context Recall).

#### 3. Score Thresholding (Bộ lọc ngưỡng điểm)
* **Cơ chế:** Đặt một lằn ranh đỏ (ví dụ: chỉ giữ chunk có Rerank Score $\ge 0.65$). Nếu chỉ có 2 chunk đạt chuẩn, chỉ gửi 2 chunk cho LLM, vứt bỏ toàn bộ các chunk còn lại.
* **Đặc điểm:** Tránh nhét rác vào Context Window của LLM.

#### 4. Context Compression (Nén ngữ cảnh - LLMLingua)
* **Cơ chế:** Dùng mô hình ngôn ngữ nhỏ loại bỏ các từ dư thừa, từ nối, câu râu ria trong các chunk mà không làm mất ý nghĩa cốt lõi.
* **Đặc điểm:** Giảm 30% - 50% số lượng token, tiết kiệm tiền API và giảm độ trễ sinh lời giải.

#### 5. Lost-in-the-Middle Reordering (Tái sắp xếp vị trí)
* **Cơ chế:** LLM chú ý tốt nhất ở đầu và cuối context window, bị "mù" ở đoạn giữa.
* **Thuật toán xếp lại:** Đưa chunk điểm cao nhất lên vị trí #1 (đầu tiên), chunk điểm nhì xuống vị trí cuối cùng, các chunk phụ xếp ở giữa.

---

### GIAI ĐOẠN 4: AGENTIC & ADAPTIVE RETRIEVAL (XU HƯỚNG HIỆN ĐẠI NHẤT)

```
                       [User Query]
                            │
                            ▼
               [Router / Classifier Agent]
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
    [NO-RAG]           [SIMPLE-RAG]       [COMPLEX-RAG]
  (Chào hỏi, logic)   (Dense/Hybrid)     (Multi-step, CRAG)
```

#### 1. Corrective RAG (CRAG - Tự sửa sai)
* **Cơ chế:** Sau khi lấy tài liệu từ Vector DB, một bộ đánh giá nhỏ (Evaluator) kiểm tra độ tin cậy:
  * **Chắc chắn đúng:** Gửi thẳng cho LLM.
  * **Không chắc chắn / Mơ hồ:** Tự động kích hoạt **Web Search API (Tavily/Google)** để tìm thêm dữ liệu bổ sung từ bên ngoài.
  * **Hoàn toàn sai:** Bỏ toàn bộ dữ liệu nội bộ, chuyển 100% sang tìm kiếm Internet.

#### 2. Self-RAG (RAG Tự phản biện bằng Reflection Tokens)
* **Cơ chế:** LLM được huấn luyện để tự đặt câu hỏi trong lúc sinh câu trả lời:
  * *"Câu hỏi này có cần tìm kiếm tài liệu không?"* `[Retrieve / No-Retrieve]`
  * *"Tài liệu lấy về có liên quan không?"* `[Is-Relevant / Not-Relevant]`
  * *"Câu trả lời mình vừa viết có đúng với tài liệu không?"* `[Supported / Partially-Supported]`

---

## 3. MA TRẬN TRA CỨU NHANH & LỰA CHỌN KIẾN TRÚC

| Chiến thuật | Độ trễ (Latency) | Chi phí (Cost) | Tác động Recall | Tác động Precision | Trường hợp áp dụng tối ưu |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Standard Dense** | ⚡ Siêu nhanh (<15ms) | 🟢 Rất rẻ | 🟡 Trung bình | 🟡 Trung bình | POC nhanh, dữ liệu nhỏ, câu hỏi đơn giản. |
| **Hybrid + RRF** | ⚡ Nhanh (<40ms) | 🟢 Rẻ | 🟢 Rất tốt | 🟢 Tốt | **Khuyến nghị mặc định cho mọi hệ thống.** |
| **Parent-Document**| ⚡ Nhanh (<50ms) | 🟢 Rẻ | 🟢 Cực tốt | 🟢 Rất tốt | Hợp đồng dài, tài liệu kỹ thuật có bối cảnh phức tạp. |
| **Hybrid + Reranker**| 🟡 Trung bình (150-300ms)| 🟡 Vừa phải | 🟢 Cực tốt | 🟢 Đỉnh cao | Ứng dụng ngân hàng, y tế, tài chính cần độ chuẩn 99%. |
| **HyDE** | 🔴 Chậm (+1-2s LLM) | 🟡 Tốn thêm LLM | 🟢 Rất tốt | 🟡 Trung bình | Câu hỏi cực ngắn, từ vựng bị lệch pha hoàn toàn. |
| **RAPTOR** | 🟡 Trung bình lúc query | 🔴 Rất tốn lúc Index | 🟢 Toàn diện | 🟢 Tốt | Câu hỏi tổng quát, tóm tắt cả cuốn sách/bộ hồ sơ. |
| **Anthropic Contextual**| ⚡ Rất nhanh lúc query | 🟡 Tốn LLM lúc Index | 🟢 Tăng 35-50% | 🟢 Rất tốt | Nâng cấp toàn diện cho Hybrid Search hiện đại. |
| **ColBERT (Late Inter.)**| ⚡ Nhanh (50-80ms) | 🔴 Tốn RAM/Disk | 🟢 Đỉnh cao | 🟢 Đỉnh cao | Tìm kiếm ngữ nghĩa đòi hỏi độ chi tiết từng từ khóa. |
| **GraphRAG** | 🔴 Rất chậm (>3-5s) | 🔴 Cực đắt (LLM extract)| 🟢 Vô địch | 🟢 Rất tốt | Điều tra tội phạm, phân tích chuỗi cung ứng, hồ sơ y bạ. |
| **CRAG (Corrective)**| 🔴 Biến thiên theo Web | 🟡 Tốn Web API | 🟢 Toàn diện | 🟢 Rất tốt | Hệ thống chăm sóc khách hàng cần bổ sung tin tức thời sự. |

---

## 4. BỘ KHUNG KIẾN TRÚC VÀNG CHO PRODUCTION (BATTLE-TESTED STACK)

Nếu bạn phải bắt tay xây dựng một hệ thống RAG cấp doanh nghiệp ngay hôm nay, đây là công thức chuẩn công nghiệp có hiệu năng/chi phí tối ưu nhất:

```
[User Query]
     │
     ▼
[Step 1: Query Normalizer / Rewriting (LLM Nhỏ: Haiku / GPT-4o-mini)]
     │
     ▼
[Step 2: Hybrid Search (Qdrant: Dense BGE-M3 + Sparse BM25)]
     │   (Lấy k = 20 ứng viên có gắn sẵn Contextual Header)
     ▼
[Step 3: Cross-Encoder Reranker (Cohere Rerank v3 / BGE-Reranker-Large)]
     │   (Chấm điểm lại & chỉ giữ lại Top-4 có Score > 0.65)
     ▼
[Step 4: Lost-in-the-Middle Reordering]
     │   (Đẩy Top-1 lên đầu prompt, Top-2 xuống cuối prompt)
     ▼
[Step 5: Generator LLM (Temperature = 0.0 + Strict Grounding Prompt)]
```
