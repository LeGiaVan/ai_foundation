# 🕸️ CẨM NANG CHUYÊN SÂU: MICROSOFT GRAPHRAG (KNOWLEDGE GRAPH + RAG)
## Bí kíp làm chủ Kiến trúc Trích xuất Tri thức & Suy luận Đồ thị Toàn diện

---

## 1. MENTAL MODEL: TẠI SAO PHẢI CẦN GRAPHRAG?

### Điểm mù chí mạng của Vector RAG truyền thống
Vector RAG chỉ tìm kiếm theo độ tương đồng ngữ nghĩa cục bộ (Local Semantic Similarity). Nó chỉ giỏi với các câu hỏi tìm kiếm trực tiếp dạng **"Cây kim trong bọc" (Needle in a Haystack)**:
* *Ví dụ:* `"Hạn mức chuyển tiền qua App của ngân hàng là bao nhiêu?"` $\rightarrow$ Vector RAG tìm đúng 1 chunk chứa số tiền là xong.

Nhưng Vector RAG **hoàn toàn bất lực trước 2 dạng bài toán**:
1. **Câu hỏi tổng thể (Global Sensemaking):**  
   * *"Toàn bộ tài liệu này nói về những chủ đề chính nào?"*
   * *"Đánh giá các rủi ro hệ thống được nhắc tới trong 500 báo cáo kiểm toán?"*  
   $\rightarrow$ Không có 1 chunk nào chứa câu trả lời. Vector Search sẽ bốc ngẫu nhiên vài chunk bề nổi, dẫn đến câu trả lời nông cạn hoặc bịa đặt.
2. **Câu hỏi liên kết mắt xích đa tầng (Multi-hop Reasoning):**  
   * *"Mối quan hệ gián tiếp giữa ông A và công ty B qua các dự án từ năm 2020 đến 2024 là gì?"*  
   $\rightarrow$ Thông tin bị xé nhỏ ở 10 tài liệu khác nhau. Vector search không thể "kết nối các dấu chấm" (Connect the dots).

👉 **GraphRAG (Microsoft Research)** ra đời để giải quyết triệt để 2 điểm mù này bằng cách: **Biến toàn bộ kho văn bản thành một Đồ thị tri thức (Knowledge Graph) phân cấp trước khi truy vấn.**

---

## 2. BỨC TRANH TOÀN CẢNH: VECTOR RAG VS. GRAPHRAG

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│       TRADITIONAL VECTOR RAG         │        MICROSOFT GRAPHRAG            │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • Biểu diễn: Từng chunk văn bản cô   │ • Biểu diễn: Mạng lưới Thực thể      │
│   lập trong không gian vector.       │   (Entities) và Quan hệ (Edges).     │
│ • Cấu trúc: Phẳng (Flat).            │ • Cấu trúc: Đồ thị phân cấp đa tầng  │
│                                      │   (Hierarchical Communities).        │
│ • Câu hỏi phù hợp: Tìm kiếm cục bộ   │ • Câu hỏi phù hợp: Tổng thể vĩ mô    │
│   (Local - Factoid, Needle).         │   (Global), Tổng hợp, Điều tra.      │
│ • Chi phí Index: Cực rẻ (chỉ embed). │ • Chi phí Index: Rất đắt (Dùng LLM   │
│                                      │   để trích xuất thực thể & tóm tắt). │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 3. QUY TRÌNH INDEXING 5 BƯỚC CỦA GRAPHRAG (PIPELINE NẠP DỮ LIỆU)

Đây là giai đoạn nặng nề và phức tạp nhất của GraphRAG:

```
[Văn bản thô]
     │
     ▼ (Bước 1: Chunks / Text Units)
[Cắt Text Units (300 - 600 tokens)]
     │
     ▼ (Bước 2: Entity & Relationship Extraction)
[LLM trích xuất: Thực thể, Quan hệ, Yêu sách (Claims)]
     │
     ▼ (Bước 3: Graph Construction)
[Xây dựng Đồ thị tri thức toàn cục]
     │
     ▼ (Bước 4: Hierarchical Community Detection - Leiden)
[Phân cụm đồ thị thành các cộng đồng đa tầng: Level 0, 1, 2]
     │
     ▼ (Bước 5: Community Summarization)
[LLM tạo Báo cáo tóm tắt cho từng cụm (Community Reports)]
```

### Chi tiết từng bước:

#### Bước 1: Text Chunking (Text Units)
* Cắt tài liệu thành các đơn vị văn bản nhỏ (Text Units) từ 300 - 600 tokens (cho phép overlap nhỏ). Kích thước này tối ưu cho việc LLM đọc và trích xuất không bị sót thông tin.

#### Bước 2: Entity & Relationship Extraction (Trích xuất bằng LLM)
LLM (ví dụ GPT-4o-mini) đọc từng Text Unit và trích xuất:
* **Entities (Thực thể - Các Node):** Tên, Phân loại (`PERSON`, `ORGANIZATION`, `GEO`, `CONCEPT`), Mô tả ngắn.
* **Relationships (Quan hệ - Các Cạnh / Edges):** Thực thể nguồn, Thực thể đích, Mô tả mối quan hệ, Trọng số (Strength).
* **Claims / Covariates (Yêu sách):** Các tuyên bố có mốc thời gian, trạng thái (ví dụ: *"Công ty X bị phạt vào tháng 8/2023"*).

#### Bước 3: Graph Construction & Resolving (Đồng nhất hóa thực thể)
* Gom các thực thể cùng trỏ về 1 đối tượng (ví dụ: `"Microsoft"`, `"MSFT"`, `"Tập đoàn Microsoft"` được gộp thành 1 Node duy nhất).

#### Bước 4: Community Detection (Thuật toán phân cụm Leiden)
* GraphRAG áp dụng **Thuật toán Leiden** để phát hiện các "cụm thực thể có quan hệ mật thiết với nhau":
  * **Level 0 (Vĩ mô):** Các cụm chủ đề bao quát toàn bộ tài liệu (ví dụ: Kinh tế, Quân sự, Y tế).
  * **Level 1 (Trung mô):** Các nhánh con (ví dụ: Ngân hàng số, Bất động sản...).
  * **Level 2 (Vi mô):** Các nhóm thực thể cụ thể (ví dụ: Ban lãnh đạo công ty A, Dự án B).

#### Bước 5: Community Summarization (Sinh báo cáo cộng đồng)
* Với mỗi cụm ở mọi tầng, LLM đọc toàn bộ các Node và Edge bên trong rồi viết thành một **Community Report** hoàn chỉnh gồm:
  * Tiêu đề chủ đề.
  * Tóm tắt nội dung cốt lõi.
  * Các phát hiện quan trọng (Key Findings kèm trích dẫn).
  * Đánh giá mức độ rủi ro / tác động.

---

## 4. HAI CHẾ ĐỘ TRUY VẤN CỐT LÕI (QUERY ENGINES)

Microsoft GraphRAG cung cấp **2 công cụ truy vấn chuyên biệt** tùy theo mục đích câu hỏi của người dùng:

```
                            [CÂU HỎI CỦA USER]
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
   [GLOBAL SEARCH ENGINE]                         [LOCAL SEARCH ENGINE]
  (Dành cho câu hỏi tổng thể vĩ mô)             (Dành cho câu hỏi thực thể cụ thể)
  • "Toàn bộ tài liệu nói về gì?"               • "Mối quan hệ giữa A và B là gì?"
  • Không dùng Text Chunks gốc!                 • Đi từ Entity -> Mở rộng láng giềng
  • Sử dụng Community Reports + Map-Reduce      • Kết hợp Chunks + Graph Edges
```

---

### 4.1. GLOBAL SEARCH (Tìm kiếm toàn cục bằng Map-Reduce)

* **Khi nào dùng:** Khi câu hỏi mang tính chất bao quát, đánh giá xu hướng, tổng hợp toàn bộ kho tài liệu.
* **Cách vận hành:**
  1. **Chọn tầng cộng đồng:** Chọn các Community Reports ở tầng phù hợp (thường là Level 1 hoặc Level 2).
  2. **Giai đoạn MAP (Phân tích song song):**
     * Chia các Community Reports thành nhiều nhóm nhỏ.
     * Cho LLM đọc từng nhóm và sinh ra một **Câu trả lời trung gian (Intermediate Answer)** kèm theo điểm đánh giá mức độ hữu ích (Rating: 0-100).
  3. **Lọc:** Loại bỏ các câu trả lời trung gian có điểm Rating = 0 hoặc quá thấp.
  4. **Giai đoạn REDUCE (Tổng hợp kết quả):**
     * Gom tất cả các câu trả lời trung gian đạt điểm cao, đưa vào LLM chính để tổng hợp thành câu trả lời cuối cùng mạch lạc, có cấu trúc chặt chẽ.

---

### 4.2. LOCAL SEARCH (Tìm kiếm cục bộ mở rộng láng giềng)

* **Khi nào dùng:** Khi câu hỏi nhắm vào một thực thể cụ thể hoặc mối quan hệ giữa các thực thể (Ví dụ: *"Dự án X của bà Y đã gây ra hậu quả gì cho công ty Z?"*).
* **Cách vận hành:**
  1. **Trích xuất Entity từ Query:** Nhận diện các thực thể có trong câu hỏi của User.
  2. **Mở rộng trên Đồ thị (K-hop Traversal):** Từ các thực thể gốc, đi theo các cạnh (Edges) để tìm các thực thể láng giềng có liên quan trực tiếp.
  3. **Gom nhặt Context đa nguồn:**
     * Lấy các **Text Units gốc** gắn liền với các thực thể này.
     * Lấy thông tin mô tả của các **Relationships (Edges)**.
     * Lấy các **Community Reports** cấp thấp chứa các thực thể đó.
  4. **Đưa vào Prompt cho LLM:** Đóng gói toàn bộ thông tin có cấu trúc trên gửi cho LLM để tạo ra câu trả lời chính xác từng chi tiết.

---

### 4.3. DRIFT SEARCH (Dynamic Reasoning and Inference with Flexible Traversal - Nâng cấp mới nhất)
* Là kỹ thuật kết hợp đỉnh cao giữa **Global Search** và **Local Search**:
  * Bắt đầu bằng cách hỏi các Community Reports để định vị chủ đề.
  * Sau đó tự động "nhảy" xuống các nút lá cụ thể (Local Traversal) để kiểm chứng chi tiết.
  * Tiết kiệm chi phí hơn Global Search thuần túy nhưng trả lời câu hỏi phức tạp sâu hơn Local Search.

---

## 5. PHÂN TÍCH CHI PHÍ, TỐC ĐỘ & ĐÁNH ĐỔI (TRADE-OFFS)

GraphRAG là "ông vua" về chất lượng câu trả lời, nhưng bạn phải trả một cái giá tương xứng:

| Tiêu chí | Vector RAG truyền thống | Microsoft GraphRAG |
| :--- | :--- | :--- |
| **Chi phí Indexing** | Rất rẻ (chỉ tốn tiền API Embedding: ~$0.02 / 100 trang). | **Rất đắt** (gọi hàng nghìn lượt LLM để trích xuất: ~$5 - $20 / 100 trang). |
| **Thời gian Indexing** | Vài giây đến vài phút. | **Vài chục phút đến vài giờ** (phụ thuộc vào Rate Limit của LLM). |
| **Độ trễ Query (Latency)** | 50ms - 500ms. | **2 giây - 8 giây** (do quy trình Map-Reduce nhiều chặng). |
| **Chất lượng câu hỏi vĩ mô**| Kém / Ảo giác (Fail 80%). | **Hoàn hảo / Toàn diện (Chính xác > 95%).** |
| **Khả năng liên kết chéo** | Rất yếu (dễ bỏ sót). | **Xuất sắc (theo vết từng mắt xích quan hệ).** |

---

## 6. KINH NGHIỆM THỰC CHIẾN TỐI ƯU HÓA CHI PHÍ CHO GRAPHRAG

Nếu dùng GPT-4o để Index GraphRAG trên tài liệu lớn, bạn có thể "cháy túi" tiền API trong 1 đêm. Đây là các chiến thuật tối ưu production:

### 1. Phân tầng LLM (Model Tiering)
* **Lúc Indexing (Trích xuất & Tóm tắt cụm):** Dùng các model rẻ và nhanh như `gpt-4o-mini`, `claude-3-5-haiku`, hoặc model local như `Qwen-2.5-72B / Llama-3.3-70B` qua vLLM / Ollama.
* **Lúc Query Reduce (Tổng hợp câu trả lời cuối):** Mới dùng model mạnh nhất (`GPT-4o`, `Claude 3.5 Sonnet`).

### 2. Tận dụng Prompt Caching
* Khi chạy trích xuất thực thể, System Prompt của GraphRAG rất dài và lặp đi lặp lại trên hàng nghìn chunks.
* Hãy bật **Prompt Caching** (OpenAI / Anthropic / DeepSeek) để giảm ngay **50% - 75% chi phí input token**.

### 3. Tinh chỉnh Chunk Size & Gleanings
* Tham số `entity_extraction.max_gleanings`: Số lần LLM quay lại đọc lại chunk để vớt thực thể sót. Mặc định là `1`. Nếu tài liệu không quá phức tạp, hãy để `= 0` để giảm một nửa số request LLM!

---

## 7. CÂY QUYẾT ĐỊNH: KHI NÀO NÊN DÙNG GRAPHRAG?

```
Câu hỏi của bài toán là gì?
 │
 ├── Chỉ là tra cứu điều khoản, tìm kiếm dữ kiện cụ thể?
 │    └── ❌ DÙNG HYBRID RAG (Dense + BM25 + Rerank) -> Rẻ, nhanh, đủ tốt!
 │
 ├── Người dùng liên tục hỏi: "Tổng hợp các rủi ro", "Ý chính toàn bộ tài liệu"?
 │    └── ✅ DÙNG GRAPHRAG (Chế độ GLOBAL SEARCH).
 │
 ├── Cần liên kết thực thể phức tạp: Điều tra tội phạm, Báo cáo kiểm toán, 
 │   Hồ sơ bệnh án đa bệnh viện, Chuỗi cung ứng?
 │    └── ✅ DÙNG GRAPHRAG (Chế độ LOCAL SEARCH / DRIFT SEARCH).
 │
 └── Dữ liệu cập nhật liên tục từng giây (Real-time Streaming)?
      └── ❌ TRÁNH GRAPHRAG (Vì chi phí dựng lại đồ thị liên tục là bất khả thi).
```
