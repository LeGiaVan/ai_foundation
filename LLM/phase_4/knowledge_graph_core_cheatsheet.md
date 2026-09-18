# 🕸️ CẨM NANG CỐT TỬ: KNOWLEDGE GRAPH (ĐỒ THỊ TRI THỨC) CHO AI ENGINEER
## Nắm vững Bản chất, Cấu trúc Dữ liệu & Ứng dụng trong RAG / GenAI

---

## 1. MENTAL MODEL: KNOWLEDGE GRAPH LÀ GÌ?

### Văn bản phẳng vs. Bảng quan hệ (SQL) vs. Đồ thị tri thức (Graph)

Hãy tưởng tượng bạn có thông tin:  
*"Bà Lan là vợ ông Tuấn. Ông Tuấn sở hữu 40% cổ phần công ty ABC. Công ty ABC lại là công ty mẹ của ngân hàng XYZ. Ngân hàng XYZ cho công ty M vay 500 tỷ."*

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│  RDBMS (BẢNG QUAN HỆ SQL)            │  KNOWLEDGE GRAPH (ĐỒ THỊ TRI THỨC)   │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • Dữ liệu bị xé nhỏ thành các Bảng.  │ • Dữ liệu mô phỏng MẠNG LƯỚI TƯ DUY  │
│ • Để tìm quan hệ giữa bà Lan và      │   tự nhiên của con người.            │
│   khoản vay của cty M: Phải JOIN     │ • Truy vết quan hệ theo bước nhảy    │
│   5 - 6 bảng liên tiếp.              │   (Hops): Đi dọc theo các mũi tên.   │
│ • Chi phí tính toán bùng nổ (O(N^k)),│ • Tốc độ tìm kiếm quan hệ là O(1)    │
│   dễ treo cơ sở dữ liệu.             │   cho mỗi bước nhảy láng giềng.      │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

👉 **Định nghĩa cốt lõi:**  
**Knowledge Graph** là một cơ sở dữ liệu lưu trữ tri thức dưới dạng **Mạng lưới các Thực thể (Nodes)** và **Mối quan hệ có ngữ nghĩa (Edges)** nối giữa chúng.

---

## 2. BỘ BA NGUYÊN TỬ CỦA KNOWLEDGE GRAPH (TRIPLES)

Mọi Knowledge Graph trên thế giới — dù phức tạp đến đâu — đều được xây dựng từ đơn vị cơ bản nhất gọi là **Bộ ba (SPO Triple)**:

$$\text{Subject (Chủ thể)} \xrightarrow{\text{Predicate (Quan hệ)}} \text{Object (Đối tượng)}$$

```
(Bà Lan) ──────[LÀ_VỢ_CỦA]──────> (Ông Tuấn)
   │                                  │
   │                                  ▼ [SỞ_HỮU_40%]
   │                               (Công ty ABC)
   │                                  │
   │                                  ▼ [LÀ_MẸ_CỦA]
   └──────────[CÓ_QUAN_HỆ_GIÁN_TIẾP]──► (Ngân hàng XYZ)
```

### 1. Nodes (Đỉnh / Thực thể - Entities):
* Đại diện cho danh từ trong thế giới thực: Người, Địa điểm, Tổ chức, Sản phẩm, Sự kiện, Khái niệm.
* *Thuộc tính (Properties):* Mỗi node có thể chứa dữ liệu riêng: `{tên: "Tuấn", tuổi: 45, cccd: "0123..."}`.

### 2. Edges (Cạnh / Mối quan hệ - Relationships):
* Luôn là **đường có hướng (Directed)** và có **động từ hành động** rõ ràng: `[SỞ_HỮU]`, `[LÀ_GIÁM_ĐỐC]`, `[ĐIỀU_TRỊ_BẰNG]`, `[CUNG_CẤP_CHO]`.
* *Thuộc tính của cạnh:* Cạnh cũng có thể chứa thông tin: `[CUNG_CẤP_CHO {từ_năm: 2021, giá_trị: "10 tỷ"}]`.

### 3. Ontology & Schema (Bản thiết kế cấu trúc tri thức):
* Là "bản vẽ quy hoạch" quy định những loại thực thể nào và quan hệ nào được phép tồn tại:
  * Ví dụ: Quy định thực thể loại `BÁC_SĨ` chỉ có thể có quan hệ `[KÊ_ĐƠN]` với `THUỐC`, chứ không thể có quan hệ `[KÊ_ĐƠN]` với `BỆNH_VIỆN`.

---

## 3. CÁCH LƯU TRỮ: MÔ HÌNH LPG (LABELED PROPERTY GRAPH) & NEO4J

Trong giới AI kỹ thuật hiện đại, chuẩn phổ biến nhất là **LPG (Labeled Property Graph)**, đại diện tiêu biểu là **Neo4j**:

* **Ngôn ngữ truy vấn Cypher:** Giống như SQL dành cho bảng, Cypher là ngôn ngữ dành cho đồ thị, sử dụng ký tự vẽ hình trực quan:
  ```cypher
  // Tìm tất cả các công ty mà ông Tuấn sở hữu cổ phần:
  MATCH (p:Person {name: "Ông Tuấn"})-[r:SỞ_HỮU]->(c:Company)
  RETURN c.name, r.percentage
  ```

---

## 4. QUY TRÌNH 3 BƯỚC XÂY DỰNG KNOWLEDGE GRAPH TỰ ĐỘNG BẰNG LLM

Khi biến văn bản tài liệu thô thành Knowledge Graph, pipeline chuẩn gồm 3 bước:

```
[Văn bản thô]
     │
     ▼
[Bước 1: Named Entity Recognition (NER)]
  -> Trích xuất các thực thể: [Bà Lan: PERSON], [Vinamilk: ORG]
     │
     ▼
[Bước 2: Relation Extraction (RE)]
  -> Trích xuất quan hệ: (Bà Lan) -[LÀ_CỔ_ĐÔNG]-> (Vinamilk)
     │
     ▼
[Bước 3: Entity Resolution / Disambiguation (Khử trùng lặp thực thể)]
  -> Gom các tên gọi khác nhau: "Vinamilk", "VNM", "CTCP Sữa VN" 
     thành 1 NODE DUY NHẤT trên đồ thị!
```

---

## 5. 3 CÁCH KẾT HỢP KNOWLEDGE GRAPH VỚI LLM / RAG HIỆN ĐẠI

Đây là phần quan trọng nhất cho công việc của một AI Engineer:

```
                               ┌────────────────────────────────────────────────────────┐
                               │     3 HƯỚNG ỨNG DỤNG KNOWLEDGE GRAPH VỚI GENAI         │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
              ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
              ▼                                            ▼                                            ▼
      [1. TEXT-TO-CYPHER]                         [2. GRAPH-AUGMENTED RAG]                     [3. FACT-CHECKING GROUNDING]
  (Hỏi đáp dữ liệu chính xác)                   (Bổ sung ngữ cảnh quan hệ)                   (Kiểm chứng chống ảo giác)
• LLM dịch câu hỏi thành Cypher               • Lấy thực thể trong câu hỏi                   • Dùng KG làm "sự thật chuẩn"
• Chạy trên Neo4j ra kết quả 100%             • Trích xuất mạng lưới láng giềng              • So khớp câu trả lời của LLM
  chính xác, không bị ảo giác số liệu.          gửi kèm vào Prompt cho LLM.                    xem có mâu thuẫn đồ thị không.
```

### Ứng dụng 1: Text-to-Cypher (Tương tự Text-to-SQL)
* **Vấn đề của Vector RAG:** Hỏi *"Có bao nhiêu công ty liên kết với ông X có vốn trên 100 tỷ?"* $\rightarrow$ Vector RAG chắc chắn đếm sai hoặc bịa.
* **Giải pháp KG:** LLM nhận câu hỏi $\rightarrow$ Tự sinh câu lệnh Cypher:
  ```cypher
  MATCH (p:Person {name: "Ông X"})-[:SỞ_HỮU]->(c:Company)
  WHERE c.von > 100
  RETURN count(c)
  ```
  $\rightarrow$ Cơ sở dữ liệu đồ thị trả về con số chính xác $100\%$.

### Ứng dụng 2: Graph-Augmented RAG (Gom cụm ngữ cảnh láng giềng)
* Khi người dùng hỏi về một sự kiện, hệ thống trích xuất thực thể liên quan $\rightarrow$ Lấy toàn bộ đồ thị con (Subgraph bán kính 1 - 2 bước nhảy) xung quanh thực thể đó $\rightarrow$ Đưa cây quan hệ này vào prompt để LLM trả lời sâu sắc và mạch lạc.

### Ứng dụng 3: Chống ảo giác tuyệt đối (Fact-Checking)
* Knowledge Graph được coi là **"Chân lý cứng" (Hard Ground Truth)**. Nếu LLM sinh ra câu: *"Công ty A đã mua lại công ty B vào năm 2022"*, hệ thống truy vấn đồ thị kiểm tra:
  * Nếu không có cạnh `[MUA_LẠI]` giữa A và B $\rightarrow$ Cảnh báo **Ảo giác (Hallucination)** ngay lập tức!

---

## 6. MA TRẬN SO SÁNH: KHI NÀO DÙNG GÌ?

| Tiêu chí | Vector Database (Qdrant, Pinecone) | Relational Database (PostgreSQL) | Knowledge Graph (Neo4j) |
| :--- | :--- | :--- | :--- |
| **Bản chất dữ liệu** | Không gian vector phi cấu trúc (Embedding). | Dữ liệu dạng bảng có cấu trúc rõ ràng. | Dữ liệu mạng lưới, nhiều liên kết phức tạp. |
| **Câu hỏi phù hợp nhất** | *"Tìm đoạn văn nói về chủ đề X"* (Tìm kiếm ngữ nghĩa). | *"Tính tổng doanh thu tháng 8"* (Thống kê, số liệu). | *"A quen ai, ai liên quan đến B, đường đi ngắn nhất giữa A và B?"* |
| **Xử lý liên kết đa tầng (Multi-hop)**| Rất kém (Bị đứt gãy thông tin). | Kém (JOIN nhiều bảng gây chậm). | **Vô địch (Đặc chế cho bài toán đường đi).** |
| **Chi phí xây dựng** | Rất rẻ và nhanh (chỉ cần chunk và embed). | Trung bình. | **Đắt đỏ (Cần thiết kế Ontology và trích xuất thực thể).** |

---

## 7. TÓM TẮT DÀNH CHO BẠN (TAKEAWAYS BỎ TÚI)

1. **Hiểu concept là đủ:** Bạn hoàn toàn đúng, trừ khi công ty bạn làm về **chống rửa tiền, phân tích gian lận ngân hàng, chuỗi cung ứng toàn cầu, hoặc chẩn đoán y khoa đa bệnh viện**, bạn không cần phải tự tay thiết kế các hệ thống Graph từ con số 0.
2. **Nhớ từ khóa then chốt:**
   * Cấu trúc cơ bản: **SPO Triple (Node - Edge - Node)**.
   * Cơ sở dữ liệu chuẩn: **Neo4j** (ngôn ngữ **Cypher**).
   * Điểm mạnh nhất: **Multi-hop reasoning (Truy vết liên kết bắc cầu nhiều bước)**.
   * Ứng dụng GenAI mạnh nhất: **Text-to-Cypher** (Truy vấn dữ liệu quan hệ chính xác 100% không lo ảo giác).
