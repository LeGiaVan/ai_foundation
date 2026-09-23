# 🏦 HƯỚNG DẪN NGHIỆP VỤ (BUSINESS DOMAIN GUIDE): Thẩm Định Rủi Ro Tín Dụng

Tài liệu này giải thích **mặt nghiệp vụ ngân hàng** của dự án FinRisk AI. Để xây dựng một AI Agent tốt, bạn cần hiểu "AI đang thay thế con người làm cái gì?" và "Tại sao nghiệp vụ này lại khó?".

---

## 1. Bài toán: Thẩm định Tín dụng Doanh nghiệp là gì?

Khi một công ty (ví dụ: Vinamilk, Hòa Phát) đến ngân hàng xin vay một số tiền lớn (hàng trăm tỷ đồng) để mở rộng sản xuất, ngân hàng không thể cho vay dựa trên "lời hứa". Ngân hàng cần đánh giá xem:
1. Doanh nghiệp làm ăn có lãi không?
2. Dòng tiền có đủ để trả nợ gốc và lãi hàng tháng không?
3. Khả năng phá sản trong 1-2 năm tới là bao nhiêu?

Quá trình trả lời 3 câu hỏi này gọi là **Thẩm định Rủi ro Tín dụng (Credit Risk Assessment)**.

### Nỗi đau (Pain points) của quy trình truyền thống:
* **Khối lượng dữ liệu khổng lồ:** Chuyên viên tín dụng (Credit Analyst) phải đọc Báo cáo Tài chính (BCTC) kiểm toán dài 100-200 trang file PDF (thường là scan, rất khó copy/paste số liệu).
* **Tính toán thủ công:** Phải nhập tay các số liệu từ bảng cân đối kế toán, báo cáo kết quả kinh doanh vào Excel để tính các chỉ số tài chính. Dễ sai sót (human error).
* **Bỏ sót "dấu hiệu ngầm" (Red Flags):** Chuyên viên có thể chỉ chú ý đến Lợi nhuận mà bỏ qua việc Dòng tiền thuần từ hoạt động kinh doanh đang âm (công ty bán được hàng nhưng không thu được tiền).
* **Thời gian:** Mất từ **2-3 ngày** cho một bộ hồ sơ.

---

## 2. Cấu trúc Bộ Hồ Sơ Tín Dụng Hoàn Chỉnh (Credit Application Package)

Một sai lầm phổ biến khi làm AI tài chính là nghĩ: *"Chỉ cần ném Báo cáo tài chính (BCTC) vào cho LLM đọc là xong"*. 

Trên thực tế tại các ngân hàng thương mại, BCTC quá khứ chỉ là **1 trong 4 trụ cột** của bộ hồ sơ tín dụng:

```
                  ┌──────────────────────────────────────────────────────────┐
                  │          BỘ HỒ SƠ THẨM ĐỊNH TÍN DỤNG DOANH NGHIỆP        │
                  └─────────────────────────────┬────────────────────────────┘
                                                │
         ┌──────────────────────┬───────────────┴──────────────┬──────────────────────┐
         ▼                      ▼                              ▼                      ▼
┌──────────────────┐  ┌──────────────────┐           ┌──────────────────┐  ┌──────────────────┐
│   1. BCTC PDF    │  │ 2. Đơn Đề Nghị   │           │ 3. Hồ Sơ TSBĐ    │  │ 4. Tin Tức Ngoài │
│   (Quá khứ 3 năm)│  │    Vay Vốn       │           │   (Collateral)   │  │    (External)    │
├──────────────────┤  ├──────────────────┤           ├──────────────────┤  ├──────────────────┤
│- Bảng CĐKT       │  │- Số tiền vay     │           │- Bất động sản    │  │- Tra cứu CIC     │
│- Báo cáo KQKD    │  │- Thời hạn (tháng)│           │- Máy móc / Xe    │  │- Tin tức kiện tụng│
│- Báo cáo LCTT    │  │- Lãi suất (%/năm)│           │- Hàng tồn kho    │  │- Nợ đọng thuế    │
│- Thuyết minh BCTC│  │- Mục đích vay    │           │- Định giá & LTV  │  │- Web Search Tool │
└──────────────────┘  └──────────────────┘           └──────────────────┘  └──────────────────┘
```

1. **Báo cáo Tài chính kiểm toán (Financial Statements):** Phản ánh sức khỏe tài chính trong quá khứ (3 năm gần nhất).
2. **Đơn đề nghị & Phương án vay vốn (Loan Application & Business Plan):** Doanh nghiệp cần vay bao nhiêu tiền? Vay trong bao lâu? Lãi suất bao nhiêu? Trả nợ như thế nào?
   > ⚠️ **Quy tắc nghiệp vụ then chốt:** Không thể tính khả năng trả nợ nếu không có khoản vay mới! Doanh nghiệp BCTC cũ rất đẹp, nhưng nếu gánh thêm khoản vay 100 tỷ thì dòng tiền có gánh nổi gốc + lãi không? Đó gọi là **DSCR dự phóng (Pro-forma DSCR)**.
3. **Hồ sơ Tài sản Bảo đảm (Collateral & Valuation):** Giá trị định giá tài sản thế chấp. Ngân hàng tính tỷ lệ **LTV (Loan-to-Value = Số tiền vay / Giá trị TSBĐ)**. LTV thường phải $\le 70\%$ đối với Bất động sản và $\le 50\%$ đối với Máy móc/Cổ phiếu.
4. **Thông tin phi tài chính ngoài doanh nghiệp (External Intelligence):** Báo cáo tín dụng CIC (nợ xấu nhóm mấy tại ngân hàng khác?), và **Search Web** tìm kiếm tin tức tiêu cực (Negative News: ban lãnh đạo bị điều tra, nợ thuế, đình chỉ hoạt động).

---

## 3. Khung chuẩn Thẩm định Tín dụng Quốc tế (Mô hình 5C Nâng Cấp)

FinRisk AI được thiết kế tự động hóa toàn diện mô hình 5C với sự kết hợp của BCTC, Khoản vay đề nghị và Web Intelligence:

1. **Capacity (Năng lực trả nợ) — Trọng tâm số 1:**
   * Không chỉ tính **DSCR lịch sử**, mà phải tính **Pro-forma DSCR (DSCR dự phóng)**:
     $$\text{Pro-forma DSCR} = \frac{\text{Dòng tiền thuần HĐKD (CFO)}}{\text{Gốc + Lãi nợ CŨ} + \text{Gốc + Lãi khoản vay MỚI}}$$
   * Nếu Pro-forma DSCR $< 1.0 \to$ Khoản vay mới sẽ đẩy doanh nghiệp vào tình trạng mất khả năng thanh toán.
2. **Capital (Vốn chủ sở hữu & Cơ cấu vốn):**
   * Tỷ lệ đòn bẩy tài chính **D/E (Debt-to-Equity)** sau khi giải ngân khoản vay mới. Doanh nghiệp có bỏ đủ vốn đối ứng vào phương án kinh doanh hay dựa 100% vào vốn vay ngân hàng?
3. **Collateral (Tài sản đảm bảo):**
   * Tính toán tỷ lệ **LTV (Loan-to-Value)** dựa trên giá trị TSBĐ định giá so với dư nợ đề xuất.
4. **Conditions (Điều kiện kinh tế / ngành nghề):**
   * `RAG Agent` đọc Thuyết minh BCTC + `Web Search Tool` quét tin tức vĩ mô, biến động giá nguyên vật liệu, chính sách lãi suất tác động đến ngành.
5. **Character (Uy tín & Pháp lý ban điều hành):**
   * `Web Search Tool` truy quét từ khóa: `"{company_name} bị kiện OR xử phạt thuế OR khởi tố OR vi phạm môi trường"`. Đây là chốt chặn phát hiện rủi ro gian lận mà BCTC tự lập không bao giờ ghi nhận.

---

## 4. Cơ Chế Phán Quyết Tín Dụng Chuẩn Ngân Hàng (Banking-Grade Decisioning)

> ⚠️ **Sự thật bắt buộc:** Theo quy định của Ngân hàng Nhà nước và Basel II/III, **không có ngân hàng nào cho phép AI tự động bấm nút "Giải ngân" (Auto-Disburse) cho doanh nghiệp.** Trách nhiệm pháp lý luôn thuộc về con người (Chuyên viên & Hội đồng Tín dụng).

Vì vậy, FinRisk AI không trả về quyết định nhị phân ngây thơ (`APPROVE` / `REJECT`), mà triển khai **Ma trận phân quyền phê duyệt (Delegation of Authority - DoA)** kết hợp **Giao ước tín dụng (Credit Covenants)**:

### 4.1. Ma trận Phân quyền & Định tuyến Thẩm định

| Risk Score | Cấp độ | Phán quyết đề xuất | Cơ chế can thiệp của Con người (HITL) |
|---|---|---|---|
| **< 40** | `LOW / MEDIUM` | **FAST_TRACK_REVIEW** | **1-Click Review:** AI tóm tắt 5 chỉ số then chốt, Chuyên viên chỉ mất 1-2 phút rà soát và ký duyệt. Rút ngắn 90% thời gian thẩm định. |
| **40 – 69** | `HIGH` | **STANDARD_AUDIT** | **Thẩm định chuyên sâu:** Kích hoạt ngắt luồng (`interrupt`). Yêu cầu Trưởng phòng Tín dụng và Thẩm định rủi ro độc lập phúc tra chi tiết. |
| **$\ge$ 70** | `CRITICAL` | **DECLINE_RECOMMENDED** | **Kiến nghị Từ chối:** Báo cáo chỉ rõ các Red Flags vi phạm khẩu vị rủi ro để chuyên viên trả lời từ chối khách hàng ngay vòng đầu. |

### 4.2. Giao ước tín dụng (Credit Covenants) do AI đề xuất
Với các hồ sơ chấp thuận, AI tự động soạn thảo các điều khoản ràng buộc:
- *Covenant tài chính:* Yêu cầu doanh nghiệp duy trì $D/E \le 2.5$ và $DSCR \ge 1.2$ trong suốt thời hạn vay.
- *Covenant dòng tiền:* Tối thiểu 70% doanh thu từ phương án kinh doanh phải nộp qua tài khoản tại ngân hàng cho vay.

---

## 5. FinRisk AI giải quyết bài toán này như thế nào?

FinRisk AI là một **Hệ thống Multi-Agent tự động hóa toàn trình**:

1. **Thu thập dữ liệu đa nguồn (Multi-Source Ingestion):**
   * BCTC PDF $\to$ Bóc tách bảng biểu bằng `pdfplumber` $\to$ Qdrant Hybrid Search.
   * Đơn đề nghị vay $\to$ Parse các thông số: Hạn mức, kỳ hạn, lãi suất, TSBĐ.
   * Tin tức thị trường $\to$ Web Search API tra cứu rủi ro pháp lý / đạo đức.

2. **Tính toán Toán học chính xác (Math Agent):**
   * Gọi Python Tools tính: Altman Z-Score, DSCR lịch sử, **Pro-forma DSCR**, D/E, Quick Ratio, **LTV**. Tuyệt đối chống Hallucination số học.

3. **Phân tích Rủi ro & Soạn thảo Covenants (Risk Agent):**
   * Chấm điểm rủi ro định lượng + định tính theo Khung 5C.
   * Đề xuất hạn mức an toàn và các điều khoản giao ước tín dụng ràng buộc.

4. **Điều phối & Kiểm soát Con người (Supervisor & HITL Node):**
   * Nếu rủi ro cao hoặc vi phạm chính sách $\to$ `interrupt()` dừng pipeline, gửi hồ sơ cho cấp có thẩm quyền phê duyệt.
   * Phân loại luồng: Fast-track cho hồ sơ xanh, Deep Audit cho hồ sơ cảnh báo.

---

## 6. Các Chỉ số Tài chính Cốt lõi (Core Financial Metrics)

Hệ thống FinRisk AI được lập trình (qua Python Tools) để tính toán chính xác tuyệt đối các chỉ số này:

### 6.1. Altman Z-Score (Dự báo nguy cơ phá sản)
* **Ý nghĩa:** Là mô hình thống kê nổi tiếng nhất để dự báo xác suất một công ty sản xuất sẽ phá sản trong 2 năm tới.
* **Ngưỡng nghiệp vụ:**
  * `Z > 2.99`: Khu vực an toàn (Safe Zone).
  * `1.81 < Z < 2.99`: Khu vực cảnh báo (Grey Zone).
  * `Z < 1.81`: Nguy cơ phá sản rất cao (Distress Zone) $\to$ Kích hoạt Human-in-the-Loop.

### 6.2. DSCR Lịch Sử & Pro-forma DSCR Dự Phóng (Hệ số khả năng trả nợ)
* **DSCR Lịch sử:** Đánh giá khả năng trả các khoản nợ cũ từ dòng tiền kinh doanh trong quá khứ.
* **Pro-forma DSCR (Tối quan trọng):** Đánh giá khả năng gánh thêm khoản vay MỚI đang đề nghị:
  $$\text{Pro-forma DSCR} = \frac{\text{CFO (Dòng tiền thuần HĐKD)}}{\text{Nghĩa vụ nợ cũ} + \text{Gốc và Lãi khoản vay mới}}$$
* **Ngưỡng nghiệp vụ:**
  * `DSCR < 1.0`: Báo động đỏ! Doanh nghiệp không tạo đủ tiền trả nợ, cấm duyệt vay nếu không có bảo lãnh đặc biệt.
  * `1.0 <= DSCR < 1.25`: Biên an toàn mỏng, cần điều kiện giám sát dòng tiền gắt gao.
  * `DSCR >= 1.25`: Mức an toàn tiêu chuẩn ngân hàng chấp nhận.

### 6.3. LTV (Loan-to-Value - Tỷ lệ Khoản vay trên Tài sản bảo đảm)
* **Ý nghĩa:** Giới hạn an toàn phòng vệ nếu khách hàng vỡ nợ:
  $$\text{LTV} = \frac{\text{Số tiền xin vay}}{\text{Giá trị định giá TSBĐ}} \times 100\%$$
* **Ngưỡng nghiệp vụ:**
  * Bất động sản (đất, nhà xưởng): $\text{LTV} \le 70\%$.
  * Máy móc thiết bị / Phương tiện vận tải: $\text{LTV} \le 50\%$.
  * Hàng tồn kho / Khoản phải thu: $\text{LTV} \le 30 - 40\%$.

### 6.4. D/E (Debt-to-Equity - Hệ số Nợ trên Vốn chủ sở hữu)
* **Ý nghĩa:** Đo lường đòn bẩy tài chính. Công ty đang dùng tiền túi (Vốn) hay tiền đi vay (Nợ) để kinh doanh?
* **Ngưỡng nghiệp vụ:** Phụ thuộc vào ngành nghề. Ngành sản xuất thường yêu cầu $< 2.0$. Cao hơn 3.0 là đòn bẩy quá rủi ro.

### 6.5. Quick Ratio (Hệ số thanh toán nhanh)
* **Ý nghĩa:** Khả năng quy đổi tài sản nhanh (Tiền + Đầu tư ngắn hạn + Phải thu) để trang trải nợ ngắn hạn đến hạn mà không cần thanh lý hàng tồn kho.
* **Ngưỡng nghiệp vụ:** Yêu cầu $> 1.0$ (tối thiểu $0.8$).

---

## 7. Giá trị cốt lõi của dự án khi nói chuyện với nhà tuyển dụng

Khi mang dự án này đi phỏng vấn tại các công ty Fintech / Ngân hàng (Techcombank, MB, VPBank, MoMo, VNPay...), bạn không nói: *"Em làm chatbot hỏi đáp tài liệu BCTC"*.

Bạn cần nói:
> *"Em xây dựng một Credit Risk Multi-Agent System mô phỏng toàn trình nghiệp vụ thẩm định tín dụng doanh nghiệp theo Khung 5C chuẩn quốc tế. Hệ thống không chỉ đọc BCTC qua Deep RAG (BGE-M3 + BM25 Qdrant), mà còn kết hợp Đơn đề nghị vay để tính Pro-forma DSCR và LTV tài sản bảo đảm. Để phòng ngừa rủi ro pháp lý, hệ thống không dùng AI tự duyệt mà áp dụng Ma trận phân quyền (Delegation of Authority) và Human-in-the-Loop ngắt luồng khi phát hiện rủi ro cao hoặc vi phạm khẩu vị rủi ro."*

Đó là cách tư duy của một kỹ sư xây dựng **Sản phẩm AI Ngân hàng (Enterprise AI Product)** giải quyết bài toán nghiệp vụ thật, khác biệt hoàn toàn với một demo RAG thông thường.
