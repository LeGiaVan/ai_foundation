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

## 2. Khung chuẩn Thẩm định Tín dụng Quốc tế (Mô hình 5C)

Nếu làm ra một sản phẩm AI cho ngân hàng mà không tuân theo bộ khung (framework) chuẩn, hệ thống đó sẽ không có giá trị thực tiễn. Trong ngành tài chính, khung chuẩn mực và phổ biến nhất được các Ngân hàng Trung ương (bao gồm NHNN Việt Nam) sử dụng làm nền tảng là **Mô hình 5C (The 5 C's of Credit)**.

FinRisk AI được thiết kế để tự động hóa phần lớn mô hình này:

1. **Capacity (Năng lực trả nợ):** *Đây là yếu tố quan trọng nhất.* Doanh nghiệp có tạo ra đủ tiền mặt để trả nợ không? 
   * **Trong FinRisk AI:** Trọng tâm của `Math Agent`. Tính toán tự động các chỉ số **DSCR (Hệ số khả năng trả nợ)**, Biên lợi nhuận, và Dòng tiền thuần.
2. **Capital (Vốn chủ sở hữu):** Doanh nghiệp bỏ bao nhiêu "tiền túi" vào dự án so với tiền đi vay? (Ngân hàng không muốn chịu 100% rủi ro).
   * **Trong FinRisk AI:** Tính toán tự động chỉ số đòn bẩy tài chính **D/E (Debt-to-Equity)**.
3. **Conditions (Điều kiện kinh tế/ngành):** Tình hình vĩ mô, lãi suất, hoặc xu hướng ngành nghề của doanh nghiệp.
   * **Trong FinRisk AI:** `Risk Agent` sử dụng LLM để đọc Thuyết minh BCTC, tìm kiếm thông tin về biến động thị trường, chính sách kế toán thay đổi, hoặc các vụ kiện tụng (Red flags).
4. **Character (Uy tín tín dụng):** Lịch sử trả nợ cũ (CIC), đạo đức ban điều hành, mức độ minh bạch.
   * **Trong FinRisk AI:** Hiện tại khó số hóa 100%, đây là lý do cần **Human-in-the-Loop** để chuyên viên tín dụng đánh giá yếu tố con người.
5. **Collateral (Tài sản đảm bảo):** Bất động sản, máy móc thế chấp nếu doanh nghiệp vỡ nợ.
   * **Trong FinRisk AI:** Được tích hợp trong việc trích xuất bảng danh mục Tài sản cố định từ file PDF BCTC.

*(Hệ thống của chúng ta — FinRisk AI — giải quyết cực tốt **Capacity, Capital, và Conditions** bằng RAG và Tool Calling).*

---

## 3. FinRisk AI giải quyết bài toán này như thế nào?

FinRisk AI không chỉ là một "Chatbot hỏi đáp PDF". Nó là một **Hệ thống Agent tự động hóa quy trình nghiệp vụ**.

### 2.1. Quy trình của FinRisk AI (Mô phỏng 1 Chuyên viên Tín dụng)

1. **Thu thập dữ liệu (Data Ingestion & RAG):**
   * Người dùng tải lên BCTC (PDF).
   * Hệ thống tự động đọc, bóc tách bảng biểu và lưu vào Vector Database (Qdrant).
   * *Nghiệp vụ:* Tương đương việc chuyên viên lật mở từng trang BCTC để tìm số liệu năm ngoái và năm nay.

2. **Trích xuất & Tính toán Toán học (Math Agent):**
   * LLM không tự nhẩm toán vì dễ sai (Hallucination). Thay vào đó, AI gọi công cụ (Tool Calling) để tính chính xác.
   * *Nghiệp vụ:* Tương đương việc chuyên viên lập công thức trên file Excel.

3. **Phân tích Chuyên sâu (Risk Agent):**
   * Dựa trên các chỉ số vừa tính, AI so sánh với chuẩn ngành (ví dụ: nợ/vốn chủ sở hữu của ngành BĐS khác với ngành Bán lẻ).
   * Tìm kiếm các "Red flags" trong phần Thuyết minh BCTC (ví dụ: công ty đang bị kiện, thay đổi chính sách kế toán).

4. **Ra quyết định & Cảnh báo (Human-in-the-Loop):**
   * Nếu công ty tốt: Đề xuất phê duyệt tín dụng.
   * Nếu rủi ro quá cao (Vượt ngưỡng): Agent tự động **tạm dừng (Interrupt)** và gửi thông báo cho Giám đốc Tín dụng vào xác nhận trước khi đi tiếp.

---

## 3. Các Chỉ số Tài chính Cốt lõi (Core Financial Metrics)

Hệ thống FinRisk AI cần được lập trình (qua Python Tools) để tính 4 chỉ số quan trọng nhất này:

### 3.1. Altman Z-Score (Dự báo nguy cơ phá sản)
* **Ý nghĩa:** Là mô hình thống kê nổi tiếng nhất để dự báo xác suất một công ty sản xuất sẽ phá sản trong 2 năm tới.
* **Ngưỡng nghiệp vụ:**
  * `Z > 2.99`: Khu vực an toàn (Safe Zone).
  * `1.81 < Z < 2.99`: Khu vực cảnh báo (Grey Zone).
  * `Z < 1.81`: Nguy cơ phá sản rất cao (Distress Zone) -> **FinRisk AI phải kích hoạt Human-in-the-Loop tại đây.**

### 3.2. DSCR (Debt Service Coverage Ratio - Hệ số khả năng trả nợ)
* **Ý nghĩa:** Trong 1 năm, công ty tạo ra được 10 đồng tiền mặt, nhưng phải trả nợ (gốc+lãi) ngân hàng 8 đồng. Vậy DSCR = 10 / 8 = 1.25.
* **Ngưỡng nghiệp vụ:**
  * `DSCR < 1.0`: Báo động đỏ! Công ty không tạo ra đủ tiền để trả nợ, phải đi vay mới trả nợ cũ. Cấm cho vay.
  * `DSCR > 1.25`: Mức an toàn tối thiểu mà các ngân hàng chấp nhận.

### 3.3. D/E (Debt-to-Equity - Hệ số Nợ trên Vốn chủ sở hữu)
* **Ý nghĩa:** Đo lường đòn bẩy tài chính. Công ty đang dùng tiền túi (Vốn) hay tiền đi vay (Nợ) để kinh doanh?
* **Ngưỡng nghiệp vụ:** Phụ thuộc vào ngành nghề. Ngành ngân hàng/BĐS có thể cao (3.0 - 5.0), nhưng ngành sản xuất thường yêu cầu `< 1.5`. Cao quá nghĩa là gánh nặng lãi vay lớn.

### 3.4. Quick Ratio (Hệ số thanh toán nhanh)
* **Ý nghĩa:** Nếu ngày mai tất cả chủ nợ đến đòi tiền, công ty có đủ tài sản "có thể quy ra tiền mặt ngay lập tức" (Tiền mặt + Đầu tư ngắn hạn + Khoản phải thu) để trả không? Không tính Hàng tồn kho vì không dễ bán ngay.
* **Ngưỡng nghiệp vụ:** Thường yêu cầu `> 1.0`.

---

## 4. Giá trị cốt lõi của dự án khi nói chuyện với nhà tuyển dụng

Khi mang dự án này đi phỏng vấn tại các công ty Fintech / Ngân hàng (Techcombank, VPBank, Momo, VNPay...), bạn không nói: *"Em làm chatbot hỏi đáp tài liệu"*.

Bạn cần nói:
> *"Em xây dựng một Agent Workflow giúp tự động hóa khâu trích xuất số liệu BCTC và tính toán các chỉ số rủi ro như DSCR, Z-Score. Điểm đặc biệt của hệ thống là khả năng chống Hallucination số học nhờ ép LLM dùng Python Tools, và cơ chế Human-in-the-Loop để Giám đốc Tín dụng can thiệp khi Z-Score rớt xuống dưới 1.8"*

Đó là cách tư duy của một kỹ sư xây dựng **Sản phẩm AI (AI Product)** giải quyết bài toán nghiệp vụ thật, khác biệt hoàn toàn với một người chỉ biết làm **AI Demo**.
