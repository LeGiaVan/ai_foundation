# Giai đoạn 4 — Tuần 10-11: Evaluation & Observability

## Mục tiêu
- Đánh giá chất lượng RAG bằng các metric cốt lõi của **RAGAS**: faithfulness, answer relevancy, context precision, context recall.
- Sử dụng **LLM-as-a-judge** (G‑Eval) với bộ dữ liệu vàng (golden dataset) để đo độ tin cậy của câu trả lời.
- Thiết lập **observability** toàn diện qua **Langfuse**: trace, span, token/cost tracking, latency, prompt versioning.
- Tự động hoá quy trình đánh giá, tích hợp CI/CD và tạo báo cáo dạng Markdown/JSON.

## Các bước thực hiện
1. **Chuẩn bị môi trường**
   - Thêm các biến môi trường vào `.env`:
     ```
     LANGFUSE_PUBLIC_KEY=...
     LANGFUSE_SECRET_KEY=...
     LANGFUSE_PROJECT=rag_evaluation
     GOLDEN_DATA_PATH=./data/golden_dataset.json
     EVAL_MODEL=groq/compound-mini   # hoặc model cao hơn cho judge
     ```
   - Cài đặt phụ thuộc:
     ```
     pip install ragas deepeval langfuse pandas
     ```

2. **Xây dựng các công cụ đánh giá**
   - `eval_metrics.py`: wrapper cho RAGAS (faithfulness, relevancy, precision, recall) và DeepEval.
   - `g_eval.py`: chạy G‑Eval trên golden dataset, trả về JSON với `score`, `explanation`.

3. **Kết nối Langfuse**
   - `langfuse_setup.py`: khởi tạo client, tạo project nếu chưa có, cung cấp hàm `log_trace`.
   - Thêm `@traceable` vào endpoint FastAPI (`rag_chatbot.py`) và vào pipeline evaluation.

4. **Orchestrator**
   - `run_evaluation.py`:
     - Tải RAG chain hiện có.
     - Chạy một batch câu hỏi (ví dụ 20 query).
     - Ghi lại các metric, thời gian, token usage.
     - Đẩy trace lên Langfuse.
     - Xuất báo cáo `reports/eval_YYYYMMDD.md`.

5. **CI/CD**
   - Tạo workflow GitHub Actions `ci/evaluation.yml`:
     - Chạy `run_evaluation.py` mỗi khi có push vào `main`.
     - Lưu báo cáo dưới dạng artifact.
     - Nếu metric trung bình < ngưỡng, đánh dấu job là thất bại.

6. **Tài liệu & Notebook**
   - `notebooks/Evaluation_Playbook.ipynb`: hướng dẫn từng bước, visualisation Langfuse dashboard, cách thay đổi version prompt.
   - Cập nhật `README.md` phần Evaluation.

## Tham khảo
- **RAGAS**: https://docs.ragas.io/en/stable/getstarted/
- **DeepEval**: https://deepeval.com/docs/getting-started-rag
- **Langfuse**: https://langfuse.com/docs/observability/get-started

## Kiểm tra & Bảo trì
- Unit test (`tests/test_eval_metrics.py`) cho mỗi metric.
- Kiểm tra trace trên Langfuse sau mỗi run.
- Cập nhật golden dataset mỗi 2‑4 tuần để phản ánh nhu cầu mới.

---

**Hành động tiếp theo**: Xác nhận tên collection Qdrant, model LLM‑as‑judge, và giới hạn chi phí token nếu có. Khi nhận phản hồi, tôi sẽ tạo `task.md` và bắt đầu triển khai.