# 📊 Cheatsheet: Đánh giá RAG (RAGAS Framework)

## 1️⃣ Tại sao cần đánh giá RAG tự động?
Khi xây dựng hệ thống RAG, việc đánh giá bằng mắt người (manual evaluation) tốn rất nhiều thời gian và không thể mở rộng (scale) khi tập dữ liệu lên đến hàng ngàn câu hỏi.
**RAGAS (Retrieval Augmented Generation Assessment)** là một framework giúp tự động hóa quá trình đánh giá này bằng cách sử dụng chính LLM như một giám khảo (LLM-as-a-judge) để chấm điểm các thành phần trong pipeline của bạn.

---

## 2️⃣ 4 Chỉ số (Metrics) cốt lõi của RAGAS

RAGAS chia RAG thành 2 phần: **Retrieval (Tìm kiếm)** và **Generation (Sinh văn bản)**, và có các chỉ số tương ứng.

### 📝 Generation Metrics (Đánh giá phần LLM sinh ra)

#### 1. Faithfulness (Độ trung thực)
- **Ý nghĩa:** Trả lời cho câu hỏi: *"Câu trả lời của LLM có hoàn toàn dựa trên context được cung cấp hay không?"*
- **Tác dụng:** Đo lường mức độ **Hallucination** (ảo giác) của LLM.
- **Tính toán:** LLM giám khảo bóc tách câu trả lời thành các ý nhỏ (statements). Sau đó kiểm tra xem từng ý nhỏ đó có thể được suy ra từ Context hay không. 
- Điểm = Số ý được support bởi Context / Tổng số ý. (0 đến 1, càng cao càng tốt).

#### 2. Answer Relevancy (Độ liên quan của câu trả lời)
- **Ý nghĩa:** Trả lời cho câu hỏi: *"Câu trả lời có đi đúng trọng tâm câu hỏi ban đầu không?"*
- **Tác dụng:** Tránh trường hợp LLM lan man hoặc trả lời vòng vo không đúng ý người hỏi.
- **Tính toán:** RAGAS dùng model để tạo ngược lại (reverse-engineer) các câu hỏi tiềm năng dựa trên Câu trả lời. Sau đó tính Cosine Similarity giữa các câu hỏi được tạo ra và câu hỏi gốc.

### 🔍 Retrieval Metrics (Đánh giá phần Tìm kiếm/Vector DB)

#### 3. Context Precision (Độ chính xác của Context)
- **Ý nghĩa:** Trả lời cho câu hỏi: *"Các chunk thực sự hữu ích có nằm ở TOP ĐẦU của danh sách kết quả không?"*
- **Tác dụng:** Đánh giá chất lượng xếp hạng (ranking) của hệ thống tìm kiếm (Qdrant/BM25/Reranker).
- **Tính toán:** LLM chấm điểm từng chunk trong danh sách kết quả xem nó có chứa câu trả lời cho câu hỏi gốc không. Hệ thống tính điểm cao hơn nếu các chunk hữu ích xuất hiện sớm hơn trong danh sách.

#### 4. Context Recall (Độ phủ của Context)
- **Ý nghĩa:** Trả lời cho câu hỏi: *"Hệ thống tìm kiếm có lấy đủ TOÀN BỘ thông tin cần thiết để trả lời câu hỏi không?"*
- **Tác dụng:** Đảm bảo LLM có đầy đủ nguyên liệu để sinh câu trả lời hoàn chỉnh. (Yêu cầu phải có `ground_truth` - câu trả lời chuẩn từ con người).
- **Tính toán:** LLM giám khảo lấy `ground_truth` chia thành các ý nhỏ, rồi kiểm tra xem các ý này có xuất hiện trong Context đã được truy xuất (retrieved context) hay không.

---

## 3️⃣ Code Thực Hành (Sử dụng RAGAS)

**Cài đặt:**
```bash
pip install ragas datasets
```

**Chuẩn bị Dataset:**
Để đánh giá RAGAS, bạn cần chuẩn bị một tập dữ liệu theo định dạng của thư viện `datasets` (Hugging Face). Dữ liệu tối thiểu cần có các trường:
- `question`: Câu hỏi đầu vào
- `answer`: Câu trả lời hệ thống RAG của bạn sinh ra
- `contexts`: Danh sách các chunks (list of strings) mà hệ thống RAG tìm được
- `ground_truths`: (Optional nhưng cần cho Context Recall) Danh sách câu trả lời chuẩn do con người tự viết (dạng list of list).

```python
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# RAGAS mặc định dùng OpenAI làm giám khảo
os.environ["OPENAI_API_KEY"] = "sk-..." 

# 1. Chuẩn bị dữ liệu 
# (Thường bạn sẽ chạy RAG pipeline qua 10-50 câu hỏi để gom data này vào dict)
data = {
    "question": ["Cơ chế hoạt động của Bi-encoder là gì?"],
    "answer": ["Bi-encoder nhúng câu hỏi và tài liệu độc lập thành các vector."],
    "contexts": [
        ["Bi-encoder xử lý query và chunk qua hai luồng song song không tương tác."],
        ["Cross-encoder chậm nhưng chính xác."]
    ],
    "ground_truths": [["Nó nhúng câu hỏi và tài liệu độc lập thành vector rồi so sánh cosine."]]
}

dataset = Dataset.from_dict(data)

# 2. Chạy Evaluation
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
)

# 3. In kết quả (result chứa điểm trung bình của các metric)
print(result)
```

### 💡 Tips & Lưu ý
- **RAGAS rất tốn API LLM**: Quá trình LLM-as-a-judge (LLM đóng vai trò giám khảo) chạy ngầm sẽ gọi API rất nhiều lần (để trích xuất ý, tính cosine). Ban đầu nên test với tập nhỏ (5-10 câu) trước khi chạy cả bộ test.
- Bạn hoàn toàn có thể thay đổi/tùy chỉnh LLM giám khảo (ví dụ thay OpenAI bằng Claude hoặc mô hình local thông qua Langchain) nếu không muốn dùng model mặc định.
