# 📚 CHEAT SHEET: Evaluation & Observability cho LLM & RAG
## Giai đoạn 4: RAGAS, DeepEval, LLM-as-a-Judge (G-Eval) & Langfuse

---

## 1. TỔNG QUAN VÀ MENTAL MODEL

Trong hệ thống GenAI / RAG cấp doanh nghiệp (Enterprise), việc viết code chạy được chỉ chiếm **30%**. **70%** còn lại là đo lường chất lượng định lượng, ngăn ngừa ảo giác (Hallucination) và giám sát chi phí/độ trễ trên production.

```
                   ┌───────────────────────────────────────────────────────────┐
                   │               HỆ THỐNG ĐÁNH GIÁ & GIÁM SÁT                 │
                   └─────────────────────────────┬─────────────────────────────┘
                                                 │
                  ┌──────────────────────────────┴──────────────────────────────┐
                  ▼                                                             ▼
     [OFFLINE EVALUATION]                                          [ONLINE OBSERVABILITY]
   (Kiểm chuẩn trước khi Deploy)                               (Giám sát thời gian thực Production)
  - RAGAS (RAG Triad Metrics)                                  - Langfuse Tracing (Execution Tree)
  - DeepEval (Unit test CI/CD)                                 - Latency (TTFT, P50, P95, P99)
  - G-Eval (LLM-as-a-Judge)                                    - Token Usage & Cost Tracking
  - Golden Dataset (50-100 ground truths)                      - User Feedback (Thumbs Up/Down)
```

---

## 2. BỘ TỨ METRIC CỐT LÕI CỦA RAG (RAG TRIAD & RAGAS)

Một hệ thống RAG gồm 2 thành phần chính: **Retriever** (Truy xuất) và **Generator** (Bộ sinh). Bộ tứ metric này giúp khoanh vùng chính xác lỗi nằm ở khâu nào.

<p align="center">
  <img src="RAGAS.png" alt="System Architecture" width="800" />
</p>

### 2.1. Faithfulness (Độ trung thực — Chống Ảo giác)
* **Ý nghĩa:** Kiểm tra xem mọi nhận định/số liệu trong câu trả lời có **bắt nguồn 100% từ Context** được truy xuất hay không, hay do LLM "tự bịa" ra.
* **Bắt buộc:** Trong tài chính, y tế, pháp lý, chỉ số này phải $\ge 0.95$.
* **Công thức tính:**
  $$\text{Faithfulness} = \frac{|\text{Số lượng claim (luận điểm) được chứng minh bởi Context}|}{|\text{Tổng số lượng claim có trong Answer}|}$$
* **Cơ chế hoạt động:**
  1. Tách `Answer` thành từng câu/mệnh đề nhỏ (Claims).
  2. Dùng LLM kiểm chứng từng Claim xem có bằng chứng trong `Context` không (Yes/No).
  3. Lấy trung bình cộng.

### 2.2. Answer Relevancy (Độ liên quan của câu trả lời)
* **Ý nghĩa:** Đo lường mức độ câu trả lời giải quyết trực tiếp câu hỏi của người dùng, không phụ thuộc vào việc đúng sự thật hay không (không xét tính đúng sai, chỉ xét tính đúng trọng tâm, không trả lời lan man, lạc đề).
* **Công thức tính:**
  LLM đọc `Answer` và tạo ngược lại $N$ câu hỏi giả định ($q_i$). Sau đó đo độ tương đồng Cosine giữa vector của câu hỏi gốc ($q$) và các câu hỏi giả định ($q_i$):
  $$\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \text{CosineSimilarity}(E(q), E(q_i))$$

### 2.3. Context Precision (Độ chính xác của ngữ cảnh - Đo Retriever)
* **Ý nghĩa:** Đo lường xem các chunk thực sự liên quan có nằm ở **vị trí đầu (Top rank)** của danh sách kết quả hay không.
* **Công thức:** Dựa trên Mean Average Precision@K (MAP@K).
  $$\text{Context Precision@K} = \frac{\sum_{k=1}^K (\text{Precision@}k \times v_k)}{\text{Tổng số chunk liên quan trong top } K}$$
  *(Với $v_k \in \{0, 1\}$ là cờ đánh dấu chunk thứ $k$ có chứa thông tin trả lời hay không).*
* **Ý nghĩa thực tế:** Nếu Context Precision thấp $\rightarrow$ Cần thêm **Cross-Encoder Reranker** để đẩy chunk đúng lên vị trí 1, 2.

### 2.4. Context Recall (Độ bao phủ của ngữ cảnh - Đo Retriever)
* **Ý nghĩa:** Đo lường xem Retriever có lấy **đủ tất cả thông tin** cần thiết được định nghĩa trong `Ground Truth` hay không.
* **Công thức:**
  $$\text{Context Recall} = \frac{|\text{Số luận điểm trong Ground Truth được tìm thấy trong Context}|}{|\text{Tổng số luận điểm có trong Ground Truth}|}$$
* **Ý nghĩa thực tế:** Nếu Context Recall thấp $\rightarrow$ Chunk size quá nhỏ bị mất ngữ cảnh, hoặc số $k$ (Top-K) quá ít, cần tăng $k$ hoặc dùng Hybrid Search (Dense + BM25).

---

### Bảng Chẩn đoán Lỗi RAG Nhanh (RAG Triad Diagnostic Matrix)

| Triệu chứng | Nguyên nhân gốc rễ | Giải pháp khắc phục |
| :--- | :--- | :--- |
| **Faithfulness Thấp**, Recall Cao | LLM bị hallucination, prompt không chặt | Hạ `temperature=0.0`, đổi prompt ép trích dẫn nguồn, chuyển sang model suy luận cao hơn. |
| **Context Recall Thấp** | Retriever bỏ sót dữ liệu quan trọng | Tăng Top-K, dùng Sentence-window retrieval, phối hợp BM25 (Hybrid search). |
| **Context Precision Thấp** | Chunks lấy về nhiều rác, thứ tự sai | Tích hợp **Cohere / BGE Reranker**, lọc bớt chunk có cosine similarity $< 0.7$. |
| **Answer Relevancy Thấp** | Trả lời vòng vo, không đúng trọng tâm | Tinh chỉnh System Prompt, bổ sung Few-shot examples, ép trả về Structured Output (JSON). |

---

## 3. LLM-AS-A-JUDGE & PHƯƠNG PHÁP G-EVAL

### 3.1. Khái niệm & Rủi ro cần kiểm soát
* **LLM-as-a-Judge:** Sử dụng một mô hình LLM mạnh (GPT-4o, Claude 3.5 Sonnet) để đóng vai giám khảo chấm điểm câu trả lời của mô hình nhỏ hơn (Llama-3-8B, GPT-4o-mini).
* **3 Bẫy sai lệch (Biases) cốt lõi và Ví dụ thực chiến:**

  #### 1. Position Bias (Thiên vị thứ tự xuất hiện trước/sau)
  * **Hiện tượng:** Khi so sánh cặp (Pairwise Comparison) giữa 2 câu trả lời $A$ và $B$, LLM Judge (như GPT-4o) có xu hướng chọn câu trả lời ở vị trí đầu tiên (hoặc cuối cùng) với tỷ lệ áp đảo (60%–70%) chỉ vì hiệu ứng vị trí đọc, bất kể chất lượng.
  * **Ví dụ thực tế:**
    ```text
    Prompt: "Câu trả lời nào tốt hơn cho câu hỏi: 'Công thức tính ROE là gì?'
    [Câu 1 - Model A]: ROE = Lợi nhuận sau thuế / Vốn chủ sở hữu bình quân.
    [Câu 2 - Model B]: ROE là tỷ số phản ánh 1 đồng vốn cổ đông tạo ra bao nhiêu lợi nhuận."
    -> Kết quả: Judge chọn [Câu 1].
    
    Khi đảo vị trí đưa Model B lên Câu 1, Model A xuống Câu 2:
    -> Kết quả: Judge lại tiếp tục chọn [Câu 1] (lúc này là Model B)!
    ```
  * **Cách khắc phục:** **Position Swapping (Đảo vị trí hai chiều)**. Chạy đánh giá 2 lần liên tiếp: lượt 1 `(A, B)` và lượt 2 `(B, A)`. Chỉ công nhận $A$ thắng nếu $A$ được chọn ở cả 2 lượt; nếu kết quả mâu thuẫn $\rightarrow$ Đánh dấu **Hòa (Tie)**.

  #### 2. Verbosity Bias (Thiên vị câu trả lời dài dòng, hoa mỹ)
  * **Hiện tượng:** LLM giám khảo có xu hướng ngộ nhận câu trả lời dài, nhiều chữ, định dạng hoa mỹ là "chuyên sâu, đầy đủ", và chấm điểm cao hơn câu trả lời ngắn gọn dù câu ngắn trả lời chính xác 100% trọng tâm.
  * **Ví dụ thực tế:**
    ```text
    User: "Thủ đô của nước Úc (Australia) là thành phố nào?"
    
    - Model A (Chính xác, súc tích): "Thủ đô của Úc là Canberra."
      -> Điểm Judge chấm: 7/10 (bị chê là quá ngắn, thiếu bối cảnh).
      
    - Model B (Dài dòng, lan man): "Nước Úc là quốc gia xinh đẹp ở Nam bán cầu. Nhiều người hay nhầm Sydney hoặc Melbourne là thủ đô. Tuy nhiên, năm 1908 thủ đô được chọn là Canberra như một sự thỏa hiệp..."
      -> Điểm Judge chấm: 9.5/10 (khen là chi tiết, giàu thông tin).
    ```
  * **Cách khắc phục:** Bổ sung ràng buộc độ súc tích vào System Prompt của Judge:
    ```text
    "Conciseness Rule: Do NOT reward verbosity. If a question can be answered directly in 1 sentence, reward brevity and heavily penalize long-winded, fluffy explanations."
    ```

  #### 3. Self-Enhancement Bias (Thiên vị 'cùng dòng họ / nhà phát triển')
  * **Hiện tượng:** LLM giám khảo ưu ái các câu trả lời có văn phong tương đồng với phong cách huấn luyện RLHF của chính công ty mình phát triển (ví dụ: GPT-4o ưu ái GPT-4o-mini hơn Claude 3.5 Haiku từ 5%–15%, và ngược lại Claude 3.5 Sonnet ưu ái các model họ Claude).
  * **Ví dụ thực tế:** Khi đánh giá một câu giải trình rủi ro tài chính, GPT-4o chấm model của OpenAI 8.8 điểm nhưng chỉ cho Claude 8.0 điểm dù nội dung tương đương, chỉ vì model OpenAI dùng các mẫu câu "It's important to consider..." quen thuộc với dữ liệu huấn luyện của OpenAI.
  * **Cách khắc phục:**
    * **Ẩn danh hoàn toàn (Anonymization):** Xóa bỏ toàn bộ tên model, thẻ metadata trước khi đưa vào Judge.
    * **Hội đồng giám khảo đa dạng (Multi-Judge Panel):** Sử dụng ít nhất 3 model từ các hãng độc lập (ví dụ: 1 model OpenAI + 1 model Anthropic + 1 model Meta Llama mã nguồn mở) rồi lấy điểm trung bình (Consensus).

### 3.2. Phương pháp G-Eval (Chain of Thought + Probability Weighting)

**G-Eval** là khung đánh giá đột phá được nhóm nghiên cứu của Microsoft giới thiệu (*Liu et al., 2023: "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"*). Đây là phương pháp đưa độ tương quan giữa LLM Judge và chuyên gia con người (Human Alignment) từ mức $\sim 0.35$ lên **hơn $0.51$ (Spearman $\rho$)**, vượt xa hoàn toàn các metric truyền thống (BLEU, ROUGE).

<p align="center">
  <img src="G-Eval.png" alt="System Architecture" width="800" />
</p>


#### 1. Tại sao LLM-as-a-Judge thông thường thất bại mà G-Eval lại thành công?
* **Cách chấm truyền thống (Naïve Scoring):** Hỏi LLM: *"Chấm điểm câu trả lời từ 1 đến 5"*.
  * *Hạn chế:* LLM thường chỉ chọn số nguyên (`3`, `4` hoặc `5`), có phương sai cực cao (lúc chạy ra 3, lúc chạy ra 5), và dễ bị ảo giác số học.
* **Đột phá của G-Eval:**
  1. **Evaluation Steps (CoT):** Ép LLM phải suy luận theo quy trình kiểm toán từng bước trước khi chốt điểm.
  2. **Continuous Expected Score (Điểm số kỳ vọng):** Thay vì lấy 1 con số nguyên duy nhất, G-Eval đo **xác suất (probability)** của từng con số từ 1 đến 5, sau đó tính **kỳ vọng toán học**.

---

#### 2. Cơ chế Toán học: Tính điểm kỳ vọng qua Token Logprobs
Khi LLM chuẩn bị sinh ra token điểm số cuối cùng ($S \in \{1, 2, 3, 4, 5\}$), API trả về giá trị `logprob` của từng token ứng viên:

$$\text{logprob}(s) = \ln P(\text{token} = s)$$

Từ đó, xác suất của từng điểm số sau khi chuẩn hóa Softmax qua 5 mức điểm:

$$P(s) = \frac{\exp(\text{logprob}(s))}{\sum_{k=1}^5 \exp(\text{logprob}(k))}$$

**Điểm số G-Eval cuối cùng (Expected Score):**
$$\text{Score} = \sum_{s=1}^5 s \times P(s)$$

> 💡 **Ý nghĩa thực tế:**
> Giả sử LLM phân vân giữa điểm 3 và điểm 4:
> * $P(\text{'3'}) = 45\%$
> * $P(\text{'4'}) = 55\%$
>
> Điểm G-Eval tính ra là: $\text{Score} = (3 \times 0.45) + (4 \times 0.55) = \mathbf{3.55}$.
> Con số $3.55$ phản ánh chính xác mức độ lưỡng lự và rủi ro của câu trả lời, mượt mà hơn rất nhiều so với việc ép cứng thành điểm 3 hay điểm 4.

---

#### 3. Triển khai Code G-Eval thuần túy với OpenAI API (Pure Python)

```python
import math
from openai import OpenAI

client = OpenAI()

def calculate_geval_score(context: str, query: str, answer: str) -> float:
    # 1. Định nghĩa prompt kèm các bước chấm CoT
    prompt = f"""
Bạn là chuyên gia thẩm định rủi ro tín dụng. Hãy đánh giá tính TRUNG THỰC (Faithfulness) của câu trả lời dựa trên tài liệu gốc.

[TÀI LIỆU GỐC (CONTEXT)]:
{context}

[CÂU HỎI]:
{query}

[CÂU TRẢ LỜI CẦN CHẤM]:
{answer}

[CÁC BƯỚC ĐÁNH GIÁ (EVALUATION STEPS)]:
1. Đọc kỹ tài liệu gốc và xác định toàn bộ số liệu/thông tin chính.
2. Tách câu trả lời thành từng luận điểm (claims) riêng biệt.
3. Đối chiếu từng luận điểm với tài liệu gốc:
   - Nếu toàn bộ luận điểm đều có căn cứ trong tài liệu: Chấm 5 điểm.
   - Nếu có 1 luận điểm nhỏ không có căn cứ nhưng không nguy hiểm: Chấm 3-4 điểm.
   - Nếu xuất hiện số liệu tài chính sai lệch nghiêm trọng hoặc tự bịa đặt: Chấm 1-2 điểm.
4. Viết giải thích ngắn gọn, sau đó ở dòng cuối cùng in đúng một chữ số từ 1 đến 5.

Đánh giá và điểm số (1-5):"""

    # 2. Gọi API với logprobs=True và top_logprobs=5
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
        logprobs=True,
        top_logprobs=5
    )

    # 3. Tìm token điểm số ở cuối output
    content_tokens = response.choices[0].logprobs.content
    score_token_logprobs = None
    
    # Duyệt ngược từ token cuối lên để tìm token điểm số '1', '2', '3', '4', hoặc '5'
    for token_data in reversed(content_tokens):
        token_str = token_data.token.strip()
        if token_str in ["1", "2", "3", "4", "5"]:
            score_token_logprobs = token_data.top_logprobs
            break

    if not score_token_logprobs:
        return 3.0 # Fallback an toàn

    # 4. Trích xuất xác suất của các số 1..5 và tính Softmax
    raw_probs = {}
    valid_scores = ["1", "2", "3", "4", "5"]
    
    for item in score_token_logprobs:
        tok = item.token.strip()
        if tok in valid_scores:
            raw_probs[int(tok)] = math.exp(item.logprob)

    total_prob = sum(raw_probs.values()) if raw_probs else 1.0
    
    # 5. Tính điểm kỳ vọng Expected Score
    expected_score = sum(score * (prob / total_prob) for score, prob in raw_probs.items())
    return round(expected_score, 2)
```

---

#### 4. Sử dụng G-Eval dựng sẵn trong Framework DeepEval

Framework `deepeval` đã đóng gói sẵn thuật toán G-Eval, cho phép định nghĩa các tiêu chí tuân thủ phức tạp chỉ với vài dòng code:

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# Định nghĩa tiêu chí tuân thủ quy chế tín dụng ngân hàng bằng G-Eval
credit_compliance_metric = GEval(
    name="Financial Policy Compliance",
    criteria="Xác định xem báo cáo thẩm định có tuân thủ đúng quy định về trích lập dự phòng rủi ro và tỷ lệ nợ/Vốn chủ sở hữu theo Thông tư NHNN hay không.",
    evaluation_steps=[
        "Kiểm tra xem tỷ lệ nợ/VCSH có được tính toán đúng công thức tài chính không.",
        "Xác minh phân loại nhóm nợ (Nhóm 1 đến Nhóm 5) có đúng với số ngày quá hạn ghi nhận trong báo cáo không.",
        "Đánh giá xem đề xuất hạn mức tín dụng có vượt trần cho phép đối với một khách hàng không."
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT
    ],
    model="gpt-4o"
)

# Chạy chấm điểm tự động
test_case = LLMTestCase(
    input="Doanh nghiệp quá hạn nợ 45 ngày thì xếp vào nhóm nợ nào và trích lập bao nhiêu?",
    actual_output="Doanh nghiệp quá hạn 45 ngày thuộc Nhóm 2 (Nợ cần chú ý), trích lập dự phòng cụ thể là 5%.",
    retrieval_context=["Thông tư 11/2021/TT-NHNN: Nhóm 2 (Nợ cần chú ý) gồm nợ quá hạn từ 10 ngày đến 90 ngày, tỷ lệ trích lập dự phòng 5%."]
)

credit_compliance_metric.measure(test_case)
print(f"Điểm số G-Eval: {credit_compliance_metric.score}")
print(f"Giải trình lý do: {credit_compliance_metric.reason}")
```

---

## 4. XÂY DỰNG GOLDEN DATASET

Một bộ dữ liệu kiểm chuẩn vàng (Golden Dataset) tiêu chuẩn cho thẩm định tài chính/kỹ thuật cần lưu dưới dạng `data/golden_dataset.json`:

```json
[
  {
    "id": "eval_fin_001",
    "question": "Biên lợi nhuận gộp quý 3/2023 của Vinamilk (VNM) là bao nhiêu và thay đổi thế nào so với cùng kỳ?",
    "ground_truth": "Biên lợi nhuận gộp quý 3/2023 của Vinamilk đạt 41.9%, tăng 2.4 điểm phần trăm so với mức 39.5% của quý 3/2022 nhờ giá nguyên liệu sữa bột đầu vào hạ nhiệt.",
    "ground_truth_contexts": [
      "BCTC Hợp nhất Q3/2023 Vinamilk: Doanh thu thuần đạt 15.637 tỷ VNĐ, Lợi nhuận gộp đạt 6.551 tỷ VNĐ (tương đương 41.9%). Q3/2022 doanh thu thuần 16.079 tỷ, LN gộp 6.351 tỷ (39.5%)."
    ],
    "metadata": {
      "ticker": "VNM",
      "period": "Q3_2023",
      "difficulty": "medium",
      "requires_math": true
    }
  }
]
```

---

## 5. THỰC CHIẾN IMPLEMENTATION CODE

### 5.1. File `eval_metrics.py` (Chạy RAGAS với Custom Groq/OpenAI)

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
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

def run_ragas_evaluation(eval_samples: list) -> dict:
    """
    eval_samples = [
        {
            "question": "...",
            "answer": "...",
            "contexts": ["chunk 1", "chunk 2"],
            "ground_truth": "..."
        }
    ]
    """
    # 1. Chuyển đổi dữ liệu sang dạng Dataset HuggingFace
    data_dict = {
        "question": [s["question"] for s in eval_samples],
        "answer": [s["answer"] for s in eval_samples],
        "contexts": [s["contexts"] for s in eval_samples],
        "ground_truth": [s["ground_truth"] for s in eval_samples],
    }
    dataset = Dataset.from_dict(data_dict)

    # 2. Khởi tạo Judge LLM và Embeddings
    judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # 3. Chạy đánh giá bộ 4 metric RAG
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=judge_llm,
        embeddings=embeddings
    )
    
    print("\n📊 KẾT QUẢ ĐÁNH GIÁ RAGAS:")
    print(results)
    return results.to_pandas()
```

---

### 5.2. File `test_ci_deepeval.py` (Unit Test Tự Động Hóa Cho CI/CD)

```python
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric
)

def test_rag_faithfulness_and_relevancy():
    """
    Tự động chặn deploy nếu chỉ số dưới ngưỡng cho phép
    """
    # 1. Giả lập kết quả trả về từ RAG Pipeline
    test_case = LLMTestCase(
        input="Tỷ lệ an toàn vốn (CAR) tối thiểu của ngân hàng theo Thông tư 41 là bao nhiêu?",
        actual_output="Theo Thông tư 41/2016/TT-NHNN, tỷ lệ an toàn vốn (CAR) tối thiểu là 8%.",
        expected_output="Tỷ lệ an toàn vốn tối thiểu là 8% theo quy định tại Thông tư 41.",
        retrieval_context=[
            "Thông tư 41/2016/TT-NHNN quy định tỷ lệ an toàn vốn tối thiểu của ngân hàng thương mại, chi nhánh ngân hàng nước ngoài là 8%."
        ]
    )

    # 2. Định nghĩa ngưỡng Metric (Threshold)
    metric_faithfulness = FaithfulnessMetric(threshold=0.9, model="gpt-4o-mini")
    metric_relevancy = AnswerRelevancyMetric(threshold=0.85, model="gpt-4o-mini")
    metric_precision = ContextualPrecisionMetric(threshold=0.8, model="gpt-4o-mini")

    # 3. Assert (Fail nếu điểm số thực tế < ngưỡng)
    assert_test(test_case, [metric_faithfulness, metric_relevancy, metric_precision])
```

---

## 6. OBSERVABILITY TOÀN DIỆN VỚI LANGFUSE

### 6.1. Tại sao cần LLM Observability & Khác biệt với Traditional APM?

Trong các hệ thống phần mềm truyền thống (Web/Microservices), các công cụ APM (Application Performance Monitoring) như **Datadog**, **Prometheus**, **New Relic** tập trung vào các chỉ số hạ tầng và HTTP: CPU/RAM, Request/Second (RPS), HTTP 4xx/5xx, và latency p95.

Tuy nhiên, với hệ thống **GenAI / RAG / Multi-Agent**, HTTP status `200 OK` **không đồng nghĩa với câu trả lời thành công**. Ứng dụng vẫn có thể trả về HTTP 200 nhưng nội dung bị **Hallucination** (bịa đặt số liệu), vi phạm quy định pháp lý, hoặc tiêu tốn chi phí token vượt ngân sách.

#### Bảng so sánh Traditional APM vs. LLM Observability (Langfuse)

| Tiêu chí | Traditional APM (Datadog, Prometheus) | LLM Observability (Langfuse) |
| :--- | :--- | :--- |
| **Đơn vị phân tích cốt lõi** | HTTP Request, Database Query, Function Call | **Trace, Span, Generation, Event, Session, User** |
| **Bản chất dữ liệu I/O** | Dữ liệu có cấu trúc (JSON, SQL records, số định lượng) | **Ngôn ngữ tự nhiên (Prompt, System Prompt, RAG Context, Completion)** |
| **Định nghĩa "Lỗi" (Failure)** | Crash, Exception, HTTP 500, DB Timeout | **Hallucination, Context Mismatch, Toxicity, Jailbreak, Model Drift** |
| **Chỉ số đo lường hiệu năng** | Latency, RPS, Memory Leak, Network I/O | **TTFT (Time-to-first-token), Tokens/sec, Cost ($), Faithfulness, Relevancy** |
| **Đơn vị chi phí** | Giờ chạy server (vCPU/RAM/Cloud node) | **Input Tokens, Output Tokens, Cached Tokens theo từng Model ID** |
| **Vòng lặp cải tiến** | Bug Fix, Refactor Code, Tối ưu SQL query | **Prompt Versioning, Few-shot curation, Reranker tuning, Golden Dataset export** |

---

### 6.2. Mô hình Dữ liệu & Cấu trúc Phân cấp (Langfuse Core Data Model)

Langfuse tổ chức dữ liệu giám sát theo một **cây thực thi phân cấp (Execution Tree / Directed Acyclic Graph)**, phản ánh chính xác từng bước xử lý bên trong một pipeline RAG hoặc chuỗi suy luận của Agent:

```text
Trace: [ID: tr-9a8b7c] - "Thẩm định rủi ro tín dụng Công ty CP Tập đoàn Hòa Phát (HPG)"
│ (Metadata: user_id="bank_analyst_01", session_id="sess_credit_2024", environment="production")
│
├── 1. Span: "preprocess_and_rewrite_query" (Duration: 85ms)
│     └── Input: "HPG nợ bao nhiêu?" -> Output: "Tỷ lệ nợ/Vốn chủ sở hữu và tổng nợ vay HPG 2023"
│
├── 2. Span: "hybrid_retrieval" (Duration: 340ms)
│     ├── Span: "dense_search_qdrant" (Duration: 120ms, top_k=10, score > 0.75)
│     └── Span: "sparse_search_bm25" (Duration: 45ms, top_k=10)
│
├── 3. Span: "cross_encoder_rerank" (Duration: 210ms)
│     └── Model: "BAAI/bge-reranker-large", In: 20 chunks -> Out: Top 3 Chunks chuẩn nhất
│
├── 4. Event: "pii_masking_triggered" (Timestamp: 10:15:32.450, Zero-duration)
│     └── Log: "Đã che giấu số tài khoản và thông tin cá nhân của người đại diện pháp luật"
│
├── 5. Generation: "llm_financial_synthesis" (Duration: 1.45s, TTFT: 320ms)
│     ├── Model: "llama-3.3-70b-versatile" (Provider: Groq)
│     ├── Prompt Template: "credit_risk_evaluation:v3" (Compiled with variables)
│     ├── Usage: { Prompt: 1,420 tokens, Completion: 380 tokens, Total: 1,800 tokens }
│     ├── Cost: $0.00124 USD
│     └── Output: "Dựa trên BCTC kiểm toán 2023 của Hòa Phát, tỷ lệ D/E đạt 0.78..."
│
└── 6. Score: [Evaluations & Feedback]
      ├── Score (Implicit/Offline): "ragas_faithfulness" = 0.96 (Đánh giá qua LLM-as-a-Judge)
      ├── Score (Explicit/Online): "user_feedback" = 1 (Người dùng bấm 👍 Thumbs Up)
      └── Score (Latency): "latency_p95_check" = PASSED (< 2.5s)
```

#### 5 Khái niệm Cốt lõi:
1. **Trace (Gốc - Root Request):** Đại diện cho một chu trình hoàn chỉnh từ lúc client gửi request đến lúc trả kết quả cuối cùng. Lưu trữ: `trace_id`, `name`, `user_id`, `session_id`, `tags`, `release`, `metadata`.
2. **Span (Công đoạn xử lý):** Đại diện cho một khoảng thời gian thực thi một công việc cụ thể không gọi LLM trực tiếp (ví dụ: query vector database, chạy reranking, gọi API bên thứ ba, xử lý logic Python).
3. **Generation (Cuộc gọi LLM chuyên biệt):** Span đặc biệt dùng riêng cho các lệnh gọi LLM. Bắt buộc ghi nhận: `model`, `model_parameters` (temperature, top_p), `prompt` (messages format), `completion`, `usage` (prompt_tokens, completion_tokens), và chi phí tiền tệ tự động tính toán.
4. **Event (Sự kiện tức thời):** Một điểm đánh dấu trong dòng thời gian không có thời lượng (duration = 0), dùng để ghi log nghiệp vụ quan trọng (ví dụ: phát hiện PII, kích hoạt fallback model, cache hit).
5. **Score (Chỉ số đánh giá chất lượng):** Lưu điểm số định lượng (`value: 0.0 - 1.0` hoặc `value: 1 - 5`) hoặc phân loại categorical (`value: "correct" / "hallucinated"`). Điểm này có thể gắn trực tiếp vào Trace hoặc vào từng Generation cụ thể.

---

### 6.3. Kiến trúc Triển khai: Self-Hosted vs. Langfuse Cloud

Doanh nghiệp có thể chọn 1 trong 2 hình thức:
* **Langfuse Cloud (Managed SaaS):** Tiện lợi, không cần bảo trì hạ tầng, free tier 50k events/tháng, tuân thủ SOC 2 Type II và GDPR.
* **Langfuse Self-Hosted (On-Premise / Private Cloud VPC):** Bắt buộc đối với các tổ chức Ngân hàng, Fintech, Y tế nơi dữ liệu tài chính không được phép ra internet công cộng.

#### File `docker-compose.yml` Chuẩn Production (Self-Hosted Enterprise)

Kiến trúc Langfuse v3+ sử dụng **PostgreSQL** (lưu entities, users, prompts) phối hợp với **ClickHouse** (cơ sở dữ liệu Columnar chuyên dụng để query hàng triệu Traces/Spans tốc độ cao):

```yaml
version: "3.8"

services:
  # 1. Ứng dụng Web UI & Ingestion API Server
  langfuse-server:
    image: ghcr.io/langfuse/langfuse:3
    depends_on:
      postgres:
        condition: service_healthy
      clickhouse:
        condition: service_healthy
      redis:
        condition: service_healthy
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://langfuse:langfuse_secure_pwd@postgres:5432/langfuse
      - NEXTAUTH_URL=http://localhost:3000
      - NEXTAUTH_SECRET=a_very_secret_key_change_me_in_production_min_32_chars
      - SALT=salt_for_encryption_keys_min_16_chars
      - CLICKHOUSE_URL=http://clickhouse:8123
      - CLICKHOUSE_USER=default
      - CLICKHOUSE_PASSWORD=clickhouse_secure_pwd
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - TELEMETRY_ENABLED=false
      - LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES=true
    restart: always

  # 2. Cơ sở dữ liệu Transactional (Users, Metadata, Prompt Registry)
  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=langfuse
      - POSTGRES_PASSWORD=langfuse_secure_pwd
      - POSTGRES_DB=langfuse
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: always

  # 3. Cơ sở dữ liệu Columnar Tốc độ cao (Traces, Spans, Generations Analytics)
  clickhouse:
    image: clickhouse/clickhouse-server:24.3-alpine
    environment:
      - CLICKHOUSE_DB=default
      - CLICKHOUSE_USER=default
      - CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1
      - CLICKHOUSE_PASSWORD=clickhouse_secure_pwd
    volumes:
      - clickhouse_data:/var/lib/clickhouse
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:8123/ping"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: always

  # 4. Hàng đợi đệm bất đồng bộ (Asynchronous Event Ingestion Queue)
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: always

volumes:
  postgres_data:
  clickhouse_data:
  redis_data:
```

#### Cấu hình Biến môi trường `.env` Ứng dụng Backend

```env
# URL đến máy chủ Langfuse (Cloud hoặc Self-hosted)
LANGFUSE_HOST="https://cloud.langfuse.com"   # Hoặc "http://localhost:3000" nếu chạy docker local
LANGFUSE_PUBLIC_KEY="pk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
LANGFUSE_SECRET_KEY="sk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

# Tối ưu hóa hiệu năng Ingestion (Không block API chính)
LANGFUSE_FLUSH_INTERVAL=0.5                  # Flush events mỗi 0.5s
LANGFUSE_MAX_RETRIES=3                       # Số lần thử lại nếu mạng chập chờn
LANGFUSE_THREADS=4                           # Số worker threads xử lý log ngầm
```

---

### 6.4. Các Phương thức Tích hợp Code Thực chiến (3 Integration Strategies)

#### Chiến lược 1: Dùng Python Native Decorator `@observe()` (Khuyến nghị cho Custom Code)
Decorator `@observe()` của Langfuse tự động liên kết các hàm lồng nhau thành cây Trace/Span mà không cần truyền biến context thủ công.

```python
import os
import time
from typing import List, Dict, Any
from langfuse.decorators import observe, langfuse_context
from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 1. Hàm con: Đo lường Vector Database Query (Tự động thành 1 Span)
@observe()
def retrieve_financial_context(query: str, top_k: int = 3) -> List[str]:
    # Ghi chú thêm metadata vào Span hiện tại
    langfuse_context.update_current_observation(
        metadata={"search_type": "hybrid", "top_k": top_k, "index": "fin_reports_2023"}
    )
    time.sleep(0.12)  # Giả lập thời gian truy vấn Qdrant
    return [
        "Trích BCTC 2023 HPG: Vốn chủ sở hữu đạt 102.000 tỷ VNĐ, Nợ phải trả đạt 78.000 tỷ VNĐ.",
        "Trích báo cáo thường niên: Tỷ lệ an toàn thanh khoản và hệ số D/E duy trì ở mức an toàn 0.76 lần."
    ]

# 2. Hàm con: Đo lường lệnh gọi LLM (Chỉ định rõ as_type="generation")
@observe(as_type="generation")
def call_llm_judge(prompt: str, model_name: str = "llama-3.3-70b-versatile") -> str:
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Bạn là chuyên gia thẩm định tín dụng tài chính cấp cao."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    output_text = response.choices[0].message.content
    
    # BẮT BUỘC: Cập nhật thông số Generation để Langfuse tự tính chi phí tiền USD
    langfuse_context.update_current_observation(
        model=model_name,
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        },
        metadata={"finish_reason": response.choices[0].finish_reason}
    )
    return output_text

# 3. Hàm gốc: Root Trace bao trọn toàn bộ quy trình
@observe()
def evaluate_company_risk(company_tax_id: str, query: str, user_id: str, session_id: str) -> Dict[str, Any]:
    # Định danh Trace cấp cao nhất
    langfuse_context.update_current_trace(
        name="Credit_Risk_Assessment_Workflow",
        user_id=user_id,
        session_id=session_id,
        tags=["risk_dept", "corporate_banking", "hpg"],
        metadata={"tax_id": company_tax_id, "app_version": "2.4.0"}
    )
    
    # Bước 1: Retrieval (Span)
    contexts = retrieve_financial_context(query=query)
    
    # Bước 2: Event (Đánh dấu logic quan trọng)
    langfuse_context.score_current_trace(
        name="retrieval_chunk_count",
        value=len(contexts),
        comment="Số lượng văn bản context cung cấp cho LLM"
    )
    
    # Bước 3: LLM Generation
    prompt_payload = f"Contexts:\n{chr(10).join(contexts)}\n\nCâu hỏi: {query}"
    final_analysis = call_llm_judge(prompt=prompt_payload)
    
    return {"analysis": final_analysis, "contexts_used": len(contexts)}
```

---

#### Chiến lược 2: Tích hợp Liền Mạch với LangChain & LangGraph (`CallbackHandler`)
Khi sử dụng **LangGraph** (xây dựng Agentic Workflow như trong dự án `FinRisk AI`), chỉ cần truyền `CallbackHandler` vào `RunnableConfig`. Toàn bộ các node, edge, prompt template, tool calls đều được log tự động.

```python
import os
from typing import TypedDict, List
from langfuse.callback import CallbackHandler
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Khởi tạo CallbackHandler đọc tự động cấu hình từ biến môi trường
langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

# 1. Định nghĩa State của Graph
class AgentState(TypedDict):
    question: str
    contexts: List[str]
    report: str

# 2. Định nghĩa các Nodes
def retrieve_node(state: AgentState):
    # Trích xuất dữ liệu giả lập từ Qdrant
    return {"contexts": ["Thông tư 41/2016/TT-NHNN quy định tỷ lệ CAR tối thiểu là 8%."]}

def synthesize_node(state: AgentState):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = f"Dựa vào tài liệu: {state['contexts']}, trả lời câu hỏi: {state['question']}"
    # Langfuse Handler sẽ tự động lồng generation này vào trong graph trace
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"report": response.content}

# 3. Dựng Workflow Graph
builder = StateGraph(AgentState)
builder.add_node("retriever", retrieve_node)
builder.add_node("synthesizer", synthesize_node)
builder.set_entry_point("retriever")
builder.add_edge("retriever", "synthesizer")
builder.add_edge("synthesizer", END)
graph = builder.compile()

# 4. Thực thi Graph với Langfuse Handler gắn kèm Metadata
async def run_finrisk_graph(user_query: str, analyst_id: str, session_id: str):
    config = {
        "callbacks": [langfuse_handler],
        "metadata": {
            "user_id": analyst_id,
            "session_id": session_id,
            "pipeline": "finrisk_v1_graph"
        },
        "tags": ["production", "vietnam_regulation"]
    }
    
    result = await graph.ainvoke(
        {"question": user_query, "contexts": [], "report": ""},
        config=config
    )
    return result["report"]
```

---

#### Chiến lược 3: Drop-in Replacement cho OpenAI SDK / Groq / LiteLLM
Nếu codebase đang sử dụng thư viện `openai` chuẩn, chỉ cần thay thế lệnh import:

```python
# Thay vì: from openai import OpenAI
from langfuse.openai import OpenAI
import os

# Tự động bắt 100% token, cost, prompt, parameters mà không cần chỉnh sửa bất kỳ dòng code nào khác
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Tóm tắt rủi ro thanh khoản ngân hàng"}],
    name="financial_risk_summary",           # Tên trace trên Langfuse
    user_id="analyst_nguyen_van_a",          # Định danh người dùng
    metadata={"department": "Risk_Auditing"} # Metadata tùy biến
)
print(completion.choices[0].message.content)
```

---

### 6.5. Đo lường Hiệu năng & Giám sát Chi phí (Latency, Tokens & Cost Tracking)

#### 1. Bóc tách Độ trễ Chuyên sâu (Deep Latency Profiling)
Người dùng cảm nhận tốc độ của chatbot RAG thông qua **Time to First Token (TTFT)** chứ không phải tổng thời gian hoàn thành (Total Generation Time).

```text
|<--------------------------- Total Latency (2.45s) -------------------------->|
|-- Retrieval (180ms) --|-- Rerank (210ms) --|-- TTFT (420ms) --|-- Stream (1.64s) --|
```
* **TTFT (Time to First Token):** Thời gian từ khi user bấm Enter đến khi chữ cái đầu tiên hiển thị trên màn hình. Ngưỡng chuẩn production: **$\text{TTFT} < 1.2\text{s}$**.
* **Bottleneck Diagnosis:**
  - Nếu `Retrieval Latency` > 800ms: Kiểm tra chỉ mục HNSW trên Qdrant/Milvus hoặc hạ `top_k`.
  - Nếu `Rerank Latency` > 500ms: Chuyển model Reranker từ CPU sang GPU hoặc giảm số chunks đầu vào từ 30 xuống 10.
  - Nếu `TTFT` > 2s: Prompt quá dài hoặc LLM Provider đang nghẽn hàng đợi (cần cân nhắc chuyển sang nhà cung cấp suy luận tốc độ cao như Groq, Cerebras hoặc vLLM local).

#### 2. Phân loại Token & Prompt Caching
Langfuse bóc tách chi tiết lượng token trong mỗi Generation:
* **Prompt Tokens (Input):** Số lượng token trong câu hỏi + System prompt + Ngữ cảnh tài liệu RAG.
* **Completion Tokens (Output):** Số lượng token do LLM sinh ra.
* **Cached Tokens (Prompt Caching):** Khi sử dụng các model như Anthropic Claude 3.5 Sonnet hoặc OpenAI GPT-4o, việc cache các đoạn tài liệu dài (như Thông tư 41 hay BCTC 100 trang) giúp giảm **50%–80% chi phí** và giảm **80% độ trễ**. Langfuse tự động hiển thị số lượng Cached Tokens này trên bảng dashboard.

#### 3. Cấu hình Bảng giá Tuỳ chỉnh (Custom Model Pricing)
Với các mô hình mã nguồn mở tự host (vLLM / Ollama) hoặc chạy qua các gateway nội bộ, có thể cấu hình bảng giá riêng trên Langfuse Settings:
* Ví dụ: Model `llama-3.3-70b-versatile` trên Groq:
  - Input: `$0.59 / 1M tokens`
  - Output: `$0.79 / 1M tokens`
* **Công thức tính tự động của Langfuse:**
  $$\text{Total Cost} = \left( \frac{\text{Input Tokens}}{1,000,000} \times P_{\text{input}} \right) + \left( \frac{\text{Output Tokens}}{1,000,000} \times P_{\text{output}} \right)$$

---

### 6.6. Vòng lặp Phản hồi & Đánh giá Trực tuyến (Feedback Loops & Online Evaluation)

Hệ thống Observability thực sự chỉ hoàn thiện khi có vòng lặp khép kín: **Giám sát $\rightarrow$ Phát hiện lỗi $\rightarrow$ Thu thập phản hồi $\rightarrow$ Cải tiến dữ liệu chuẩn**.

```
    ┌────────────────┐          ┌───────────────────┐          ┌────────────────────┐
    │  Người dùng    │  Gửi 👍  │ API Backend       │  Score   │ Langfuse Dashboard │
    │  hoặc LLM Judge├─────────►│ ghi nhận Feedback ├─────────►│ Lưu trữ & Alert    │
    └────────────────┘          └───────────────────┘          └─────────┬──────────┘
                                                                         │
                                                                         ▼
                                                               ┌────────────────────┐
                                                               │ Xuất Trace lỗi làm │
                                                               │ Golden Dataset mới │
                                                               └────────────────────┘
```

#### 1. Thu thập Phản hồi Người dùng Thực tế (Explicit User Feedback)
Khi người dùng bấm nút Thumbs Up / Down hoặc để lại nhận xét trên giao diện Web, Frontend gọi API gửi `trace_id` về server:

```python
from langfuse import Langfuse

langfuse = Langfuse()

def record_user_feedback(trace_id: str, is_helpful: bool, reason: str = ""):
    """
    Ghi nhận phản hồi của Chuyên viên thẩm định ngân hàng vào Trace tương ứng
    """
    langfuse.score(
        trace_id=trace_id,
        name="user_feedback",
        value=1.0 if is_helpful else 0.0,
        data_type="BOOLEAN",
        comment=reason
    )
```

#### 2. Tự động Đánh giá Trực tuyến trên Production (Online LLM-as-a-Judge)
Không thể chạy toàn bộ 100% request qua Ragas vì tốn chi phí. Thay vào đó, thiết lập **Sampling 5%–10%** các request trên production để chạy một Background Task chấm điểm `Faithfulness`:

```python
from fastapi import BackgroundTasks
from langfuse import Langfuse
from groq import Groq

langfuse = Langfuse()
groq_client = Groq()

async def evaluate_trace_faithfulness_task(trace_id: str, query: str, context: str, answer: str):
    """
    Background worker: Sử dụng mô hình Judge nhỏ để chấm điểm trung thực
    """
    prompt_judge = f"""
    Hãy đối chiếu Câu trả lời với Ngữ cảnh tài liệu và chấm điểm độ trung thực (Faithfulness) từ 0.0 đến 1.0.
    Chỉ trả về duy nhất 1 con số thập phân.
    Ngữ cảnh: {context}
    Câu trả lời: {answer}
    """
    judge_res = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt_judge}],
        temperature=0.0
    )
    
    score_val = float(judge_res.choices[0].message.content.strip())
    
    # Ghi điểm trực tiếp vào Trace trên Langfuse
    langfuse.score(
        trace_id=trace_id,
        name="online_faithfulness",
        value=score_val,
        comment="Đánh giá tự động ngầm bởi Llama-3.3-70B Judge"
    )
```

#### 3. Tinh lọc Dataset từ Production Traces (Curating Golden Dataset)
Trên giao diện Langfuse:
1. Tạo bộ lọc: Tìm tất cả các Trace có `user_feedback == 0` (Thumbs Down) HOẶC `online_faithfulness < 0.7`.
2. Bấm nút **"Add to Dataset"** $\rightarrow$ Lưu vào bộ `finrisk_hard_failures_dataset`.
3. Bộ dataset này được xuất tự động (Export) vào thư mục `tests/golden_dataset/` để làm bài test hồi quy (Regression Test) trong quy trình CI/CD.

---

### 6.7. Quản lý Prompt Tập trung & A/B Testing (Prompt Management & Versioning)

#### Rủi ro khi Hardcode Prompt trong Mã nguồn:
* Mọi chỉnh sửa câu chữ prompt đều phải tạo Pull Request, chờ review, build Docker image, và redeploy toàn bộ cụm server.
* Không thể phân quyền cho chuyên viên phân tích tài chính (Domain Experts / Prompt Engineers) tự tinh chỉnh prompt.
* Không thể rollback tức thì về phiên bản cũ khi prompt mới gây ảo giác nghiêm trọng.

#### Giải pháp: Langfuse Prompt Registry
Quản lý toàn bộ prompt trên giao diện web của Langfuse với cơ chế **Semantic Versioning**, gán nhãn môi trường (`production`, `staging`), và **In-Memory Caching (TTL)** để đảm bảo độ trễ gần như bằng 0.

```python
from langfuse import Langfuse
import os

langfuse = Langfuse()

def generate_risk_report(company_name: str, financial_metrics_json: str):
    # 1. Kéo Prompt được gắn nhãn 'production' từ Langfuse (Có cache in-memory 5 phút)
    # Không tạo thêm HTTP request mỗi lần invoke hàm!
    prompt_template = langfuse.get_prompt(
        name="financial_risk_report_prompt",
        label="production",
        cache_ttl_seconds=300
    )
    
    # 2. Biên dịch template với các biến số nghiệp vụ
    compiled_messages = prompt_template.compile(
        company_name=company_name,
        metrics=financial_metrics_json,
        current_year="2024"
    )
    
    # 3. Sử dụng với LLM và tự động liên kết Trace với Phiên bản Prompt đó
    from langfuse.openai import OpenAI
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=compiled_messages,
        # LIÊN KẾT PROMPT VỚI GENERATION:
        # Giúp Langfuse thống kê được prompt version nào cho điểm cao hơn trên Dashboard A/B testing
        langfuse_prompt=prompt_template
    )
    
    return response.choices[0].message.content
```

#### Quy trình A/B Testing Prompt trên Production:
1. **Prompt Version 1 (Control):** System Prompt truyền thống, liệt kê tiêu chí thẩm định.
2. **Prompt Version 2 (Treatment):** System Prompt bổ sung kỹ thuật Chain-of-Thought và ép trích dẫn điều khoản luật.
3. Trong code backend: Dùng hàm băm `hash(user_id) % 2` để chia 50% traffic gọi `label="v1"` và 50% traffic gọi `label="v2"`.
4. Trên Langfuse Dashboard: So sánh trực tiếp 2 phiên bản qua biểu đồ:
   - Tỷ lệ `user_feedback` tích cực (Win Rate).
   - Tỷ lệ vi phạm ảo giác (`faithfulness`).
   - Lượng token tiêu thụ trung bình.

---

### 6.8. File Mẫu Thực Chiến Hoàn Chỉnh: `finrisk_observability_pipeline.py`

File mã nguồn Python hoàn chỉnh dưới đây minh họa việc tích hợp toàn diện: **FastAPI Streaming + Langfuse Decorator + Background Scoring + HTTP Feedback Endpoint + Graceful Shutdown**:

```python
"""
finrisk_observability_pipeline.py
Hệ thống Thẩm định Tín dụng Tự động với Giám sát Toàn diện Langfuse
"""

import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from groq import AsyncGroq

# 1. Khởi tạo Clients
langfuse = Langfuse()
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# 2. Quản lý vòng đời FastAPI (Graceful Flush Langfuse khi tắt App)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 FinRisk AI Observability Service is starting...")
    yield
    print("🛑 Service shutting down. Flushing all remaining Langfuse telemetry events...")
    langfuse.flush()

app = FastAPI(title="FinRisk AI Enterprise Observability", lifespan=lifespan)

# 3. Pydantic Schemas
class RiskAnalysisRequest(BaseModel):
    company_name: str = Field(..., example="Tập đoàn Hòa Phát")
    tax_code: str = Field(..., example="0900189284")
    query: str = Field(..., example="Đánh giá khả năng thanh toán nợ vay ngắn hạn năm 2023.")
    session_id: str = Field(default="sess_default")

class UserFeedbackRequest(BaseModel):
    trace_id: str = Field(..., example="tr-823b12ef-...")
    is_positive: bool = Field(..., description="True nếu hài lòng (Thumbs up), False nếu phàn nàn (Thumbs down)")
    comment: str = Field(default="", example="Số liệu chi phí tài chính rất chuẩn xác.")

# 4. Các bước xử lý nghiệp vụ được bọc bởi @observe

@observe()
async def retrieve_financial_data(tax_code: str, query: str):
    """Giả lập tìm kiếm Hybrid Search trên Vector DB Qdrant"""
    langfuse_context.update_current_observation(
        metadata={"tax_code": tax_code, "retrieval_strategy": "dense_sparse_hybrid"}
    )
    await asyncio.sleep(0.15) # Giả lập I/O database latency
    return (
        f"Dữ liệu BCTC {tax_code}: Nợ ngắn hạn: 52.300 tỷ VNĐ. "
        f"Tài sản ngắn hạn: 64.100 tỷ VNĐ. Tỷ số thanh toán hiện hành: 1.23 lần."
    )

@observe()
async def rerank_and_filter(raw_context: str):
    """Giả lập bước Reranking bằng Cross-Encoder"""
    await asyncio.sleep(0.08)
    return raw_context

# 5. Hàm sinh phản hồi Streaming tích hợp quan sát Generation
async def stream_generator(prompt: str, trace_id: str) -> AsyncGenerator[str, None]:
    """Stream token về Client đồng thời ghi nhận vào Langfuse Generation"""
    
    # Kéo prompt đã quản lý trên Langfuse
    try:
        managed_prompt = langfuse.get_prompt("credit_analyst_agent", label="production")
        system_instruction = managed_prompt.compile()
    except Exception:
        system_instruction = "Bạn là chuyên gia phân tích tín dụng ngân hàng thận trọng và chuẩn xác."

    stream = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        stream=True
    )
    
    collected_chunks = []
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            collected_chunks.append(delta)
            yield delta

    full_output = "".join(collected_chunks)
    
    # Flush thủ công event ngầm
    langfuse.flush()

# 6. Endpoint Chính: Phân tích rủi ro có Tracing & Streaming
@app.post("/api/v1/analyze-risk")
@observe()
async def analyze_company_risk(
    req: RiskAnalysisRequest,
    x_user_id: str = Header(default="analyst_guest")
):
    # Cấu hình Trace gốc
    trace_id = langfuse_context.get_current_trace_id()
    langfuse_context.update_current_trace(
        name="Credit_Risk_Assessment_API",
        user_id=x_user_id,
        session_id=req.session_id,
        tags=["credit_risk", req.company_name, "streaming"],
        metadata={"tax_code": req.tax_code}
    )
    
    # Bước 1: Trích xuất dữ liệu
    context = await retrieve_financial_data(tax_code=req.tax_code, query=req.query)
    
    # Bước 2: Rerank
    filtered_context = await rerank_and_filter(raw_context=context)
    
    full_prompt = (
        f"Doanh nghiệp: {req.company_name} (MST: {req.tax_code})\n"
        f"Tài liệu đối chiếu:\n{filtered_context}\n\n"
        f"Yêu cầu thẩm định: {req.query}"
    )
    
    # Trả về StreamingResponse kèm trace_id trong Header HTTP để Frontend dễ lưu lại
    return StreamingResponse(
        stream_generator(prompt=full_prompt, trace_id=trace_id),
        media_type="text/event-stream",
        headers={"X-Langfuse-Trace-Id": str(trace_id)}
    )

# 7. Endpoint Tiếp nhận Phản hồi Người dùng (Human-in-the-loop Feedback)
@app.post("/api/v1/feedback")
async def submit_feedback(fb: UserFeedbackRequest):
    """Nhận Thumbs Up / Down từ giao diện người dùng và gán vào Trace tương ứng"""
    try:
        langfuse.score(
            trace_id=fb.trace_id,
            name="user_feedback",
            value=1.0 if fb.is_positive else 0.0,
            comment=fb.comment
        )
        return {"status": "success", "message": "Feedback recorded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

### 6.9. Best Practices & Các Lỗi Thường Gặp Cần Tránh Khi Vận Hành

#### 1. Nguyên tắc Không Block Luồng Chính (Non-blocking Asynchronous Telemetry)
* **Tuyệt đối không:** Gọi các hàm đồng bộ blocking như `langfuse.flush()` bên trong từng lượt request API của khách hàng. Việc này sẽ làm tăng thời gian chờ của người dùng thêm từ $100\text{ms} - 500\text{ms}$ chỉ để đợi ghi log.
* **Đúng chuẩn:** Langfuse SDK sử dụng hàng đợi bất đồng bộ chạy nền (`background worker thread`). Dữ liệu được gom thành batch và gửi ngầm định kỳ (mỗi $0.5\text{s}$). Chỉ gọi `langfuse.flush()` duy nhất một lần khi ứng dụng chuẩn bị shutdown (tín hiệu `SIGTERM` / `SIGINT`).

#### 2. Xử lý Thất lạc Context Trong Môi trường Bất đồng bộ (`asyncio`)
Khi sử dụng `asyncio.gather()` hoặc chạy nhiều task song song, context của `langfuse_context` có thể bị phân mảnh nếu không cẩn thận.
* **Cách khắc phục:** Truyền tường minh `trace_id` hoặc truyền `CallbackHandler` trực tiếp vào từng instance `ainvoke()` thay vì dựa hoàn toàn vào biến toàn cục.

#### 3. Tuân thủ Bảo mật Dữ liệu Ngân hàng (Data Privacy & PII Masking)
Theo quy định an toàn thông tin ngân hàng:
* Không bao giờ gửi dữ liệu chưa che giấu như: Số chứng minh nhân dân/CCCD, Số tài khoản ngân hàng, Mật khẩu, Số dư cá nhân lên cloud công cộng.
* Sử dụng bộ lọc tiền xử lý Regex Masking hoặc thư viện **Microsoft Presidio** để biến đổi dữ liệu trước khi đẩy vào Langfuse:
  ```python
  # Ví dụ Masking: "001100234567" -> "0011****4567"
  def mask_account_number(text: str) -> str:
      import re
      return re.sub(r'(\d{4})\d{4,6}(\d{4})', r'\1****\2', text)
  ```

#### 4. Chiến lược Lấy mẫu (Sampling Strategy) Khi Tải Cao (High QPS)
Nếu hệ thống phục vụ hàng triệu request mỗi ngày:
* Việc log 100% trace có thể làm quá tải hạ tầng ClickHouse và tiêu tốn nhiều chi phí lưu trữ.
* Cấu hình tỷ lệ lấy mẫu: **Trace 100% các request có phát sinh lỗi hoặc request từ người dùng nội bộ/VIP**, nhưng chỉ **lấy mẫu ngẫu nhiên 5%–10% các request thông thường** để theo dõi chỉ số thống kê P95/P99.


---

## 7. CI/CD WORKFLOW (GITHUB ACTIONS)

Tự động chạy đánh giá và xuất báo cáo markdown khi có pull request vào nhánh `main`:

Tạo file `.github/workflows/evaluation_ci.yml`:
```yaml
name: RAG Evaluation & Quality Gate CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  rag-evaluation:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Source Code
      uses: actions/checkout@v3

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
        cache: 'pip'

    - name: Install Evaluation Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install ragas deepeval pytest pandas langfuse

    - name: Run DeepEval Quality Gate
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
      run: |
        # Chạy test suite, trả mã lỗi khác 0 nếu Faithfulness < 0.90
        pytest tests/test_ci_deepeval.py -v --junitxml=report_eval.xml

    - name: Upload Test Report
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: rag-evaluation-results
        path: report_eval.xml
```

---

## 8. CHECKLIST THỰC HÀNH TỪNG TUẦN (PHASE 4 ACTION PLAN)

- [ ] **Tuần 10.1: Golden Dataset & RAGAS Setup**
  - [ ] Thu thập 20–50 cặp câu hỏi - câu trả lời chuẩn (Ground Truth) từ BCTC.
  - [ ] Viết module [eval_metrics.py](file:///d:/ai_foundation/LLM/phase_4/eval_metrics.py) chạy thử 4 chỉ số Ragas.
  - [ ] Phân tích điểm chuẩn (Baseline scores) trước khi tối ưu RAG.
- [ ] **Tuần 10.2: G-Eval & DeepEval Unit Testing**
  - [ ] Viết bộ test `assert_test` kiểm tra độ trung thực và tính chuẩn xác số liệu.
  - [ ] Cấu hình chạy test tự động trong thư mục `tests/`.
- [ ] **Tuần 11.1: Langfuse Observability**
  - [ ] Tạo tài khoản Langfuse Cloud hoặc dựng container Docker nội bộ.
  - [ ] Tích hợp `@observe()` và `CallbackHandler` vào FastAPI và LangGraph agents.
  - [ ] Kiểm tra dashboard: Latency P95, Token usage, Estimated cost USD.
- [ ] **Tuần 11.2: Automation & CI/CD Pipeline**
  - [ ] Viết workflow `.github/workflows/evaluation_ci.yml`.
  - [ ] Thiết lập ngưỡng chặn (Quality Gate): Fail build nếu `Faithfulness < 0.90`.
  - [ ] Xuất báo cáo markdown `eval_report_final.md` đính kèm vào portfolio dự án.