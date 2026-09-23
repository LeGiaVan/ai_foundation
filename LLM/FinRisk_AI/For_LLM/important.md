# 📋 HANDOFF CONTEXT — FinRisk AI (Plan B — Evidence-Aware Financial Analysis)
> Đọc file này để hiểu toàn bộ ngữ cảnh dự án trước khi tiếp tục code.

---

## 1. Dự án là gì?

**FinRisk AI Plan B** — Hệ thống **tự động phân tích BCTC và tính Financial Ratios có Provenance**.

Bài toán cốt lõi:
> "Nhận PDF BCTC → Trích xuất Financial Facts (có truy xuất trang/bảng/dòng) → Tính 13 Financial Ratios deterministic → Sinh báo cáo phân tích có citation đầy đủ."

**Tại sao chuyển từ FinRisk AI v3?**
FinRisk AI v3 (48 sections, 8+ agents) là kiến trúc enterprise cần team 5–10 người. Plan B giữ **100% code đã build** và thêm 3 layer mới có chiều sâu kỹ thuật thực sự — vừa sức xây dựng một mình trong 8–10 tuần.

---

## 2. Tech Stack

| Tầng | Công nghệ | Ghi chú |
|------|-----------|---------| 
| **LLM** (dev) | Groq — `openai/gpt-oss-120b` | Miễn phí, nhanh, dùng cho Fact Extraction |
| **LLM** (prod) | Claude Haiku / GPT-4o-mini | Đổi `LLM_PROVIDER` trong `.env` |
| **Structured Output** | Pydantic schemas | LLM trả về typed `FinancialFact` list |
| **Multi-Agent** | LangGraph `StateGraph` | State Machine, HITL, Checkpointer |
| **Config** | pydantic-settings | Đọc từ `.env`, singleton `get_settings()` |
| **Vector DB** | Qdrant (Local Docker) | Hybrid Search — Dense (BGE-M3) + Sparse (BM25) |
| **Embedding** | `sentence-transformers` BAAI/bge-m3 | Dense 1024d |
| **Reranker** | `sentence-transformers` BGE-Reranker-v2-m3 | Cross-Encoder ~1.1GB |
| **External Search** | Tavily Search API | Character (5C) — Fallback mode nếu không có key |
| **PDF Parsing** | `pdfplumber` | Table → Markdown, text extraction |
| **Evaluation** | Ragas + custom metrics | Extraction F1, Numeric Accuracy, Provenance Accuracy |

---

## 3. Cấu trúc thư mục (Hiện tại + Kế hoạch Plan B)

```
FinRisk AI/
├── .env.example              ← Copy thành .env, điền GROQ_API_KEY
├── .gitignore
├── pyproject.toml            ← pip install -e ".[dev]"
├── main.py                   ← Entry point CLI
├── scripts/
│   ├── index_sample.py       ← Index PDF vào Qdrant (--demo)
│   └── analyze_bctc.py       ← [KẾ HOẠCH] CLI: PDF → Financial Analysis Report
├── tests/
│   ├── test_graph.py         ← Multi-Agent, tools, supervisor (12 tests PASS)
│   ├── test_retriever.py     ← Chunking, PDF table, hybrid search (10 tests PASS)
│   ├── test_phase3_tools.py  ← Pro-forma DSCR, LTV, DoA, Covenants (45 tests PASS)
│   └── test_extractor.py     ← [KẾ HOẠCH] FactStore, Provenance, Fact Extractor
├── eval/
│   └── financial_eval.py     ← [KẾ HOẠCH] Evaluation suite + ablation study
├── For_LLM/
│   ├── important.md          ← FILE NÀY
│   └── new.md                ← Proposal v3 cũ (tham khảo)
└── src/
    ├── config.py             ← Settings + ngưỡng nghiệp vụ + Qdrant + LTV + Tavily
    ├── extractor/            ← [KẾ HOẠCH] Layer mới — Financial Fact Extraction
    │   ├── fact_extractor.py ← LLM structured output → list[FinancialFact]
    │   ├── fact_store.py     ← In-memory store, lookup theo (concept, period)
    │   └── provenance.py     ← Build trace: Metric → Facts → Page/Table/Row
    ├── tools/
    │   └── financial_tools.py  ← 7 tools hiện tại → [KẾ HOẠCH] 13 tools
    ├── retriever/            ← ✅ Hoàn tất Phase 2
    │   ├── qdrant_client.py
    │   ├── embedder.py
    │   ├── indexer.py        ← PDF parse + Recursive Splitter + Table Preservation
    │   └── searcher.py       ← Hybrid Search RRF + Reranker + Parent Dedup
    └── agents/
        ├── state.py          ← ✅ FinRiskState + [KẾ HOẠCH] FinancialFact + ProvTrace
        ├── supervisor.py     ← Routing deterministic
        ├── rag_agent.py      ← Qdrant search + fallback
        ├── math_agent.py     ← ReAct loop + [KẾ HOẠCH] auto-populate từ Fact Store
        ├── risk_agent.py     ← DoA 4 cấp + Credit Covenants + Pro-forma scoring
        └── graph.py          ← Assembly + [KẾ HOẠCH] nâng cấp report_node
```

---

## 4. Pipeline hoàn chỉnh (Plan B)

```
[USER] Upload PDF BCTC + thông tin hồ sơ vay
    │
    ▼
[FACT EXTRACTOR] — MỚI (src/extractor/fact_extractor.py)
    │  LLM Structured Output: list[FinancialFact]
    │  Mỗi fact: {concept, value, period, page, table_id, row_label, confidence}
    │
    ▼
[FACT STORE] — MỚI (src/extractor/fact_store.py)
    │  In-memory dict: key = (concept, period)
    │
    ▼
[SUPERVISOR] → routing deterministic:
    │
    ├── Chưa có financial_metrics? → [MATH AGENT]
    │       Auto-populate từ Fact Store (không cần nhập tay raw_financials)
    │       LLM bind_tools(FINANCIAL_TOOLS) → gọi 13 Python tools
    │       Tool tính: Z-Score, DSCR, Pro-forma DSCR, D/E, Quick Ratio,
    │                  Current Ratio, Gross Margin, EBITDA Margin, ROA, ROE,
    │                  Interest Coverage, LTV, Working Capital
    │       Sau mỗi tool → gọi build_provenance() → state.provenance_traces
    │       search_external_risk_news() → external_news_context
    │       Kết quả → state.financial_metrics
    │
    ├── Chưa có retrieved_context? → [RAG AGENT]
    │       Hybrid Search (Dense + Sparse + RRF) trên Qdrant
    │       Cross-Encoder Rerank → top parent chunks (narrative context)
    │       Kết quả → state.retrieved_context (có page metadata)
    │
    ├── Chưa có risk_assessment? → [RISK AGENT]
    │       Rule-based score từ metrics (kể cả Pro-forma DSCR + LTV)
    │       LLM phân tích 5C + external_news_context
    │       → state.risk_assessment (DoA recommendation + credit_covenants)
    │       → Nếu score ≥ 40 hoặc khoản vay lớn: requires_human_approval = True
    │
    ├── Cần HITL? → [HITL NODE]
    │       interrupt() → pipeline dừng
    │       Human nhập "APPROVED" / "REJECTED"
    │       → resume → [supervisor]
    │
    └── Xong → [REPORT NODE] (nâng cấp)
            LLM sinh báo cáo với citations đầy đủ:
            - Mỗi metric kèm [Trang X, Bảng Y, Dòng Z]
            - Section "Provenance Trail"
            - Section "Data Gaps"
            → state.final_report → END
```

---

## 5. Ngưỡng nghiệp vụ (config.py) — Giữ nguyên

### Scoring Model (Rule-based)

| Chỉ số | Ngưỡng | Điểm rủi ro |
|--------|--------|-------------|
| **Pro-forma DSCR** | < 1.0 | **+45** |
| Pro-forma DSCR | 1.0 – 1.25 | +20 |
| DSCR lịch sử | < 1.0 | +40 |
| DSCR lịch sử | 1.0 – 1.25 | +15 |
| Altman Z | < 1.81 | +30 |
| Altman Z | 1.81 – 2.99 | +10 |
| LTV | BREACH | +25 |
| LTV | MARGINAL | +10 |
| D/E | > 3.0 | +20 |
| D/E | 2.0 – 3.0 | +8 |
| Quick Ratio | < 0.8 | +10 |
| **Score cap** | — | **100 max** |

### LTV Thresholds

| Loại TSBĐ | Trần tối đa |
|-----------|-------------|
| Bất động sản | 70% |
| Máy móc/thiết bị | 50% |
| Chứng khoán niêm yết | 60% |
| Tài sản khác | 40% |

### Ma trận DoA

| Risk Score | Khoản vay | Recommendation | HITL? |
|------------|-----------|----------------|-------|
| < 40 | Bất kỳ | `FAST_TRACK_REVIEW` | ❌ |
| 40 – 69 | < 10 tỷ | `STANDARD_AUDIT` | ✅ Trưởng phòng |
| 40 – 69 | ≥ 10 tỷ | `CREDIT_COMMITTEE` | ✅ Hội đồng |
| < 40 | ≥ 10 tỷ | `CREDIT_COMMITTEE` | ✅ Hội đồng |
| ≥ 70 | Bất kỳ | `DECLINE_RECOMMENDED` | ✅ |

---

## 6. Cách chạy ngay (Hiện tại)

```bash
# 1. Cài dependencies
pip install -e .

# 2. Khởi động Qdrant (Docker)
docker run -d -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  --name qdrant --restart unless-stopped qdrant/qdrant

# 3. Tạo .env
copy .env.example .env
# Điền GROQ_API_KEY=gsk_xxxx

# 4. Index dữ liệu mẫu
python scripts/index_sample.py --demo

# 5. Chạy pipeline (Phase 1-3 đã hoàn chỉnh)
python main.py

# 6. Tests unit (không cần Qdrant)
pytest tests/ -v -m "not integration"

# 7. Tests integration (cần Qdrant Docker)
pytest tests/ -v -m integration
```

---

## 7. Quy ước code

- **Không hardcode API key** — luôn qua `get_settings()`
- **LLM không tự tính toán số học** — mọi phép tính phải qua Python Tool
- **Thêm tool mới** — tạo `@tool` trong `financial_tools.py` + thêm vào `FINANCIAL_TOOLS` list
- **Thêm Agent mới** — tạo `node_function(state, llm)` → đăng ký trong `graph.py`
- **Mỗi FinancialFact phải có provenance** — `page`, `table_id`, `row_label` bắt buộc
- Tìm comment `# PLAN B:` và `# KẾ HOẠCH:` để biết nơi cần mở rộng

---

## 8. Tài liệu tham chiếu

| File | Mục đích |
|------|---------| 
| `MD/project_proposal_fintech.md` | Proposal Plan B đầy đủ (v4.0) — lộ trình 10 tuần |
| `MD/business_domain_guide.md` | Nghiệp vụ thẩm định tín dụng + Khung 5C |
| `MD/rag_pipeline_cheatsheet.md` | Dense/Sparse, RRF, Cross-Encoder, Parent-Doc |
| `MD/production_devops_cheatsheet.md` | DevOps (dùng cho Phase C sau này) |
| `For_LLM/new.md` | FinRisk AI v3 proposal — tham khảo bài toán đầy đủ |
| `../phase_4/retriever_strategies_cheatsheet.md` | Chiến thuật RAG nâng cao |

---

## 9. Lịch sử cập nhật & Trạng thái hiện tại

### ✅ Đã hoàn thành (Phase 1 — Tuần 1)
1. **Multi-Agent Core**:
   - Supervisor Router → Math Agent → Risk Agent → HITL → Report.
   - 4 tools: `altman_z_score`, `dscr`, `debt_to_equity`, `quick_ratio`.
   - HITL interrupt khi risk_score ≥ 40.
   - **12/12 PASS** `pytest tests/test_graph.py`.

### ✅ Đã hoàn thành (Phase 2 — Tuần 2–3)
2. **Deep RAG Engine**:
   - Qdrant Hybrid Search (Dense BGE-M3 + Sparse BM25 + RRF). API chuẩn `qdrant-client 1.19`.
   - Recursive Separator Splitter (`\n\n` → `\n` → `. ` → `; ` → `, ` → ` `).
   - Markdown Table Preservation — bảng tài chính không bao giờ bị cắt vụn.
   - Chunk Overlap 30 tokens — child chunks liền kề gối đầu nhau.
   - BGE-Reranker-v2-m3 Cross-Encoder + Parent Deduplication.
   - **10/10 PASS** `pytest tests/test_retriever.py`.

### ✅ Đã hoàn thành (Phase 3 — Tuần 4)
3. **Banking-Grade Logic & DoA**:
   - `LoanApplication` TypedDict, `pro_forma_dscr` field, `external_news_context`, `credit_covenants`.
   - 7 tools: thêm `calculate_pro_forma_dscr`, `calculate_ltv`, `search_external_risk_news`.
   - DoA 4 cấp: `_map_to_doa()` + khoản vay ≥ 10 tỷ tự động → `CREDIT_COMMITTEE`.
   - Credit Covenants tự động: `_generate_default_covenants()`.
   - **45/45 PASS** `pytest tests/test_phase3_tools.py`.
   - **Tổng: 57/57 PASS** `pytest tests/`.

---

### 🚀 Bước tiếp theo (Plan B — Phase 4, Tuần 5–6)

#### Bước 1: Thêm `FinancialFact` + `ProvTrace` vào State
File: `src/agents/state.py`
```python
class FinancialFact(TypedDict):
    fact_id: str
    concept: str          # "total_assets", "revenue", ...
    value: float
    currency: str         # "VND"
    period: str           # "2024", "Q2/2024"
    page: int
    table_id: str | None
    row_label: str | None
    confidence: float
    source_text: str

class ProvTrace(TypedDict):
    metric_name: str
    formula: str
    inputs: list[FinancialFact]
    result: float
```

Thêm vào `FinRiskState`:
```python
financial_facts: list[FinancialFact]
provenance_traces: list[ProvTrace]
```

#### Bước 2: Viết `fact_extractor.py`
File: `src/extractor/fact_extractor.py`
- Nhận `pages: list[dict]` từ `parse_pdf()`
- Dùng LLM với structured output (Pydantic) → `list[FinancialFact]`
- Financial Ontology: map alias tiếng Việt → canonical concept

#### Bước 3: Viết `fact_store.py`
File: `src/extractor/fact_store.py`
```python
def get_fact(concept: str, period: str) -> FinancialFact | None: ...
def get_all_periods() -> list[str]: ...
```

#### Bước 4: Viết `provenance.py`
File: `src/extractor/provenance.py`
```python
def build_provenance(metric_name, formula, input_concepts, fact_store) -> ProvTrace: ...
```

#### Bước 5: Kết nối `loan_application` vào Math Agent
File: `src/agents/math_agent.py`
```python
# Trong math_agent_node():
if loan_app := state.get("loan_application"):
    # Auto-trigger pro_forma_dscr và ltv tools
```

#### Bước 6: Cập nhật `main.py` demo
Thêm `loan_application` vào initial state mẫu.

#### Bước 7: Nâng cấp `report_node` trong `graph.py`
```python
# Thêm vào prompt:
f"Provenance: {state.get('provenance_traces', [])}"
f"Credit Covenants: {risk.get('credit_covenants', [])}"
```

---

## 10. Research Ablation (Tuần 9–10)

So sánh để chứng minh technical contribution:

| Baseline | Chunking | Calculation | Expected |
|----------|----------|-------------|---------|
| S1 | Fixed-size (500 tokens) | LLM tự tính | Baseline thấp nhất |
| S2 | Recursive | LLM tự tính | Tốt hơn S1 |
| S3 | Table-preserving | LLM tự tính | Tốt cho bảng số |
| S4 | Parent-Child RAG | LLM tự tính | Tốt cho context |
| **S5** | **Evidence-Aware + Fact Extraction** | **Deterministic Python** | **Best overall** |

**Mục tiêu chứng minh**: Numeric accuracy S5 >> S1-S4. LLM không được tính số vì hallucinate.
