# 📋 HANDOFF CONTEXT — FinRisk AI (Phase 2 Complete)
> Đọc file này để hiểu toàn bộ ngữ cảnh dự án trước khi tiếp tục code.

---

## 1. Dự án là gì?

**FinRisk AI** — Hệ thống Multi-Agent tự động phân tích Báo cáo Tài chính (BCTC) và Thẩm định Rủi ro Tín dụng doanh nghiệp.

Bài toán: Thay thế 2-3 ngày làm thủ công của chuyên viên tín dụng bằng pipeline AI tự động dưới 5 phút. Áp dụng khung nghiệp vụ chuẩn quốc tế **5C (Capacity, Capital, Conditions, Character, Collateral)**.

---

## 2. Tech Stack

| Tầng | Công nghệ | Ghi chú |
|------|-----------|---------|
| **LLM** (học) | Groq — `openai/gpt-oss-120b` | Miễn phí, nhanh |
| **LLM** (production) | Claude Haiku / GPT-4o-mini | Đổi `LLM_PROVIDER` trong `.env` |
| **Multi-Agent** | LangGraph `StateGraph` | State Machine, HITL, Checkpointer |
| **Config** | pydantic-settings | Đọc từ `.env`, singleton `get_settings()` |
| **Vector DB** | Qdrant (Local Docker) | Hybrid Search — Dense (BGE-M3) + Sparse (BM25) |
| **Embedding** | `fastembed` BAAI/bge-m3 | Dense 1024d, auto-download ~600MB |
| **Reranker** | `sentence-transformers` BGE-Reranker-v2-m3 | Cross-Encoder ~1.1GB |
| **Observability** | Langfuse self-hosted (Phase 3) | Chưa cài |
| **API** | FastAPI (Phase 4) | Chưa cài |
| **Deploy** | Docker Compose + Nginx + VPS (Phase 4) | Chưa cài |

---

## 3. Cấu trúc thư mục (Phase 1 — đã có)

```
FinRisk AI/
├── .env.example              ← Copy thành .env, điền GROQ_API_KEY
├── .gitignore
├── pyproject.toml            ← pip install -e ".[dev]"
├── main.py                   ← Entry point CLI, chạy thử pipeline
├── scripts/
│   └── index_sample.py       ← Index PDF vào Qdrant (--demo để test nhanh)
├── For_LLM/
│   └── important.md          ← FILE NÀY
└── src/
    ├── config.py             ← Tất cả settings + ngưỡng nghiệp vụ + Qdrant config
    ├── tools/
    │   └── financial_tools.py  ← 4 @tool: Z-Score, DSCR, D/E, Quick Ratio
    ├── retriever/            ← MỚI (Phase 2)
    │   ├── qdrant_client.py  ← Singleton Qdrant connection + ensure_collection()
    │   ├── embedder.py       ← Dense (BGE-M3) + Sparse (BM25) wrapper
    │   ├── indexer.py        ← PDF parse (pdfplumber) + chunk + upsert
    │   └── searcher.py       ← Hybrid Search RRF + Cross-Encoder rerank
    └── agents/
        ├── state.py          ← FinRiskState (TypedDict)
        ├── supervisor.py     ← Routing node (deterministic)
        ├── rag_agent.py      ← ✅ Phase 2: Qdrant search + fallback to raw
        ├── math_agent.py     ← ReAct loop: LLM → Tool → Observe
        ├── risk_agent.py     ← Rule-based score + LLM + HITL trigger
        └── graph.py          ← Assembly: Nodes, Edges, Checkpointer
```

---

## 4. Luồng hoạt động Pipeline

```
User nhập câu hỏi + raw_financials (số liệu BCTC)
    │
    ▼
[supervisor] → routing deterministic:
    ├── Chưa có financial_metrics? → [math_agent]
    │       LLM bind_tools(FINANCIAL_TOOLS) → gọi Python tools
    │       Tool tính Z-Score, DSCR, D/E, Quick Ratio
    │       Kết quả → state.financial_metrics
    │
    ├── Chưa có risk_assessment? → [risk_agent]
    │       Rule-based score từ metrics
    │       + LLM phân tích định tính context
    │       → state.risk_assessment + risk_score (0-100)
    │       → Nếu score >= 40: requires_human_approval = True
    │
    ├── Cần HITL? → [hitl_node]
    │       interrupt() → pipeline dừng
    │       Human nhập "APPROVED" / "REJECTED"
    │       → resume → [supervisor]
    │
    └── Xong → [report_node] → LLM viết báo cáo tổng hợp → END
```

---

## 5. Ngưỡng nghiệp vụ quan trọng (config.py)

| Chỉ số | Ngưỡng | Ý nghĩa | Điểm rủi ro |
|--------|--------|---------|------------|
| DSCR | < 1.0 | Không đủ tiền trả nợ | +40 |
| Altman Z | < 1.81 | Nguy cơ phá sản (DISTRESS) | +30 |
| D/E | > 3.0 | Đòn bẩy quá cao | +20 |
| Quick Ratio | < 0.8 | Thanh khoản yếu | +10 |
| Risk Score | ≥ 70 | → REJECT + HITL | — |
| Risk Score | ≥ 40 | → REVIEW + HITL | — |

---

## 6. Cách chạy ngay

```bash
# 1. Cài dependencies
pip install -e .

# 2. Khởi động Qdrant (Docker)
docker run -d -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  --name qdrant --restart unless-stopped qdrant/qdrant

# 3. Tạo .env
copy .env.example .env
# Điền GROQ_API_KEY=gsk_xxxx (lấy tại console.groq.com — miễn phí)

# 4. Index dữ liệu mẫu vào Qdrant
python scripts/index_sample.py --demo

# 5. Chạy pipeline đầy đủ
python main.py

# 6. Chạy tests (unit tests không cần Qdrant)
pytest tests/ -v -m "not integration"

# 7. Chạy integration tests (cần Qdrant Docker)
pytest tests/ -v -m integration
```

---

## 7. Điểm mở rộng theo Phase

| Phase | Việc cần làm | File cần sửa |
|-------|-------------|--------------|
| **Phase 3** | Thêm `@observe()` decorator Langfuse vào mỗi agent node | `src/agents/*.py` |
| **Phase 3** | Chuyển System Prompts từ code vào Langfuse Prompt Registry | `src/agents/supervisor.py`, `math_agent.py`, `risk_agent.py` |
| **Phase 3** | Ragas evaluation: Golden Testset 50 câu, đo Faithfulness/Recall | File mới: `eval/ragas_eval.py` |
| **Phase 4** | Viết `src/api/main.py` FastAPI, bọc `get_graph()` thành endpoints | File mới |
| **Phase 4** | Đổi `MemorySaver` → `AsyncPostgresSaver` trong `graph.py` | `src/agents/graph.py` — tìm `# Phase 4:` |
| **Phase 4** | Deploy: Docker Compose, Nginx, SSL, GitHub Actions CI/CD | Xem `production_devops_cheatsheet.md` |

---

## 8. Quy ước code quan trọng

- **Không hardcode API key** — luôn qua `get_settings()`
- **Thêm tool mới** — chỉ cần tạo `@tool` trong `financial_tools.py` + thêm vào `FINANCIAL_TOOLS` list
- **Thêm Agent mới** — tạo `node_function(state, llm)` → đăng ký trong `graph.py`
- **LLM không tự tính toán** — mọi phép tính phải qua Python Tool
- Tìm comment `Scale-up hint:` và `# PHASE N:` để biết nơi cần mở rộng

---

## 9. Tài liệu tham chiếu trong project

| File | Mục đích |
|------|---------|
| `MD/project_proposal_fintech.md` | Lộ trình 12 tuần + Tech Stack đầy đủ |
| `MD/business_domain_guide.md` | Nghiệp vụ thẩm định tín dụng + Khung 5C |
| `MD/production_devops_cheatsheet.md` | Nginx, SSL, CI/CD, Docker, Chi phí |
| `MD/rag_pipeline_cheatsheet.md` | Giải thích Dense/Sparse, RRF, Cross-Encoder, Parent-Doc |
| `MD/Mindset.md` | Tư duy Vertical Slice, Vibe Coding đúng cách |
| `../phase_4/langfuse_observability_cheatsheet.md` | Langfuse tích hợp Phase 3 |
| `../phase_4/retriever_strategies_cheatsheet.md` | Chiến thuật RAG nâng cao |
