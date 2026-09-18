# 🌐 Autonomous Multi-Agent Research Assistant (Phase 3 Capstone)

An enterprise-grade, stateful Multi-Agent research system built with **LangGraph**, **FastAPI**, and **Groq LLM (`openai/gpt-oss-120b`)**. The system autonomously decomposes complex user inquiries, coordinates specialized domain workers, enforces security with **Human-in-the-Loop (HITL)** safeguards, and personalizes responses using **Long-Term Agentic Memory**.

---

## 🏗️ System Architecture

The following diagram illustrates the cyclical state machine, worker routing, and interrupt checkpoints:

<p align="center">
  <img src="architechture.png" alt="System Architecture" width="520" />
</p>

### High-Level Workflow
1. **User Request Submission**: A research query is submitted via the asynchronous FastAPI endpoint (`POST /research`).
2. **Supervisor Orchestration**: The `supervisor` node injects long-term user preferences from the `MemoryStore` and plans the next optimal step.
3. **Specialized Worker Execution**:
   - **`rag_worker`**: Retrieves proprietary internal documentation from local vector stores (**Qdrant**).
   - **`web_search_worker`**: Queries the live web for breaking news and real-time data (**DuckDuckGo Search**).
   - **`calculator_worker`**: Executes sandboxed, validated mathematical calculations.
4. **Human-in-the-Loop (HITL) Gate**: Before executing external or costly web searches, the execution is automatically suspended (`interrupt_before=["web_search_worker"]`). Execution resumes only upon user review and approval via `POST /research/{job_id}/approve`.
5. **Synthesis & Structured Output**: The `synthesizer` aggregates all observations, applies personal constraints, and returns a verified Pydantic schema.

---

## 🎯 Key Architectural Highlights

### 1. 🧠 Long-Term Agentic Memory (`MemoryStore`)
Unlike conversational window memory (short-term buffer) which resets across sessions, this system features a persistent, cross-session memory layer:
- **Automatic Fact & Preference Extraction**: Detects persona traits, formatting preferences, and domain expertise from user inputs.
- **Dynamic Retrieval**: Re-injects relevant historical memories into the system prompt at the start of every session.
- **Autonomous Memory Consolidation**: When memories exceed 10 records per user, an LLM consolidation routine purges redundancies and summarizes preferences.

### 2. 🛑 Human-In-The-Loop (HITL) Guardrails
- **Cost & API Control**: Prevents recursive web crawling and excessive API token consumption.
- **Prompt Injection Defense**: Defends against Indirect Prompt Injections by requiring human authorization before taking external network actions.
- **State Checkpointing**: Employs LangGraph's `MemorySaver` checkpointer, persisting execution thread state across asynchronous HTTP requests.

### 3. 📦 Strict Structured Output Guarantee
Guarantees consistent, machine-readable JSON responses adhering to the `ResearchResult` schema:
```json
{
  "answer": "Detailed, markdown-formatted response with comparative tables...",
  "sources": [
    "DuckDuckGo Web Search",
    "Tài liệu nội bộ Qdrant Knowledge Base"
  ],
  "agents_used": [
    "Supervisor Agent",
    "Web Search Agent",
    "Calculator Agent"
  ],
  "confidence": 0.97
}
```

---

## 🔌 API Reference & Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/research` | Submit a research query; returns an asynchronous `job_id`. |
| `GET` | `/research/{job_id}` | Poll job status (`running`, `waiting_approval`, `completed`, `error`). |
| `POST` | `/research/{job_id}/approve` | Approve (`approve=true`) or reject (`approve=false`) a pending HITL tool call. |
| `GET` | `/user/{user_id}/memories` | Inspect all persistent memory entries associated with a user profile. |
| `DELETE` | `/user/{user_id}/memories` | Clear long-term memory for a given user profile. |

---

## 🚀 Quickstart & Setup

### 1. Prerequisites & Virtual Environment
Ensure dependencies are installed in your virtual environment:
```powershell
# Navigate to Phase 3 directory
cd d:\ai_foundation\LLM\phase_3

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Ensure required libraries are installed
pip install langgraph langchain-groq qdrant-client duckduckgo_search fastapi uvicorn pydantic
```

### 2. Configuration (`.env`)
Create a `.env` file in `d:\ai_foundation\LLM\phase_3` with your API credentials:
```env
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_docs
```

### 3. Running the FastAPI Server
You can launch the server from either the project root or the capstone package directory:
```powershell
# Option A: From LLM/phase_3 root
uvicorn src.capstone.api:app --reload --port 8000

# Option B: From within src/capstone
cd src/capstone
uvicorn api:app --reload --port 8000
```
Interactive API documentation is accessible at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

---

## 🧪 Automated Verification Suite

Run the end-to-end integration and unit test suite:
```powershell
python test_capstone.py
```

### Test Suite Execution Output:
```text
🚀 BẮT ĐẦU KIỂM THỬ TOÀN DIỆN CAPSTONE PROJECT PHASE 3 🚀

=================================================================
🌟 1. KIỂM TRA LONG-TERM AGENTIC MEMORY STORE
=================================================================
✅ Đã truy xuất 2 memories của 'user_nam_ai':
   - Tôi thích câu trả lời ngắn gọn, có bảng biểu so sánh.
   - Chuyên môn: Computer Vision & Edge AI.
👉 Test MemoryStore: ĐẠT CHUẨN ✅

=================================================================
🌟 2. KIỂM TRA CÁC TOOLS ĐỘC LẬP
=================================================================
🧮 [calculate] 15 * 8 = 120
📚 [rag_search] Kết quả: YOLOv11 (1.5ms GPU, 11.5ms CPU)
🌐 [web_search] Kết quả: DuckDuckGo Live Search OK
📊 [format_table]: Markdown Table Formatting OK
👉 Test Tools: ĐẠT CHUẨN ✅

=================================================================
🌟 3. KIỂM TRA END-TO-END: RESEARCH AGENT + HITL + STRUCTURED OUTPUT
=================================================================
[Bước 1] Gửi câu hỏi nghiên cứu tới POST /research...
[API 🛑] Job e6a7143d-... tạm dừng (HITL), chờ duyệt tại /approve...
[Bước 2] Trạng thái hiện tại: waiting_approval
[Bước 3] Phê duyệt qua POST /research/{job_id}/approve...
[API 🎉] Job hoàn thành xuất sắc!
🎯 Confidence: 0.97 | Agents Used: Web Search Agent
👉 Test End-to-End + HITL: ĐẠT CHUẨN ✅

=================================================================
🌟 4. KIỂM TRA AGENT ÁP DỤNG LONG-TERM MEMORY TỪ SESSION TRƯỚC
=================================================================
📝 Câu trả lời Session 2: Tự động render bảng so sánh thuộc tính YOLOv11n
   và tập trung vào Edge AI / Computer Vision theo đúng sở thích người dùng!
👉 Test Long-Term Memory Continuity: ĐẠT CHUẨN ✅

=================================================================
🌟 TẤT CẢ CÁC BÀI TEST CAPSTONE PHASE 3 ĐÃ VƯỢT QUA 100%!
=================================================================
```

---

## 📂 Repository File Structure

```text
d:/ai_foundation/LLM/phase_3/
├── src/
│   └── capstone/
│       ├── __init__.py
│       ├── agent_graph.py         # LangGraph StateGraph, Supervisor, HITL & Workers
│       ├── api.py                 # FastAPI Web Server & Background Job Orchestrator
│       ├── memory_store.py        # Long-Term Agentic Memory & LLM Consolidation
│       ├── tools.py               # 5 Production Tools (RAG, Web, Calc, Table, Datetime)
│       ├── schemas.py             # Pydantic Schemas for Structured Output & API
│       ├── architechture.png      # Rendered LangGraph Architecture Diagram
│       ├── graph_diagram.md       # Raw Mermaid Source Code
│       ├── user_memories.json     # Persistent Memory Database
│       └── README.md              # English Project Documentation
├── test_capstone.py               # Comprehensive Automated Verification Suite
└── plan.md                        # Learning Roadmap & Checklist
```
