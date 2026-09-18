import os
import sys
import uuid
import asyncio
from typing import Dict, Any, Optional

# Hỗ trợ import linh hoạt dù chạy uvicorn từ thư mục nào
current_dir = os.path.dirname(os.path.abspath(__file__))
phase_3_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if phase_3_dir not in sys.path:
    sys.path.insert(0, phase_3_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastapi import FastAPI, HTTPException, BackgroundTasks
# BackgroundTasks khai báo với Instance là hãy chạy ngầm Task này
from langchain_core.messages import HumanMessage, AIMessage

try:
    from src.capstone.schemas import (
        ResearchRequest, 
        ResearchJobResponse, 
        JobStatusResponse, 
        ApproveRequest, 
        ResearchResult, 
        PendingAction
    )
    from src.capstone.agent_graph import research_graph, memory_store
except ImportError:
    from schemas import (
        ResearchRequest, 
        ResearchJobResponse, 
        JobStatusResponse, 
        ApproveRequest, 
        ResearchResult, 
        PendingAction
    )
    from agent_graph import research_graph, memory_store

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

app = FastAPI(
    title="Capstone Phase 3: Research Assistant Agent API",
    description="Multi-Agent System với LangGraph Supervisor, RAG, Web Search, HITL và Long-Term Memory."
)

# Quản lý trạng thái các Jobs trong RAM
# Cấu trúc: job_id -> {"status", "user_id", "question", "result", "pending_action", "error"}
JOBS: Dict[str, Dict[str, Any]] = {}

async def _execute_graph_job(job_id: str, is_resume: bool = False, approved: bool = True):
    """Hàm chạy hoặc resume LangGraph trong background."""
    config = {"configurable": {"thread_id": job_id}}
    
    try:
        if not is_resume:
            user_id = JOBS[job_id]["user_id"]
            question = JOBS[job_id]["question"]
            initial_input = {
                "messages": [HumanMessage(content=question)],
                "user_id": user_id,
                "agents_used": [],
                "sources": []
            }
            # Chạy bước đầu tiên cho đến khi gặp interrupt hoặc kết thúc
            research_graph.invoke(initial_input, config=config)
        else:
            if not approved:
                # Nếu người dùng từ chối, inject thông báo từ chối vào state
                current_state = research_graph.get_state(config)
                agents = list(current_state.values.get("agents_used", []))
                agents.append("Web Search (Từ chối bởi User)")
                research_graph.update_state(
                    config, 
                    {
                        "messages": [AIMessage(content="Người dùng đã từ chối quyền truy cập Web Search. Hãy tổng hợp câu trả lời dựa trên những dữ liệu nội bộ hiện có.")],
                        "agents_used": agents
                    }
                )
            # Tiếp tục chạy từ điểm tạm dừng
            research_graph.invoke(None, config=config)

        # Kiểm tra trạng thái đồ thị sau khi chạy
        state = research_graph.get_state(config)
        
        # Kiểm tra nếu đang bị tạm dừng tại interrupt_before=["web_search_worker"]
        if state.next == ("web_search_worker",):
            JOBS[job_id]["status"] = "waiting_approval"
            JOBS[job_id]["pending_action"] = {
                "tool": "web_search",
                "input": {"query": JOBS[job_id]["question"]},
                "reason": "Agent cần tìm kiếm dữ liệu mở rộng trên Internet. Hành động này tốn chi phí và gọi ra ngoài, cần bạn xác nhận."
            }
            print(f"[API 🛑] Job {job_id} đã tạm dừng (HITL), đang chờ người dùng phê duyệt tại /approve...")
        else:
            # Đã chạy xong đến END
            final_res = state.values.get("final_result")
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["pending_action"] = None
            JOBS[job_id]["result"] = final_res
            print(f"[API 🎉] Job {job_id} đã hoàn thành xuất sắc!")
            
    except Exception as e:
        print(f"[API ❌] Lỗi khi chạy Job {job_id}: {e}")
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)

# === CÁC FASTAPI ENDPOINTS ===

@app.post("/research", response_model=ResearchJobResponse)
async def start_research(req: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Bắt đầu một phiên nghiên cứu mới.
    Trả về job_id để client theo dõi tiến trình.
    """
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "user_id": req.user_id,
        "question": req.question,
        "pending_action": None,
        "result": None,
        "error": None
    }
    
    # Kích hoạt chạy đồ thị bất đồng bộ
    background_tasks.add_task(_execute_graph_job, job_id, is_resume=False)
    
    return ResearchJobResponse(
        job_id=job_id,
        status="running",
        message="Yêu cầu nghiên cứu đã được tiếp nhận và đang được Multi-Agent xử lý."
    )

@app.get("/research/{job_id}", response_model=JobStatusResponse)
async def get_research_status(job_id: str):
    """
    Kiểm tra trạng thái của job nghiên cứu.
    Các trạng thái: 'running', 'waiting_approval', 'completed', 'error'.
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy job_id '{job_id}'")
        
    job = JOBS[job_id]
    pending_act = None
    if job["pending_action"]:
        pending_act = PendingAction(**job["pending_action"])
        
    res_obj = None
    if job["result"]:
        res_obj = ResearchResult(**job["result"])
        
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        pending_action=pending_act,
        result=res_obj,
        error=job["error"]
    )

@app.post("/research/{job_id}/approve", response_model=JobStatusResponse)
async def approve_research_action(job_id: str, req: ApproveRequest, background_tasks: BackgroundTasks):
    """
    Phê duyệt hoặc từ chối hành động HITL (Web Search) đang bị tạm dừng.
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy job_id '{job_id}'")
        
    job = JOBS[job_id]
    if job["status"] != "waiting_approval":
        raise HTTPException(
            status_code=400, 
            detail=f"Job '{job_id}' không ở trạng thái 'waiting_approval' (Trạng thái hiện tại: {job['status']})"
        )
        
    job["status"] = "running"
    job["pending_action"] = None
    
    # Resume graph
    background_tasks.add_task(_execute_graph_job, job_id, is_resume=True, approved=req.approve)
    
    return JobStatusResponse(
        job_id=job_id,
        status="running",
        pending_action=None,
        result=None,
        error=None
    )

@app.get("/user/{user_id}/memories")
async def get_user_memories(user_id: str):
    """Lấy danh sách các sở thích và thông tin Agent đã ghi nhớ về người dùng."""
    memories = memory_store.retrieve(user_id)
    return {
        "user_id": user_id,
        "total_memories": len(memories),
        "memories": memories
    }

@app.delete("/user/{user_id}/memories")
async def clear_user_memories(user_id: str):
    """Xóa bộ nhớ của người dùng."""
    memory_store._memories[user_id] = []
    memory_store._save_to_disk()
    return {"message": f"Đã xóa toàn bộ memory của user '{user_id}'."}
