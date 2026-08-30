import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import List

from config import settings
from models import HealthResponse, UploadResponse, AskRequest, AskResponse, DocumentMetadata
from document_parser import parse_document
from chunker import chunk_document
from vector_store import vector_store_instance
from rag_pipeline import ask_question, ask_question_stream

app = FastAPI(title="Document Q&A API (Phase 2 Capstone)")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        dense_model=settings.dense_embedding_model,
        sparse_model=settings.sparse_embedding_model
    )

@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        # 1. Đọc và extract text
        pages_data = await parse_document(file)
        if not pages_data:
            raise HTTPException(status_code=400, detail="Không tìm thấy text trong tài liệu.")
            
        # 2. Chunking
        chunks = chunk_document(pages_data, settings.chunk_size, settings.chunk_overlap)
        
        # 3. Embedding & Upsert
        doc_id = str(uuid.uuid4())
        chunk_count = vector_store_instance.upsert_document(doc_id, file.filename, chunks)
        
        return UploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            chunk_count=chunk_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentMetadata])
async def list_documents():
    try:
        docs = vector_store_instance.get_all_documents()
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    try:
        vector_store_instance.delete_document(doc_id)
        return {"status": "success", "message": f"Đã xóa tài liệu {doc_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        response = ask_question(request.question, request.doc_id)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    try:
        stream = ask_question_stream(request.question, request.doc_id)
        return StreamingResponse(stream, media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
