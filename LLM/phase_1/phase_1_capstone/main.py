from fastapi import FastAPI, Depends
from schemas import DocumentSummary, QueryRequest, SummarizeRequest
from llm_service import GroqClient, DocumentProcessor, BaseLLMClient

app = FastAPI(title="Capstone API")

# --- HÀM DEPENDENCY INJECTION ---
def get_llm_client() -> BaseLLMClient:
    return GroqClient()

def get_document_processor(client: BaseLLMClient = Depends(get_llm_client)) -> DocumentProcessor:
    return DocumentProcessor(client)

# --- ROUTES ---

@app.post("/summarize", response_model=DocumentSummary)
async def summarize_document(
    request: SummarizeRequest,
    doc_processor: DocumentProcessor = Depends(get_document_processor)
):
    # Phải có chữ 'await' vì summarize_long_document là hàm async
    result = await doc_processor.summarize_long_document(request.text)
    return result

@app.post("/chat")
async def answer_question(
    request: QueryRequest,
    doc_processor: DocumentProcessor = Depends(get_document_processor)
):
    # Hiện tại hàm answer_question_stream của bạn đang dùng lệnh print() để in ra console
    # Do đó Endpoint này gọi xong sẽ in chữ ra Terminal của bạn.
    await doc_processor.answer_question_stream(request.context_text, request.question)
    
    return {"status": "Success, please check your Terminal/Console for the stream"}