from groq import Groq
from config import settings
from vector_store import vector_store_instance
from models import AskResponse, SourceSnippet
import time
from reranker import reranker_instance

groq_client = Groq(api_key=settings.groq_api_key)

SYSTEM_PROMPT = """Bạn là trợ lý AI trả lời câu hỏi dựa trên tài liệu được cung cấp.
Dưới đây là một số thông tin (context) được tìm thấy từ tài liệu.
Hãy dựa CHỈ VÀO THÔNG TIN TRONG CONTEXT để trả lời câu hỏi của người dùng.
Nếu thông tin không đủ để trả lời, HÃY TRẢ LỜI RÕ RÀNG là "Không có trong tài liệu". KHÔNG ĐƯỢC bịa đặt hay sử dụng kiến thức bên ngoài.

CONTEXT:
{context}
"""

def build_context_string(chunks: list[dict]) -> str:
    context = ""
    for i, c in enumerate(chunks):
        context += f"--- Nguồn {i+1} (Tệp: {c['filename']}, Trang: {c['page']}) ---\n"
        context += f"{c['text']}\n\n"
    return context

def ask_question(question: str, doc_id: str = None) -> AskResponse:
    # 1. Retrieval (lấy top 10)
    chunks = vector_store_instance.search_hybrid(query=question, doc_id=doc_id, top_k=10)
    
    if not chunks:
        return AskResponse(answer="Không có trong tài liệu", sources=[])
        
    # 2. Reranking (lấy top 3)
    reranked_chunks = reranker_instance.rerank(question, chunks, top_k=3)
    
    # Xây dựng context từ top 3
    context_str = build_context_string(reranked_chunks)
    
    # 2. Sinh câu trả lời với LLM
    response = groq_client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context_str)},
            {"role": "user", "content": question}
        ],
        temperature=0.1
    )
    
    answer = response.choices[0].message.content
    
    # 3. Trả về kết quả kèm sources
    sources = [
        SourceSnippet(
            text=c["text"],
            doc_name=c["filename"],
            page_number=c["page"],
            score=c["score"]
        ) for c in reranked_chunks
    ]
    
    return AskResponse(answer=answer, sources=sources)

def ask_question_stream(question: str, doc_id: str = None):
    # Lấy top 10
    chunks = vector_store_instance.search_hybrid(query=question, doc_id=doc_id, top_k=10)
    
    if not chunks:
        yield "Không có trong tài liệu"
        return
        
    # Rerank lấy top 3
    reranked_chunks = reranker_instance.rerank(question, chunks, top_k=3)
        
    context_str = build_context_string(reranked_chunks)
    
    stream = groq_client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context_str)},
            {"role": "user", "content": question}
        ],
        temperature=0.1,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
            time.sleep(0.05)  # 🐌 CỐ TÌNH LÀM CHẬM LẠI ĐỂ MẮT NGƯỜI NHÌN THẤY HIỆU ỨNG GÕ CHỮ


