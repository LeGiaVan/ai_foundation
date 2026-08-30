from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import tiktoken

def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))

def chunk_document(pages_data: list[Dict], chunk_size: int, chunk_overlap: int) -> list[Dict]:
    """
    Cắt text thành các chunks dựa trên kích thước.
    Giữ nguyên metadata (page, filename) của trang gốc cho mỗi chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=count_tokens,  # Đo bằng token thay vì ký tự
    )
    
    chunks = []
    for page_info in pages_data:
        text = page_info["text"]
        page_chunks = splitter.split_text(text)
        
        for c in page_chunks:
            chunks.append({
                "text": c,
                "page": page_info["page"],
                "filename": page_info["filename"]
            })
            
    return chunks
