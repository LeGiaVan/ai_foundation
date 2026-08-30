import pymupdf  # PyMuPDF
from docx import Document
import io
from fastapi import UploadFile

async def extract_text_from_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """Trích xuất text từ PDF, lưu số trang"""
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    pages_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            pages_text.append({
                "text": text,
                "page": page_num + 1,
                "filename": filename
            })
    return pages_text

async def extract_text_from_docx(file_bytes: bytes, filename: str) -> list[dict]:
    """Trích xuất text từ DOCX. Xem toàn bộ file như 1 'trang'"""
    doc = Document(io.BytesIO(file_bytes))
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    
    joined_text = "\n".join(full_text)
    if not joined_text.strip():
        return []
        
    return [{
        "text": joined_text,
        "page": 1,
        "filename": filename
    }]

async def parse_document(file: UploadFile) -> list[dict]:
    file_bytes = await file.read()
    filename = file.filename
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    
    if ext == "pdf":
        return await extract_text_from_pdf(file_bytes, filename)
    elif ext == "docx":
        return await extract_text_from_docx(file_bytes, filename)
    else:
        raise ValueError(f"Định dạng file không được hỗ trợ: {ext}")
