# LangChain Text Splitters — Cheatsheet

---

## 1. Import cơ bản

```python
from langchain_text_splitters import (
    CharacterTextSplitter,          # Fixed-size splitting
    RecursiveCharacterTextSplitter, # Recursive splitting (phổ biến nhất)
)
```

---

## 2. Fixed-Size Splitting (`CharacterTextSplitter`)

Cắt đều theo một separator cố định (mặc định là `"\n\n"`).

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",   # Cắt tại đâu (mặc định: "\n\n")
    chunk_size=500,      # Kích thước tối đa mỗi chunk (tính theo ký tự)
    chunk_overlap=50,    # Số ký tự chồng lấn giữa 2 chunk liền kề
    length_function=len, # Hàm đo kích thước (mặc định: len = đếm ký tự)
)

chunks = splitter.split_text(text)
# Trả về: list[str]
```

**Đặc điểm:**
- Chỉ cắt tại đúng 1 loại separator duy nhất
- Nếu đoạn văn giữa 2 separator dài hơn `chunk_size` → vẫn giữ nguyên (không cắt tiếp)
- Đơn giản, nhanh, nhưng dễ bị cắt giữa ý

---

## 3. Recursive Character Text Splitting (`RecursiveCharacterTextSplitter`) ⭐

**Đây là splitter được dùng nhiều nhất trong thực tế.**

Thử tách lần lượt theo thứ tự ưu tiên: `"\n\n"` → `"\n"` → `". "` → `" "` → `""`.
Nếu đoạn nào vẫn quá dài sau khi cắt bằng separator đầu tiên, nó sẽ tự động dùng separator tiếp theo để cắt nhỏ hơn.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],  # Thứ tự ưu tiên (mặc định)
    chunk_size=500,       # Kích thước tối đa mỗi chunk
    chunk_overlap=50,     # Số ký tự chồng lấn
    length_function=len,  # Hàm đo kích thước
)

chunks = splitter.split_text(text)
# Trả về: list[str]
```

**Đặc điểm:**
- Chunk thường kết thúc ở ranh giới tự nhiên (cuối đoạn, cuối câu)
- Ít bị cắt giữa từ hoặc giữa ý so với Fixed-size
- Logic giống hệt hàm `chunk_text()` bạn đã tự viết ở Phase 1

---

## 4. Đo kích thước bằng Token thay vì Ký tự

Mặc định LangChain đo bằng `len()` (đếm số ký tự). Nhưng LLM/Embedding model tính theo **token**, không phải ký tự. Để chính xác hơn, bạn có thể thay `length_function` bằng hàm đếm token:

```python
import tiktoken

# Tạo hàm đếm token
def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=count_tokens,  # Đo bằng token thay vì ký tự
)
```

---

## 5. Tạo Documents thay vì list[str]

LangChain có khái niệm `Document` — là một object chứa `page_content` (nội dung) và `metadata` (thông tin bổ sung như tên file, số trang...).

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Cách 1: split_text() → trả về list[str]
chunks = splitter.split_text(text)

# Cách 2: split_documents() → trả về list[Document] (giữ nguyên metadata)
docs = [Document(page_content=text, metadata={"source": "report.pdf", "page": 1})]
chunked_docs = splitter.split_documents(docs)

# Mỗi chunked_doc giữ nguyên metadata từ Document gốc
for doc in chunked_docs:
    print(doc.page_content)  # Nội dung chunk
    print(doc.metadata)      # {"source": "report.pdf", "page": 1}
```

---

## 6. So sánh nhanh

| Tiêu chí | `CharacterTextSplitter` | `RecursiveCharacterTextSplitter` |
|---|---|---|
| Cách cắt | 1 separator duy nhất | Nhiều separator theo thứ tự ưu tiên |
| Chất lượng chunk | Trung bình (có thể cắt giữa ý) | Tốt (cắt tại ranh giới tự nhiên) |
| Tốc độ | Nhanh hơn một chút | Nhanh (chênh lệch không đáng kể) |
| Khi nào dùng | Prototype nhanh, dữ liệu đơn giản | **Hầu hết mọi trường hợp** |

---

## 7. Ví dụ hoàn chỉnh: So sánh 2 splitters trên cùng 1 văn bản

```python
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

text = """Trí tuệ nhân tạo (AI) là một lĩnh vực rộng lớn.

Học máy (ML) là nhánh con của AI. ML học từ dữ liệu thay vì được lập trình tường minh. Các thuật toán phổ biến: hồi quy, cây quyết định, mạng nơ-ron.

Học sâu (Deep Learning) dùng mạng nơ-ron nhiều tầng. CNN xử lý ảnh. Transformer xử lý ngôn ngữ."""

# --- Fixed-size ---
fixed_splitter = CharacterTextSplitter(
    separator="\n\n", chunk_size=100, chunk_overlap=20
)
fixed_chunks = fixed_splitter.split_text(text)

# --- Recursive ---
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=20
)
recursive_chunks = recursive_splitter.split_text(text)

# --- In kết quả so sánh ---
print("=== FIXED-SIZE ===")
for i, c in enumerate(fixed_chunks):
    print(f"Chunk {i+1} ({len(c)} chars): {c[:80]}...")

print("\n=== RECURSIVE ===")
for i, c in enumerate(recursive_chunks):
    print(f"Chunk {i+1} ({len(c)} chars): {c[:80]}...")
```

---

## 8. Quy tắc vàng chọn chunk_size và chunk_overlap

| Tham số | Gợi ý | Giải thích |
|---|---|---|
| `chunk_size` | 256-512 tokens (Semantic Search) | Chunk nhỏ → vector chính xác hơn |
| `chunk_size` | 512-1024 tokens (Q&A) | Chunk lớn hơn → đủ ngữ cảnh trả lời |
| `chunk_overlap` | 10-20% của chunk_size | Quá ít → mất ý. Quá nhiều → tốn bộ nhớ |
