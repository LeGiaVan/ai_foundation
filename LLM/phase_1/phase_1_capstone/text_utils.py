import tiktoken

def count_tokens(text: str, model: str) -> int:
    """
    Counting Token Using Tiktoken
    (Hàm này giữ lại để dùng ở các file khác nếu cần)
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
    
def chunk_text(text: str, model: str, chunk_size: int = 2000) -> list[str]:
    """
    Chunking Text
    Strategy:
        - Parse by Paragraphs
        - If Para Token > Max Token Size => Split by Sentences
        - If Sentence Token > Max Token Size => Split by Words
    """
    # Khởi tạo encoding 1 lần duy nhất để tối ưu hiệu năng (Performance fix)
    encoding = tiktoken.encoding_for_model(model)
    
    def get_tokens(s: str) -> int:
        return len(encoding.encode(s))

    chunks = []
    # Lưu các mảnh (piece) dưới dạng Tuple: (nội_dung, dấu_phân_cách) để không làm hỏng format gốc
    current_pieces = []
    current_tokens = 0

    def add_piece(piece_text: str, delimiter: str):
        """
        Hàm phụ trợ để thêm một 'mảnh' (đoạn/câu/từ) vào chunk hiện tại.
        
        Quy trình xử lý:
        1. Tính toán số token của mảnh mới (bao gồm cả dấu phân cách).
        2. Nếu thêm vào mà vượt quá giới hạn `chunk_size`:
           - Đóng gói (join) các mảnh hiện tại thành một chuỗi hoàn chỉnh và lưu vào list `chunks`.
           - Bắt đầu một chunk mới. Đồng thời, giữ lại (overlap) mảnh cuối cùng của chunk trước đó 
             để đảm bảo không bị mất ngữ cảnh giữa 2 chunk.
           - Đưa mảnh mới hiện tại vào chunk mới này.
        3. Nếu chưa vượt quá giới hạn:
           - Trực tiếp nối mảnh này vào danh sách mảnh hiện tại.
        """
        nonlocal current_pieces, current_tokens
        # nonlocal giống như global, nhưng global cho cho biến ở file, nonlocal cho cho biến ở hàm cha
        piece_tokens = get_tokens(piece_text + delimiter)
        
        # Nếu thêm mảnh này vào bị vượt quá chunk_size và chunk hiện tại không rỗng
        if current_tokens + piece_tokens > chunk_size and current_pieces:
            # Gộp các mảnh hiện tại thành 1 Chunk hoàn chỉnh và lưu lại
            chunk_str = "".join([p[0] + p[1] for p in current_pieces])
            chunks.append(chunk_str.strip())
            
            # Start new chunk với Overlap (Giữ lại mảnh cuối cùng của chunk trước làm ngữ cảnh)
            last_piece = current_pieces[-1]
            current_pieces = [last_piece, (piece_text, delimiter)]
            current_tokens = get_tokens(last_piece[0] + last_piece[1]) + piece_tokens
        else:
            current_pieces.append((piece_text, delimiter))
            current_tokens += piece_tokens

    # Bước 1: Chia theo đoạn văn (Paragraphs)
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        if get_tokens(para) <= chunk_size:
            add_piece(para, '\n\n')
        else:
            # Bước 2: Paragraph quá dài -> Cắt thành câu (Sentences)
            sentences = para.split('. ')
            for sentence in sentences:
                if get_tokens(sentence) <= chunk_size:
                    add_piece(sentence, '. ')
                else:
                    # Bước 3: Sentence quá dài -> Cắt thành từ (Words)
                    words = sentence.split(' ')
                    for word in words:
                        add_piece(word, ' ')

    # Lưu lại chunk cuối cùng nếu còn sót dữ liệu
    if current_pieces:
        chunk_str = "".join([p[0] + p[1] for p in current_pieces])
        chunks.append(chunk_str.strip())

    return chunks