import tiktoken


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Đếm số token của một chuỗi text."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    model: str = "gpt-4o",
) -> list[str]:
    """
    Recursive Character Text Splitting với token-based overlap.

    Chiến lược tách theo thứ tự ưu tiên:
        1. Đoạn văn (split by "\\n\\n")
        2. Dòng (split by "\\n")
        3. Câu (split by ". ")
        4. Từ (split by " ")

    Args:
        text: Văn bản cần chia nhỏ.
        chunk_size: Số token tối đa mỗi chunk.
        chunk_overlap: Số token chồng lấn giữa 2 chunk liền kề.
        model: Tên model để tính token (dùng tiktoken).

    Returns:
        Danh sách các chunk (list[str]).
    """
    # Khởi tạo encoding 1 lần duy nhất
    encoding = tiktoken.encoding_for_model(model)

    def get_tokens(s: str) -> int:
        return len(encoding.encode(s))

    # --- Bước 1: Tách văn bản thành các pieces nhỏ theo thứ tự ưu tiên ---
    separators = ["\n\n", "\n", ". ", " "]

    def split_recursive(text_block: str, sep_index: int = 0) -> list[str]:
        """
        Tách text_block thành các mảnh nhỏ hơn chunk_size.
        Nếu mảnh nào vẫn quá lớn, tiếp tục tách bằng separator tiếp theo.
        """
        # Nếu đã hết separator để tách, trả về nguyên khối
        if sep_index >= len(separators):
            return [text_block]

        sep = separators[sep_index]
        parts = text_block.split(sep)
        pieces = []

        for part in parts:
            if get_tokens(part) <= chunk_size:
                pieces.append(part)
            else:
                # Mảnh vẫn quá lớn -> Tách tiếp bằng separator nhỏ hơn
                sub_pieces = split_recursive(part, sep_index + 1)
                pieces.extend(sub_pieces)

        return pieces

    pieces = split_recursive(text)
    # Loại bỏ các mảnh rỗng
    pieces = [p for p in pieces if p.strip()]

    # --- Bước 2: Gộp các pieces thành chunks, có kiểm soát overlap ---
    chunks = []
    current_chunk_pieces = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = get_tokens(piece)

        # Nếu thêm piece này vào mà vượt chunk_size -> Đóng chunk hiện tại
        if current_tokens + piece_tokens > chunk_size and current_chunk_pieces:
            # Lưu chunk hiện tại
            chunk_text_str = " ".join(current_chunk_pieces).strip()
            chunks.append(chunk_text_str)

            # Tính overlap: Giữ lại các pieces cuối sao cho tổng token ≈ chunk_overlap
            overlap_pieces = []
            overlap_tokens = 0

            for prev_piece in reversed(current_chunk_pieces):
                prev_tokens = get_tokens(prev_piece)
                if overlap_tokens + prev_tokens > chunk_overlap:
                    break
                overlap_pieces.insert(0, prev_piece)
                overlap_tokens += prev_tokens

            # Bắt đầu chunk mới với phần overlap + piece hiện tại
            current_chunk_pieces = overlap_pieces + [piece]
            current_tokens = overlap_tokens + piece_tokens
        else:
            current_chunk_pieces.append(piece)
            current_tokens += piece_tokens

    # Lưu chunk cuối cùng nếu còn sót
    if current_chunk_pieces:
        chunk_text_str = " ".join(current_chunk_pieces).strip()
        chunks.append(chunk_text_str)

    return chunks


# --- DEMO ---
if __name__ == "__main__":
    sample_text = """Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính. AI tập trung vào việc tạo ra các hệ thống có khả năng thực hiện các tác vụ đòi hỏi trí thông minh của con người.

Học máy (Machine Learning) là một nhánh con của AI. Thay vì lập trình tường minh từng quy tắc, hệ thống ML học từ dữ liệu để đưa ra dự đoán hoặc quyết định. Các thuật toán phổ biến bao gồm: hồi quy tuyến tính, cây quyết định, mạng nơ-ron.

Học sâu (Deep Learning) là một nhánh con của Machine Learning. Deep Learning sử dụng mạng nơ-ron nhiều tầng (deep neural networks) để xử lý dữ liệu phức tạp như hình ảnh, âm thanh, và văn bản. Các kiến trúc nổi tiếng bao gồm CNN cho xử lý ảnh và Transformer cho xử lý ngôn ngữ tự nhiên.

Mô hình ngôn ngữ lớn (Large Language Models - LLMs) là các mô hình Deep Learning được huấn luyện trên lượng dữ liệu văn bản khổng lồ. Chúng có khả năng hiểu và sinh ra ngôn ngữ tự nhiên. GPT, Claude, Gemini là những ví dụ tiêu biểu."""

    chunks = chunk_text(sample_text, chunk_size=100, chunk_overlap=20)

    for i, chunk in enumerate(chunks):
        tokens = count_tokens(chunk)
        print(f"\n--- Chunk {i + 1} ({tokens} tokens) ---")
        print(chunk)

    print(f"\n\nTổng số chunks: {len(chunks)}")
