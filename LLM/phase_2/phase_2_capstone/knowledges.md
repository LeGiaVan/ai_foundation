1. Tại sao fitz dùng ngay với Bytes được fitz.open(stream=file_bytes, filetype="pdf") còn docx thì phải Document(io.BytesIO(file_bytes))?
    - 
    - Do cách viết của 2 file. Fitz mới, có chức năng stream=file_bytes đọc trực tiếp. Còn docx cũ, chỉ đọc được raw text, nên cần io.BytesIO để convert từ file_bytes thành file ảo và lưu trên RAM.
    - Kiểu dữ liệu file_bytes có dạng b'%PDF-1.4...'