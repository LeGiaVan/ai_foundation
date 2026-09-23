Regex - Regular Expression
    - Công cụ tìm kiếm, so khớp và xử lý chuỗi văn bản theo quy luật.
    - re.sub()
    - re.match() 
    - re.search()
    - re.compile(): biên dịch để dùng lại nhiều lần

@staticmethod
    - biến một hàm bên trong class thành một hàm độc lập, KHÔNG phụ thuộc vào đối tượng (self).
    - 2 cách gọi:
        + Cách 1: Gọi trực tiếp từ Class (KHÔNG cần khởi tạo đối tượng)
        doc = TOCInspector._build_fallback_structure(total_pages=54)
        + Cách 2: Gọi từ bên trong một hàm khác của đối tượng
        return self._build_fallback_structure(total_pages)


