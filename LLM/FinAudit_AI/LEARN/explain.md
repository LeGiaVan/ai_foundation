models.py

    - BlockType: Phân loại từng Block

    - StorageTarget: Đích lưu Data: sql / vector

    - ParsedBlock: Định nghĩa 1 khối văn bản / bản biểu được bóc tách từ PDF

    - ClassifiedBlock: Phân loại Block sau khi đi qua Module block_classifier.py => Quyết định Block đi qua sql / vector

    - Section: Là một đơn vị gom nhóm các ParsedBlock có chung Đề mục => Phục vụ Parent-Chill Chunking.

    - ParsedDocument: Định nghĩa toàn bộ văn bản BCTC sau khi được xử lý.

    - VerificationStatus: trang thái kiểm soát đối với dữ liệu tài chính cho sql.

    - FinancialFact: Fact được xử lý từ BCTC và lưu vào sql.

    - FinancialRatio: được tính toán từ các Fact bằng formula_engine.py

    - VerificationReport: Báo cáo kết quả kiểm toán số học toàn diện cho 1 BCTC

toc_inspector.py

    - DocumentStructure: Class lưu kết quả của quá trình trích xuất Table of Content.
        + Số trang, có thấy TOC không, Các Core Statement ở trang nào.
        + Intro Page => Bỏ qua.

    - TOCInspector: Lớp giao diện, mục đích là Phát hiện và Xử lý các thông tin trong TOC.

        + def _get_page_blocks:
            _ Nếu là Native PDF => Dùng text_parser để lấy các Block ra.
            _ Nếu là Non Native PDF => Dùng vision_pipeline để trích xuất text tình hình ảnh.
            => Trả về list các Block của 1 page. (entries)

        + def _find_toc_in_blocks:
            _ Duyệt tất cả các Block trong list, nếu phát hiện các Keyword cho thấy Block đó là TOC.
            _ Thì dùng def _parse_toc_markdown() cho nguyên Block đó để trả về một list các Tiêu đề và số trang ({"title": title, "pages": parsed_pages, "raw": page_raw})
            _ Có cơ chế FallBack rất hay ở đây là các Mục Lục không phải lúc nào cũng được kẻ bảng => Nên đôi lúc trong file MD nó chỉ là các plain text như bình dường (Có dạng: Báo cáo tình hình tài chính .................... 6) => dùng def _parse_toc_plain_text()

        + def _build_structure_from_toc:
            _ Muốn build cấu trúc của TOC, trước hết phải tính offset (độ lệch giữa trang in trong TOC và trang PDF) => Dùng def _detect_offset_by_anchor:
                _ Logic của hàm này là: Đọc TOC, lấy số trang đầu tiên được in, quét các trang tiếp theo để tìm trang Nội dung ứng với trang đầu tiên đó => Độ lệch
                _ Hàm chọn 4 dòng đầu tiên của Mục lục (BẢNG CÂN ĐỐI, BÁO CÁO KẾT QUẢ,..) => Lấy trang in nhỏ nhất (thường là 1) [1]
                _ Quét lần lượt khoảng 5 trang tiếp theo để tìm trang có Nội dung đầu tiên. Kiểm tra bằng def _is_anchor_match với các từ khóa         core_terms = ["CÂN ĐỐI", "KẾT QUẢ", "LƯU CHUYỂN", "THUYẾT MINH", "BAN ĐIỀU HÀNH", "BAN GIÁM ĐỐC", "KIỂM TOÁN"]
                _ Nếu Match => Xác nhận đó là trang Nội dung đầu tiên. [2]
                _ Lấy [2] - [1] = Offset
