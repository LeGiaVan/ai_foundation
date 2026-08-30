import requests

# Văn bản có xuống dòng thoải mái nhờ 3 dấu ngoặc kép
tai_lieu = """
Vào ngày 26 tháng 8 năm 2026, Meta đồng ý chi trả tối đa 16,68 tỷ USD để dàn xếp các vụ kiện do 29 bang của Hoa Kỳ đưa ra liên quan đến Facebook và Instagram. Các bang cáo buộc Meta thiết kế nền tảng theo cách có thể khiến trẻ em và thanh thiếu niên bị cuốn vào mạng xã hội, gây hiểu lầm cho công chúng về mức độ an toàn của các dịch vụ và thu thập thông tin cá nhân của trẻ em không đúng cách. Meta phủ nhận hành vi sai trái nhưng vẫn đồng ý với thỏa thuận, qua đó kết thúc một vụ kiện lớn liên quan đến các công ty mạng xã hội.
Những người bị ảnh hưởng nhiều nhất bởi thỏa thuận này là trẻ em và thanh thiếu niên sử dụng Facebook và Instagram, cùng với cha mẹ và gia đình của họ. Vụ kiện tập trung vào những lo ngại rằng việc sử dụng mạng xã hội có thể ảnh hưởng tiêu cực đến sức khỏe và sự phát triển của người trẻ, đồng thời Meta chưa làm đủ để bảo vệ người dùng chưa đủ tuổi. Thỏa thuận cũng giải quyết các khiếu nại riêng về quyền riêng tư liên quan đến một số bang, trong đó có vụ bê bối Cambridge Analytica.
Sau thỏa thuận, Facebook và Instagram sẽ thay đổi cách hoạt động đối với người dùng tuổi teen tại Hoa Kỳ. Meta dự kiến áp dụng giới hạn thời gian sử dụng, bao gồm tối đa hai giờ mỗi ngày và hạn chế sử dụng từ nửa đêm đến 6 giờ sáng, trừ khi được cha mẹ cho phép. Thông báo cũng sẽ được giảm trong giờ học, đồng thời Meta sẽ tăng cường xác minh độ tuổi, kiểm soát của phụ huynh và các biện pháp bảo vệ người trẻ khỏi nội dung bị giới hạn độ tuổi hoặc có hại. Nhìn chung, thỏa thuận này buộc Meta phải có trách nhiệm lớn hơn trong việc bảo vệ người dùng trẻ tuổi và thay đổi cách Facebook, Instagram hoạt động đối với thanh thiếu niên.
những điều kiện thay đổi
-Thời gian nghỉ có ích” (Productive Pauses):Trẻ em sẽ được yêu cầu tạm dừng sau 15 phút sử dụng liên tục, và tiếp tục được nhắc nghỉ sau 60 phút và 90 phút, nhằm ngăn việc lướt nội dung vô tận.
-“Giới hạn ban đêm” (Nighttime Blocks):Hạn chế trẻ em truy cập nền tảng từ 12:00 đêm đến 6:00 sáng.
-Hạn chế sử dụng trong giờ học: Trong năm học, vào các ngày trong tuần từ 8:00 sáng đến 3:00 chiều, trẻ em sẽ không nhận được thông báo đẩy (push notifications).
-Biện pháp xác minh độ tuổi chặt chẽ hơn:Tăng cường các biện pháp để xác minh chính xác độ tuổi của người dùng trẻ tuổi.
-Kiểm soát nội dung an toàn và phù hợp với độ tuổi: Tăng cường bảo vệ trẻ em trước bắt nạt, nội dung cổ súy rối loạn ăn uống, cũng như nội dung liên quan đến tự tử và tự làm hại bản thân.
-Kiểm soát của phụ huynh mạnh mẽ và dễ sử dụng hơn: Cung cấp cho cha mẹ những công cụ dễ sử dụng và hiệu quả hơn để quản lý hoạt động của con trên nền tảng.
-Hạn chế các tính năng so sánh xã hội: Hạn chế những tính năng như bộ lọc làm đẹp và số lượt thích (like) hiển thị công khai, vốn có liên quan đến những tác động tiêu cực đến sức khỏe tinh thần của trẻ em và thanh thiếu niên.
-Đánh giá độc lập thường xuyên: Cả quá trình triển khai và hiệu quả của các biện pháp này sẽ được một đơn vị kiểm toán độc lập và các bang tham gia thỏa thuận thường xuyên đánh giá.
"""

# Gửi Request lên API của bạn
response = requests.post(
    "http://127.0.0.1:8000/summarize",
    json={"text": tai_lieu} # Thư viện requests sẽ tự động chuẩn hóa JSON cho bạn
)

import json

if response.status_code == 200:
    # Parse kết quả thành dạng Dictionary của Python
    data = response.json()
    
    # Mở file summary_result.json và ghi dữ liệu vào
    with open("summary_result.json", "w", encoding="utf-8") as f:
        # ensure_ascii=False giúp lưu tiếng Việt không bị lỗi font, indent=4 giúp format đẹp dễ nhìn
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print("Tuyệt vời! Kết quả đã được lưu thành công vào file 'summary_result.json'.")
else:
    print("Lỗi hệ thống! Status Code:", response.status_code)
    print("Chi tiết lỗi:", response.text)

