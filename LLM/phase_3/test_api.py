import requests

url = "http://127.0.0.1:8000/ask"
payload = {
    "question": "Hôm nay là ngày mấy? YOLOv11n xử lý object detection mất bao lâu? Nếu nhân đôi thời gian đó lên thì là bao nhiêu?"
}

print("Đang gửi câu hỏi tới Agent...")
response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()
    print("\n🤖 CÂU TRẢ LỜI CUỐI:")
    print(data["final_answer"])
    print("\n⚙️  CHAIN OF THOUGHT (Các công cụ đã dùng):")
    for step in data["chain_of_thought"]:
        print(f"- Đã gọi hàm: {step['tool_name']}")
        print(f"  Input: {step['tool_input']}")
        print(f"  Output: {step['tool_output']}")
else:
    print(f"Lỗi: {response.status_code}")
    print(response.text)
