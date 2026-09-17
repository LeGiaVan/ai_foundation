import sys
import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

url = "http://127.0.0.1:8000/ask"
payload = {
    "question": "23*59 bằng bao nhiêu"
}

print("Đang gửi câu hỏi tới Agent...")
response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()
    print("\n🤖 CÂU TRẢ LỜI CUỐI:")
    print(data["final_answer"])
    print("\n⚙️  CHAIN OF THOUGHT (Các bước suy luận & công cụ):")
    for step in data["chain_of_thought"]:
        if step.get("thought"):
            print(f"🧠 Suy nghĩ (Thought): {step['thought']}")
        print(f"⚡ Công cụ (Action): {step['tool_name']}")
        print(f"📥 Tham số (Action Input): {step['tool_input']}")
        print(f"👁️ Kết quả (Observation): {step['tool_output']}\n")
else:
    print(f"Lỗi: {response.status_code}")
    print(response.text)

