import os
import time

# API Key của bạn
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Khởi tạo LLM và Parser dùng chung
llm = ChatGroq(model="groq/compound-mini", temperature=0)
parser = StrOutputParser()

print("="*50)
print("BÀI TẬP 1: PromptTemplate với 2 biến")
prompt1 = PromptTemplate.from_template("Kể một câu chuyện rất ngắn về {topic} bằng tiếng {language}.")
print("Output sau khi format:\n", prompt1.format(topic="mèo", language="Anh"))

print("\n" + "="*50)
print("BÀI TẬP 2: Xây chain đơn giản")
chat_prompt2 = ChatPromptTemplate.from_template("Viết một câu thơ ngắn về {topic}.")
chain2 = chat_prompt2 | llm | parser
print("Kết quả invoke:\n", chain2.invoke({"topic": "mùa thu"}))

print("\n" + "="*50)
print("BÀI TẬP 3: Thêm .stream()")
print("Streaming response (in từng chữ): ", end="")
for chunk in chain2.stream({"topic": "mùa đông"}):
    print(chunk, end="", flush=True)
print() # Xuống dòng

print("\n" + "="*50)
print("BÀI TẬP 4: .batch() vs .invoke() tuần tự")
topics = [{"topic": "mùa xuân"}, {"topic": "mùa hạ"}, {"topic": "mùa thu"}]

# Cách 1: Chạy tuần tự (.invoke)
start_invoke = time.time()
for t in topics:
    chain2.invoke(t)
time_invoke = time.time() - start_invoke
print(f"Thời gian chạy tuần tự (.invoke): {time_invoke:.2f} giây")

# Cách 2: Chạy song song (.batch)
start_batch = time.time()
chain2.batch(topics)
time_batch = time.time() - start_batch
print(f"Thời gian chạy song song (.batch): {time_batch:.2f} giây")

print("\n" + "="*50)
print("BÀI TẬP 5: Dùng RunnablePassthrough")
# Thay vì dùng retriever thực tế (sẽ gây lỗi NameError vì chưa định nghĩa), 
# ta giả lập context trả về một chuỗi kiến thức có sẵn bằng hàm lambda.
prompt5 = ChatPromptTemplate.from_template("Dựa vào thông tin sau: {context}\nHãy trả lời: {question}")

chain5 = (
    {
        "context": lambda x: "LangChain là một framework để xây dựng ứng dụng LLM.", 
        "question": RunnablePassthrough() # Pass thẳng input ("LangChain dùng để làm gì?") vào biến question
    }
    | prompt5
    | llm
    | parser
)

print("Kết quả RunnablePassthrough (RAG chain giả lập):")
print(chain5.invoke("LangChain dùng để làm gì?"))