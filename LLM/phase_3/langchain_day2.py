import os
import json
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import (
    PydanticOutputParser, 
    JsonOutputParser
)
from langchain_core.exceptions import OutputParserException

# API Key của bạn
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

model = ChatGroq(model="groq/compound-mini", temperature=0)

print("="*50)
print("BÀI TẬP 1: JsonOutputParser")
json_parser = JsonOutputParser()
json_prompt = PromptTemplate(
    template="Trả lời câu hỏi sau. Định dạng JSON bắt buộc phải có 3 keys: 'answer', 'confidence', 'sources'.\nCâu hỏi: {query}\n\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)
json_chain = json_prompt | model | json_parser
res_json = json_chain.invoke({"query": "LangChain được viết bằng ngôn ngữ lập trình nào?"})
print("Kết quả JSON:", json.dumps(res_json, indent=2, ensure_ascii=False))
print("Kiểu dữ liệu nhận được:", type(res_json))

print("\n" + "="*50)
print("BÀI TẬP 2: PydanticOutputParser với ProductInfo")
class ProductInfo(BaseModel):
    name: str = Field(description="Tên sản phẩm")
    price: int = Field(description="Giá tiền của sản phẩm (số nguyên)")
    category: str = Field(description="Thuộc nhóm danh mục sản phẩm nào")
    in_stock: bool = Field(default=True, description="Còn hàng hay không, True/False")

pydantic_parser = PydanticOutputParser(pydantic_object=ProductInfo)
pydantic_prompt = PromptTemplate(
    template="Trích xuất thông tin sản phẩm từ đoạn mô tả sau:\n{query}\n\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
)
pydantic_chain = pydantic_prompt | model | pydantic_parser

text_desc = "Cửa hàng bán Laptop Dell XPS 13 cực đẹp, giá 25000000 VNĐ. Máy thuộc dòng laptop văn phòng. Hiện tại kho đang tạm hết hàng."
res_pydantic = pydantic_chain.invoke({"query": text_desc})
print("Tên:", res_pydantic.name)
print("Giá:", res_pydantic.price)
print("Loại:", res_pydantic.category)
print("Còn hàng:", res_pydantic.in_stock)
print("Kiểu dữ liệu nhận được:", type(res_pydantic))


print("\n" + "="*50)
print("BÀI TẬP 3: Lỗi 400 Bad Request với .with_structured_output()")
print("Thử gọi .with_structured_output() với model groq/compound-mini...")
try:
    structured_model = model.with_structured_output(ProductInfo)
    structured_model.invoke("Đánh giá sản phẩm test")
except Exception as e:
    print("=> Bắt được lỗi mong muốn khi dùng with_structured_output!")
    print("=> Tên lỗi:", type(e).__name__)
    print("=> Nội dung:", e)
    print("=> GIẢI THÍCH: Model 'groq/compound-mini' không có khả năng tool calling ở dưới nền. Vì thế ta phải dùng PydanticOutputParser để 'ép' model trả về JSON thông qua Prompt Engineering thay vì dùng hàm tiện ích này.")


print("\n" + "="*50)
print("BÀI TẬP 4: Implement Retry logic thủ công")
# Cố tình đưa prompt mơ hồ để LLM trả về text thường thay vì JSON
bad_prompt = PromptTemplate.from_template(
    "Nói về một chiếc điện thoại iPhone 16. (Lưu ý: Không format JSON, chỉ nói 1 câu bình thường)"
)
raw_chain = bad_prompt | model

print("=> Đang lấy output sai từ LLM...")
raw_response = raw_chain.invoke({})
print("Output gốc (bị sai format):", raw_response.content)

max_retries = 3
for attempt in range(max_retries):
    try:
        print(f"\n--- Cố gắng parse lần {attempt + 1} ---")
        # Thử parse output thô thành Pydantic (sẽ thất bại ở lần 1 vì nó là text thường)
        parsed_result = pydantic_parser.parse(raw_response.content)
        print("=> Tuyệt vời! Parse thành công JSON:")
        print(parsed_result)
        break
    except OutputParserException as e:
        print("=> Parse thất bại! LLM trả về sai format Pydantic.")
        if attempt == max_retries - 1:
            print("=> Đã hết số lần thử. Bỏ cuộc.")
        else:
            print("=> Đưa thông báo lỗi lại cho LLM để nó tự sửa...")
            # Retry logic: Đưa phần output lỗi + nội dung lỗi vào prompt yêu cầu LLM sửa lại
            retry_prompt = PromptTemplate.from_template(
                "Đầu ra trước đó của bạn bị lỗi không thể parse thành JSON:\n{raw_response}\n\nThông báo lỗi hệ thống là:\n{error}\n\nHãy sửa lại nội dung trên và trả về ĐÚNG định dạng sau (không chứa text dư thừa):\n{format_instructions}"
            )
            retry_chain = retry_prompt | model
            raw_response = retry_chain.invoke({
                "raw_response": raw_response.content,
                "error": str(e),
                "format_instructions": pydantic_parser.get_format_instructions()
            })