# 🧠 Tuần 3 — Prompt Engineering + Advanced API Features (Cheatsheet)

---

## Ngày 6: Few-Shot Prompting (Dạy AI bằng ví dụ mẫu)

### 6.1. Ba cấp độ Prompting

| Cấp độ | Ý nghĩa | Khi nào dùng |
|---|---|---|
| **Zero-shot** | Chỉ mô tả yêu cầu bằng lời, KHÔNG cho ví dụ | Câu hỏi đơn giản, AI đã biết cách làm sẵn |
| **One-shot** | Cho AI xem **1 ví dụ** mẫu (input → output) | Cần AI hiểu sơ định dạng bạn muốn |
| **Few-shot** (Multishot) | Cho AI xem **2-5 ví dụ** mẫu (input → output) | Cần AI bắt chước chính xác phong cách/định dạng |

### 6.2. Tại sao cần Few-shot?

Có những lúc bạn mô tả bằng lời rất khó truyền đạt chính xác điều bạn muốn. Ví dụ:
- Bạn muốn AI phân loại cảm xúc và chỉ trả đúng 1 từ viết hoa (`TÍCH_CỰC`, `TIÊU_CỰC`, `TRUNG_LẬP`).
- Nếu chỉ nói bằng lời (Zero-shot), AI có thể trả lời loạn xạ: lúc thì `Tích cực`, lúc thì `positive`, lúc thì viết hẳn một đoạn giải thích dài dòng.
- Nhưng nếu bạn **cho nó xem 3 ví dụ mẫu**, nó sẽ "bắt chước" y chang định dạng đó. Đây là sức mạnh của Few-shot.

### 6.3. Ví dụ code: Zero-shot vs Few-shot

**Zero-shot (Không có ví dụ):**
```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=100,
    system="Bạn là AI phân loại cảm xúc bình luận. Chỉ trả lời bằng 1 từ viết hoa: TÍCH_CỰC, TIÊU_CỰC, hoặc TRUNG_LẬP.",
    messages=[
        {"role": "user", "content": "Sản phẩm tệ quá, không bao giờ mua lại."}
    ]
)
# Kết quả có thể là: "TIÊU_CỰC" (đúng)
# Nhưng cũng có thể là: "Đây là bình luận tiêu cực vì..." (sai định dạng!)
```

**Few-shot (Có 3 ví dụ mẫu):**
```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=100,
    system="Bạn là AI phân loại cảm xúc bình luận. Chỉ trả lời đúng 1 từ.",
    messages=[
        # Ví dụ mẫu 1
        {"role": "user", "content": "Giao hàng nhanh, đóng gói cẩn thận, rất hài lòng!"},
        {"role": "assistant", "content": "TÍCH_CỰC"},
        
        # Ví dụ mẫu 2
        {"role": "user", "content": "Hàng bị lỗi, liên hệ không ai trả lời."},
        {"role": "assistant", "content": "TIÊU_CỰC"},
        
        # Ví dụ mẫu 3
        {"role": "user", "content": "Sản phẩm bình thường, không có gì đặc biệt."},
        {"role": "assistant", "content": "TRUNG_LẬP"},
        
        # Câu hỏi thật (cần AI phân loại)
        {"role": "user", "content": "Sản phẩm tệ quá, không bao giờ mua lại."}
    ]
)
# Kết quả gần như chắc chắn: "TIÊU_CỰC" (đúng định dạng 100%)
```

### 6.4. Mẹo khi dùng Few-shot

1. **Đa dạng ví dụ:** Nên cho ví dụ đại diện cho TẤT CẢ các trường hợp (Tích cực, Tiêu cực, Trung lập). Nếu bạn chỉ cho 3 ví dụ Tích cực, AI sẽ thiên vị trả Tích cực cho mọi thứ.
2. **Đặt ví dụ mẫu trong System Prompt hoặc đầu Messages:** Vì phần này ít thay đổi nên sẽ ăn được **Prompt Caching** (Tiết kiệm tiền khi phân loại hàng ngàn bình luận).
3. **3-5 ví dụ là đủ.** Quá nhiều ví dụ sẽ tốn token mà không cải thiện thêm chất lượng.

---

## Ngày 7: Chain-of-Thought / CoT (Bắt AI suy nghĩ từng bước)

### 7.1. Bản chất của CoT

LLM sinh ra câu trả lời **từng token một, từ trái sang phải**. Nếu bạn hỏi một bài toán phức tạp và bắt nó trả lời ngay lập tức (chỉ 1 con số), nó phải "đoán" đáp án mà không có quá trình suy luận trung gian → rất dễ sai.

CoT giải quyết bằng cách: **Yêu cầu AI viết ra các bước suy luận trung gian trước, rồi mới đưa ra đáp án cuối cùng.** Giống như khi bạn làm bài kiểm tra Toán, giáo viên bắt phải "trình bày bài giải" chứ không được ghi mỗi đáp số.

### 7.2. Khi nào CẦN dùng CoT?

| Loại bài toán | Cần CoT? | Lý do |
|---|---|---|
| Toán logic, suy luận nhiều bước | ✅ CẦN | AI dễ sai nếu không được "suy nghĩ" |
| Phân tích ưu/nhược điểm | ✅ CẦN | Cần cân nhắc nhiều yếu tố |
| Phân loại cảm xúc (1 từ) | ❌ KHÔNG | Đơn giản, CoT chỉ lãng phí token |
| Dịch thuật | ❌ KHÔNG | AI đã giỏi sẵn, không cần suy luận |

### 7.3. Ví dụ code: Không CoT vs Có CoT

**Không CoT (Hỏi thẳng):**
```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=500,
    messages=[{
        "role": "user",
        "content": """Một cửa hàng bán áo với giá 200.000đ/chiếc. 
        Nếu mua từ 3 chiếc trở lên được giảm 15%. 
        Thuế VAT là 10% (tính trên giá sau giảm).
        Hỏi: Mua 5 chiếc thì phải trả bao nhiêu? 
        Chỉ trả lời con số cuối cùng."""
    }]
)
# AI có thể đưa ra đáp án SAI vì phải tính quá nhiều bước trong đầu
```

**Có CoT (Bắt suy luận từng bước):**
```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    messages=[{
        "role": "user",
        "content": """Một cửa hàng bán áo với giá 200.000đ/chiếc. 
        Nếu mua từ 3 chiếc trở lên được giảm 15%. 
        Thuế VAT là 10% (tính trên giá sau giảm).
        Hỏi: Mua 5 chiếc thì phải trả bao nhiêu?
        
        Hãy suy nghĩ từng bước trước khi đưa ra đáp án cuối cùng.
        Đặt quá trình suy luận trong thẻ <thinking>...</thinking>.
        Đặt đáp án cuối cùng trong thẻ <answer>...</answer>."""
    }]
)
# AI sẽ trả về:
# <thinking>
# Bước 1: Giá gốc 5 chiếc = 5 x 200.000 = 1.000.000đ
# Bước 2: Giảm 15% → 1.000.000 x 0.85 = 850.000đ
# Bước 3: Thuế VAT 10% → 850.000 x 1.10 = 935.000đ
# </thinking>
# <answer>935.000đ</answer>
```

### 7.4. Tách riêng phần suy luận và đáp án trong code

```python
import re

ai_text = response.content[0].text

# Lấy phần suy luận (để debug hoặc log)
thinking = re.search(r"<thinking>(.*?)</thinking>", ai_text, re.DOTALL)

# Lấy phần đáp án (để trả về cho User)
answer = re.search(r"<answer>(.*?)</answer>", ai_text, re.DOTALL)

if answer:
    final_answer = answer.group(1).strip()
    print(f"Đáp án: {final_answer}")
```

**Lợi ích của việc tách:** Bạn có thể chỉ trả phần `<answer>` về cho khách hàng (gọn gàng), còn phần `<thinking>` thì lưu vào log để debug khi AI sai.

---

## Ngày 8: Structured Output / JSON Mode (Ép AI trả JSON chuẩn)

### 8.1. Vấn đề cần giải quyết

Khi bạn yêu cầu AI "trả lời bằng JSON", nó có thể trả về:
- JSON đúng: `{"name": "iPhone 15", "price": 25000000}`
- JSON sai: ` ```json\n{"name": "iPhone 15"}\n``` ` (bọc trong markdown code block)
- Hoàn toàn không phải JSON: `"Sản phẩm là iPhone 15 với giá 25 triệu"`

**Giải pháp của Anthropic:** Dùng **Tool Use (Function Calling)** để ép AI trả đúng cấu trúc 100%.

### 8.2. Cách hoạt động: "Lừa" AI bằng Tool Use

Ý tưởng cực kỳ thông minh: Bạn không thực sự muốn AI "gọi một công cụ". Bạn chỉ lợi dụng cơ chế Tool Use để ép AI trả dữ liệu theo đúng JSON Schema mà bạn định nghĩa.

**Bước 1: Định nghĩa Pydantic Model (Schema)**
```python
from pydantic import BaseModel

class ProductInfo(BaseModel):
    name: str           # Tên sản phẩm
    price: float        # Giá (VNĐ)
    category: str       # Danh mục (Điện thoại, Laptop, Phụ kiện...)
```

**Bước 2: Gọi Claude với Tool Use**
```python
import anthropic
import json

client = anthropic.Anthropic()

# Định nghĩa "công cụ ảo" với schema lấy từ Pydantic
extract_tool = {
    "name": "extract_product_info",
    "description": "Trích xuất thông tin sản phẩm từ đoạn văn bản mô tả.",
    "input_schema": ProductInfo.model_json_schema()
    # Pydantic tự động sinh ra JSON Schema chuẩn:
    # {"type": "object", "properties": {"name": {"type": "string"}, ...}, "required": [...]}
}

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[extract_tool],
    tool_choice={"type": "tool", "name": "extract_product_info"},  # Bắt buộc phải dùng tool này
    messages=[{
        "role": "user",
        "content": "iPhone 15 Pro Max 256GB chính hãng, giá chỉ 28 triệu 990 nghìn, freeship toàn quốc!"
    }]
)
```

**Bước 3: Parse kết quả bằng Pydantic**
```python
# Claude trả về dưới dạng tool_use block
tool_result = response.content[0]  # type: "tool_use"
raw_data = tool_result.input       # {"name": "iPhone 15 Pro Max 256GB", "price": 28990000, "category": "Điện thoại"}

# Validate và parse bằng Pydantic (Nếu sai schema sẽ báo lỗi rõ ràng)
try:
    product = ProductInfo(**raw_data)
    print(product.name)      # iPhone 15 Pro Max 256GB
    print(product.price)     # 28990000.0
    print(product.category)  # Điện thoại
except Exception as e:
    print(f"AI trả dữ liệu sai cấu trúc: {e}")
```

### 8.3. Tích hợp vào FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ExtractRequest(BaseModel):
    text: str  # Đoạn văn bản cần trích xuất

class ProductInfo(BaseModel):
    name: str
    price: float
    category: str

@app.post("/extract", response_model=ProductInfo)
async def extract_product(req: ExtractRequest):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[extract_tool],
        tool_choice={"type": "tool", "name": "extract_product_info"},
        messages=[{"role": "user", "content": req.text}]
    )
    
    raw_data = response.content[0].input
    
    try:
        product = ProductInfo(**raw_data)
        return product
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"AI trích xuất sai cấu trúc: {e}")
```

### 8.4. Khi nào dùng Tool Use vs Khi nào dùng Prompt thường?

| Tình huống | Cách tiếp cận |
|---|---|
| Cần JSON **chính xác tuyệt đối** để code xử lý tiếp | ✅ Dùng Tool Use (ép cứng Schema) |
| Chỉ cần AI trả lời dạng text có cấu trúc (markdown, bullet points) | Prompt thường là đủ |
| Cần gọi API/hàm bên ngoài dựa trên quyết định của AI | ✅ Dùng Tool Use (mục đích chính thống) |

---

## Ngày 9: Streaming Response (Trả lời "từng chữ một")

### 9.1. Vấn đề cần giải quyết

Khi bạn gọi `client.messages.create()`, FastAPI phải **đợi Claude suy nghĩ xong 100%** rồi mới trả toàn bộ kết quả cho User. Nếu câu trả lời dài 2000 chữ, User phải ngồi nhìn vòng tròn loading quay mòn mỏi 10-15 giây rồi BÙM hiện ra cả bài văn.

**Streaming** giải quyết bằng cách: Claude vừa nghĩ vừa gửi từng mảnh nhỏ (chunk) về. User sẽ thấy chữ hiện ra **từng từ một**, giống hệt trải nghiệm trên ChatGPT.

### 9.2. Streaming phía Claude (Nhận từng chunk từ AI)

```python
import anthropic

client = anthropic.Anthropic()

# Dùng .stream() thay vì .create()
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Viết một bài thơ về Hà Nội"}]
) as stream:
    for text in stream.text_stream:
        # 'text' là một mảnh nhỏ (vài chữ), được in ra NGAY KHI CLAUDE SINH RA
        print(text, end="", flush=True)
```

### 9.3. Streaming phía FastAPI (Chuyển tiếp chunk cho Client)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import anthropic

app = FastAPI()
client = anthropic.Anthropic()

async def generate_stream(user_message: str):
    """Generator: Nhận chunk từ Claude, lập tức yield ra cho Client"""
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_message}]
    ) as stream:
        for text in stream.text_stream:
            yield text  # Gửi từng mảnh nhỏ ra ngoài Client NGAY LẬP TỨC

@app.post("/chat/stream")
async def chat_stream(message: str):
    return StreamingResponse(
        generate_stream(message),
        media_type="text/plain"
    )
```

### 9.4. Cách test Streaming

**❌ KHÔNG dùng Swagger UI (`/docs`):** Swagger UI không hỗ trợ hiển thị streaming. Nó sẽ đợi toàn bộ response xong mới hiển thị (giống non-streaming).

**✅ Dùng `curl` với flag `-N`:**
```bash
curl -N -X POST "http://localhost:8000/chat/stream?message=Viết bài thơ về Sài Gòn"
```
Flag `-N` tắt buffering, cho phép bạn nhìn thấy từng chữ hiện ra dần dần trên terminal.

### 9.5. So sánh Non-streaming vs Streaming

| Tiêu chí | Non-streaming (`/chat`) | Streaming (`/chat/stream`) |
|---|---|---|
| **Thời gian thấy chữ đầu tiên (TTFB)** | 5-15 giây (phải chờ AI nghĩ xong) | < 1 giây (chữ đầu tiên hiện gần như ngay) |
| **Tổng thời gian nhận hết** | Giống nhau | Giống nhau |
| **Trải nghiệm User** | Chờ lâu → bùm hiện hết | Thấy chữ chạy dần → cảm giác "AI đang suy nghĩ" |
| **Độ phức tạp code** | Đơn giản (`return response`) | Phức tạp hơn (`StreamingResponse` + generator) |
| **Khi nào dùng** | API cho máy gọi máy (backend-to-backend) | Giao diện Chat cho người dùng cuối |

### 9.6. Kết hợp Streaming + Async (Phiên bản Production)

```python
# Dùng AsyncAnthropic để không block Event Loop (Quy tắc tử thần!)
client = anthropic.AsyncAnthropic()

async def generate_stream_async(user_message: str):
    async with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_message}]
    ) as stream:
        async for text in stream.text_stream:
            yield text

@app.post("/chat/stream")
async def chat_stream(message: str):
    return StreamingResponse(
        generate_stream_async(message),
        media_type="text/plain"
    )
```

**Lưu ý quan trọng:** Ở phiên bản Production, bạn phải dùng `AsyncAnthropic()` (bất đồng bộ) thay vì `Anthropic()` (đồng bộ). Nếu dùng bản đồng bộ trong hàm `async def`, bạn sẽ vi phạm **Quy tắc tử thần** (Block Event Loop), khiến tất cả User khác bị treo khi 1 User đang streaming.

---

## 📋 Tóm tắt nhanh Tuần 3

| Kỹ thuật | Mục đích | Keyword nhớ |
|---|---|---|
| **Few-shot** | Dạy AI bằng ví dụ mẫu → Output nhất quán | "Cho xem mẫu trước" |
| **Chain-of-Thought** | Bắt AI suy luận từng bước → Tăng độ chính xác | "Trình bày bài giải" |
| **Structured Output** (Tool Use) | Ép AI trả đúng JSON Schema → Code parse được | "Pydantic + Tool = JSON chuẩn" |
| **Streaming** | Trả lời từng chữ → UX mượt mà như ChatGPT | "`stream()` + `StreamingResponse`" |
