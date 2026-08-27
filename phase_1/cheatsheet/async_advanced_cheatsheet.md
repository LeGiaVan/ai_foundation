# Cheatsheet: 3 Kỹ Thuật Async Nâng Cao

---

## 1. AsyncGroq — Không chặn Event Loop

### Vấn đề
Khi bạn gọi một hàm **blocking** (đồng bộ) bên trong `async def`,
toàn bộ Event Loop bị đóng băng. Không request nào khác được xử lý
trong lúc đó.

```
Timeline (SAI - dùng Groq sync):
Request A --> [=== chờ Groq 3 giây ===] --> done
Request B -->                      (bị chặn, phải đợi A xong mới chạy!)

Timeline (ĐÚNG - dùng AsyncGroq):
Request A --> [chờ Groq...]
Request B -->     [chờ Groq...]   (chạy song song, không bị chặn!)
```

### Cú pháp

```python
# ❌ SAI: Dùng Groq đồng bộ trong async def
from groq import Groq
self.client = Groq(api_key="...")

async def generate_text(self, prompt: str) -> str:
    response = self.client.chat.completions.create(...)  # BLOCKING!
    return response.choices[0].message.content


# ✅ ĐÚNG Cách 1: Dùng AsyncGroq (khuyên dùng)
from groq import AsyncGroq
self.client = AsyncGroq(api_key="...")

async def generate_text(self, prompt: str) -> str:
    response = await self.client.chat.completions.create(...)  # Non-blocking!
    return response.choices[0].message.content


# ✅ ĐÚNG Cách 2: Đẩy hàm sync xuống thread riêng
import asyncio

async def generate_text(self, prompt: str) -> str:
    # asyncio.to_thread chạy hàm sync trong Thread riêng,
    # không làm nghẽn Event Loop chính
    response = await asyncio.to_thread(
        self.client.chat.completions.create,
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

---

## 2. asyncio.Semaphore — Giới hạn Request Đồng Thời

### Vấn đề
Groq (và mọi LLM API) đều có **Rate Limit** (VD: 30 req/phút).
Nếu bạn dùng `asyncio.gather` để gửi 100 request cùng lúc,
Groq sẽ chặn với lỗi `429 Too Many Requests`.

```
Không có Semaphore:
gather(100 tasks) --> 100 requests cùng lúc --> 💥 429!

Có Semaphore(5):
gather(100 tasks) --> 5 chạy --> 5 xong --> 5 tiếp --> ... --> an toàn ✅
```

### Cú pháp

```python
import asyncio

# Semaphore = "cái cổng" chỉ cho phép N người vào cùng lúc
semaphore = asyncio.Semaphore(5)

async def call_with_limit(prompt: str) -> str:
    async with semaphore:  # Xin vào cổng, nếu đủ chỗ thì vào, không thì đợi
        result = await generate_text(prompt)
        return result
    # Ra khỏi `with` = rời cổng, nhường chỗ cho người đang đợi

# Dù có 100 tasks, nhưng mọi lúc cũng chỉ có 5 cái chạy song song
tasks = [call_with_limit(p) for p in prompts]
results = await asyncio.gather(*tasks)
```

### Áp dụng vào Capstone

```python
class DocumentProcessor:
    MAX_CONCURRENT = 5

    async def summarize_long_document(self, text: str) -> DocumentSummary:
        chunks = chunk_text(text, "gpt-3.5-turbo")
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)

        async def map_with_limit(prompt: str) -> str:
            async with semaphore:
                return await self.llm_client.generate_text(prompt)

        map_prompts = [MAP_PROMPT_TEMPLATE.format(text=c) for c in chunks]
        map_results = await asyncio.gather(
            *[map_with_limit(p) for p in map_prompts],
            return_exceptions=True  # Kết hợp luôn với kỹ thuật số 3!
        )
```

---

## 3. gather(return_exceptions=True) — Xử Lý Lỗi Một Phần

### Vấn đề

```python
# Hành vi MẶC ĐỊNH (return_exceptions=False):
results = await asyncio.gather(task1, task2, task3_lỗi, task4)
# task3 bị lỗi --> Exception lan ra ngoài --> task4 bị hủy
# Kết quả: 💥 crash, mất hết kết quả của task1 và task2

# Với return_exceptions=True:
results = await asyncio.gather(task1, task2, task3_lỗi, task4,
                               return_exceptions=True)
# Kết quả: ["ok1", "ok2", ConnectionError("..."), "ok4"]
# Trả về list đầy đủ, không crash!
```

### Pattern xử lý: Tách valid / failed rồi Retry

```python
map_results = await asyncio.gather(*map_tasks, return_exceptions=True)

valid_results = []
failed_prompts = []

for i, result in enumerate(map_results):
    if isinstance(result, Exception):
        print(f"[WARN] Chunk {i} lỗi: {result}. Sẽ retry...")
        failed_prompts.append(map_prompts[i])  # Lưu lại prompt để retry
    else:
        valid_results.append(result)

# Retry những cái bị lỗi
if failed_prompts:
    retry_tasks = [self.llm_client.generate_text(p) for p in failed_prompts]
    retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
    # Chỉ lấy những cái thành công trong lần retry
    valid_results.extend(r for r in retry_results if not isinstance(r, Exception))

# Tiếp tục với valid_results đã đầy đủ nhất có thể
reduce_prompt = REDUCE_PROMPT_TEMPLATE.format(text="\n\n---\n\n".join(valid_results))
```

---

## Tổng kết

| Vấn đề | Triệu chứng | Giải pháp |
|---|---|---|
| Blocking call trong `async def` | Server đơ khi có 1 request đang chạy | `AsyncGroq` hoặc `asyncio.to_thread` |
| Quá nhiều request cùng lúc | Lỗi `429 Too Many Requests` | `asyncio.Semaphore(N)` |
| 1 task lỗi hủy tất cả | Crash dù 4/5 chunk thành công | `gather(..., return_exceptions=True)` |
