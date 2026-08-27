# Cheatsheet: Abstract Base Class (ABC) trong Python

## 1. Vấn đề ABC giải quyết là gì?

Giả sử bạn đang xây dựng một hệ thống thanh toán. Bạn muốn đảm bảo rằng bất kỳ
ai tạo ra một "cổng thanh toán" mới (VNPay, Momo, Stripe...) đều PHẢI implement
đầy đủ các hàm cần thiết, không được bỏ sót.

**Không dùng ABC (Dễ bị bug âm thầm):**
```python
class BasePayment:
    def pay(self, amount): pass       # Chỉ có `pass`
    def refund(self, amount): pass    # Chỉ có `pass`

class VNPayGateway(BasePayment):
    def pay(self, amount):
        print(f"VNPay: Thanh toán {amount} VND")
    # ❌ Quên implement hàm refund!
    # Python KHÔNG báo lỗi, âm thầm trả về None khi gọi refund()

gateway = VNPayGateway()
gateway.refund(100_000)  # Chạy "thành công" nhưng không làm gì cả!
```

**Dùng ABC (Bắt lỗi sớm, ngay khi khởi tạo object):**
```python
from abc import ABC, abstractmethod

class BasePayment(ABC):              # Bước 1: Kế thừa từ ABC
    
    @abstractmethod                  # Bước 2: Đánh dấu hàm bắt buộc
    def pay(self, amount: int): ...
    
    @abstractmethod
    def refund(self, amount: int): ...

class VNPayGateway(BasePayment):
    def pay(self, amount: int):
        print(f"VNPay: Thanh toán {amount} VND")
    # ❌ Quên implement hàm refund!

gateway = VNPayGateway()
# 💥 TypeError: Can't instantiate abstract class VNPayGateway
#    with abstract method refund
# => Python bắt lỗi NGAY LẬP TỨC, không chờ đến lúc gọi hàm!
```

---

## 2. Cú pháp đầy đủ

```python
from abc import ABC, abstractmethod
from typing import Type

# --- ĐỊNH NGHĨA LỚP CHA (Interface) ---
class Animal(ABC):
    
    # Hàm abstract: Class con BẮT BUỘC phải implement
    @abstractmethod
    def speak(self) -> str: ...
    
    @abstractmethod
    def move(self) -> str: ...
    
    # Hàm bình thường: Class con KHÔNG BẮT BUỘC, có thể dùng luôn
    def describe(self):
        print(f"Tôi kêu: {self.speak()} và tôi {self.move()}")


# --- IMPLEMENT (Class Con) ---
class Dog(Animal):
    def speak(self) -> str:
        return "Gâu gâu"
    
    def move(self) -> str:
        return "chạy bằng 4 chân"

class Bird(Animal):
    def speak(self) -> str:
        return "Chip chip"
    
    def move(self) -> str:
        return "bay bằng cánh"


# --- SỬ DỤNG ---
dog = Dog()
dog.describe()   # ✅ Output: Tôi kêu: Gâu gâu và tôi chạy bằng 4 chân

bird = Bird()
bird.describe()  # ✅ Output: Tôi kêu: Chip chip và tôi bay bằng cánh

# Không thể tạo object từ class cha trực tiếp:
animal = Animal()  # 💥 TypeError!
```

---

## 3. Áp dụng vào project Capstone của bạn

```python
from abc import ABC, abstractmethod
from typing import Type
from pydantic import BaseModel

class BaseLLMClient(ABC):
    """
    Đây là bản "hợp đồng" (contract) cho mọi LLM Client.
    Bất kỳ ai implement class này đều phải cung cấp đủ 3 hàm dưới đây.
    """

    @abstractmethod
    async def generate_structured(self, prompt: str, schema: Type[BaseModel]) -> BaseModel: ...

    @abstractmethod
    async def generate_text(self, prompt: str) -> str: ...

    @abstractmethod
    async def stream_chat(self, system_prompt: str, user_prompt: str): ...


# GroqClient phải implement đủ 3 hàm trên, nếu không Python báo lỗi ngay!
class GroqClient(BaseLLMClient):
    async def generate_structured(self, ...): ...
    async def generate_text(self, ...): ...
    async def stream_chat(self, ...): ...


# Lợi ích: Sau này muốn dùng OpenAI thay thế Groq,
# chỉ cần tạo OpenAIClient kế thừa BaseLLMClient là xong,
# không cần đụng vào DocumentProcessor hay main.py!
class OpenAIClient(BaseLLMClient):
    async def generate_structured(self, ...): ...
    async def generate_text(self, ...): ...
    async def stream_chat(self, ...): ...
```

---

## 4. Tổng kết

| Đặc điểm | Class thường | ABC |
|---|---|---|
| Tạo object trực tiếp | ✅ Được | ❌ Báo lỗi ngay |
| Quên implement hàm con | ✅ Chạy, trả về None | ❌ Báo lỗi ngay |
| Mục đích | Tái sử dụng code | Định nghĩa "hợp đồng" |
| Hay gặp ở đâu | Logic thông thường | Plugin, Strategy Pattern, DI |

> **Câu hỏi để nhớ lâu:** ABC giống như một bản **hợp đồng lao động**.
> Nó không tự làm việc, nhưng nó đảm bảo rằng người được thuê (class con)
> phải thực hiện đầy đủ các điều khoản (implement đủ hàm) trước khi "được đi làm".
