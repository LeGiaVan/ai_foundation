# import asyncio

# async def task(name: str, seconds: int) -> str:
#     await asyncio.sleep(seconds)   # tạm dừng ở đây
#     return f"{name} xong"

# async def main():
#     results = await asyncio.gather(
#         task("A", 2),
#         task("B", 2),
#         task("C", 2),
#     )
#     print(results)   # ~2 giây (cùng lúc), không phải 6 giây (tuần tự)

"""
Event Loop là một vòng lặp vô tận, được kích hoạt ghi gọi asyncio.run(), có chức năng như sau:
Vòng 1:
  - Kiểm tra A: "A xong chưa?" → Chưa, A còn chờ 2 giây. OK, ghi chú "quay lại A sau"
  - Kiểm tra B: "B xong chưa?" → Chưa, B còn chờ 2 giây. OK, ghi chú "quay lại B sau"
  - Kiểm tra C: "C xong chưa?" → Chưa, C còn chờ 2 giây. OK, ghi chú "quay lại C sau"

Vòng 2 (sau 1 giây):
  - Kiểm tra A: "A xong chưa?" → Chưa (chỉ mới chờ 1 giây, còn 1 giây nữa)
  - Kiểm tra B: "B xong chưa?" → Chưa
  - Kiểm tra C: "C xong chưa?" → Chưa

Vòng 3 (sau 2 giây):
  - Kiểm tra A: "A xong chưa?" → XONGp rồi! Lấy kết quả, xóa A khỏi danh sách chờ
  - Kiểm tra B: "B xong chưa?" → XONG rồi! 
  - Kiểm tra C: "C xong chưa?" → XONG rồi!
  - Không còn gì để chờ → thoát khỏi vòng lặp, chương trình kết thúc
"""

# import asyncio
# import time

# def slow_task(name: str, seconds: int) -> str:
#     """Hàm sync, block (tốn CPU/I/O)"""
#     print(f"[{name}] Bắt đầu, chờ {seconds} giây")
#     time.sleep(seconds)   # giả lập công việc tốn thời gian
#     print(f"[{name}] Xong")
#     return f"{name} hoàn thành"

# async def main():
#     loop = asyncio.get_event_loop()
    
#     # Chạy slow_task trên luồng riêng
#     result = await loop.run_in_executor(None, slow_task, "Task 1", 2)
#     # executor — "ai sẽ chạy công việc" (Thread pool hay Process pool). None = dùng default (ThreadPoolExecutor)
#     # function — hàm sync muốn chạy
#     # *args — tham số truyền cho hàm
#     print(f"Kết quả: {result}")

# # asyncio.run(main()): Kết quả mất ~2s

# async def sequential():
#     loop = asyncio.get_event_loop()
    
#     # Chạy lần lượt
#     r1 = await loop.run_in_executor(None, slow_task, "Task A", 2)
#     r2 = await loop.run_in_executor(None, slow_task, "Task B", 2)
#     r3 = await loop.run_in_executor(None, slow_task, "Task C", 2)
    
#     return [r1, r2, r3]

# # asyncio.run(sequential()) Mất tổng cộng ~6s

# async def parallel():
#     loop = asyncio.get_event_loop()
    
#     # Chạy cùng lúc (dùng gather)
#     results = await asyncio.gather(
#         loop.run_in_executor(None, slow_task, "Task A", 2),
#         loop.run_in_executor(None, slow_task, "Task B", 2),
#         loop.run_in_executor(None, slow_task, "Task C", 2),
#     )
    
#     return results

# # asyncio.run(parallel()) Mất tổng cộng ~2s

# async def many_tasks():
#     loop = asyncio.get_event_loop()
    
#     tasks = [
#         loop.run_in_executor(None, slow_task, f"Task {i}", 1)
#         for i in range(5)
#     ]
    
#     results = await asyncio.gather(*tasks)
#     return results

# # import time
# # start = time.time()
# # asyncio.run(many_tasks())
# # print(f"Tổng thời gian: {time.time() - start:.1f} giây")

# import asyncio
# import time
# from concurrent.futures import ThreadPoolExecutor

# def slow_task(name: str, seconds: int) -> str:
#     print(f"[{name}] Bắt đầu")
#     time.sleep(seconds)
#     print(f"[{name}] Xong")
#     return f"{name} hoàn thành"

# async def main():
#     # Tạo thread pool với 2 luồng
#     executor = ThreadPoolExecutor(max_workers=4)
#     loop = asyncio.get_event_loop()
    
#     # Chạy 4 task, nhưng chỉ có 2 luồng → phải xếp hàng
#     tasks = [
#         loop.run_in_executor(executor, slow_task, f"Task {i}", 1)
#         for i in range(4)
#     ]
    
#     results = await asyncio.gather(*tasks)
#     print(results)
    
#     executor.shutdown(wait=True)  # dọn dẹp thread pool

# start = time.time()
# asyncio.run(main())
# print(f"Tổng thời gian: {time.time() - start:.1f} giây")

# from asyncio import base_events
# from fastapi import FastAPI
# import asyncio
# import time

# app = FastAPI()

# def heavy_computation(n: int) -> int:
#     """Hàm sync, tốn CPU"""
#     print(f"Bắt đầu tính toán {n}...")
#     time.sleep(2)  # giả lập
#     return n * 2

# @app.get("/slow")
# async def slow_endpoint(n: int = 5):
#     """Endpoint này gọi hàm sync, KHÔNG dùng run_in_executor (TỆ)"""
#     start = time.time()
#     result = heavy_computation(n)   # BLOCK event loop!
#     total_time = time.time() - start
#     return {"result": result, "time_taken": f"{total_time:.1f} giây"}


# @app.get("/fast")
# async def fast_endpoint(n: int = 5):
#     """Endpoint này dùng run_in_executor (ĐÚNG)"""
#     start = time.time()
#     loop = asyncio.get_event_loop()
#     result = await loop.run_in_executor(None, heavy_computation, n)
#     total_time = time.time() - start
#     return {"result": result, "time_taken": f"{total_time:.1f} giây"}

from fastapi import FastAPI
import asyncio
import time

app = FastAPI()

async def fetch_image_from_api(url: str) -> str:
    """Gọi API lấy ảnh (async)"""
    print(f"Fetching từ {url}...")
    await asyncio.sleep(1)  # giả lập: gọi API mất 1 giây
    return "image_data_base64"

def process_image_sync(image_data: str) -> str:
    """Xử lý ảnh (sync, CPU-heavy)"""
    print("Bắt đầu xử lý ảnh...")
    time.sleep(2)  # giả lập: xử lý mất 2 giây
    return f"Processed: {image_data[:20]}..."

@app.get("/analyze-image")
async def analyze_image(url: str):
    loop = asyncio.get_event_loop()
    
    # 1. Fetch ảnh (async)
    image = await fetch_image_from_api(url)
    
    # 2. Xử lý ảnh (sync, chạy trên thread pool)
    processed = await loop.run_in_executor(None, process_image_sync, image)
    
    return {"processed": processed}