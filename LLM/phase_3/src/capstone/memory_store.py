import os
import json
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class MemoryStore:
    """
    Quản lý Long-Term Agentic Memory cho người dùng.
    Hỗ trợ:
    - Lưu trữ vĩnh viễn vào file JSON (hoặc Qdrant/SQLite).
    - Trích xuất thông tin người dùng từ hội thoại.
    - Tìm kiếm và nạp các memory phù hợp vào System Prompt.
    - Consolidate (tóm tắt / gộp) khi số lượng memory > 10.
    """
    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            storage_path = os.path.join(base_dir, "user_memories.json")
        self.storage_path = storage_path
        self._memories: dict[str, list[str]] = self._load()
        
        # LLM phục vụ trích xuất và tóm tắt memory
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            max_retries=2,
            timeout=30
        )

    def _load(self) -> dict[str, list[str]]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[MemoryStore] Lỗi đọc file memory: {e}, khởi tạo rỗng.")
                return {}
        return {}

    def _save_to_disk(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._memories, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[MemoryStore] Lỗi ghi file memory: {e}")

    def save(self, user_id: str, memory_text: str):
        """Lưu một mẩu thông tin/sở thích mới của user."""
        memory_text = memory_text.strip()
        if not memory_text:
            return
            
        if user_id not in self._memories:
            self._memories[user_id] = []
            
        # Tránh lưu trùng lặp
        if memory_text not in self._memories[user_id]:
            self._memories[user_id].append(memory_text)
            self._save_to_disk()
            print(f"[MemoryStore 💾] Đã lưu memory cho '{user_id}': {memory_text}")
            
        # Kiểm tra nếu vượt quá 10 mẩu thông tin thì tiến hành hợp nhất (consolidation)
        if len(self._memories[user_id]) > 10:
            self.consolidate(user_id)

    def retrieve(self, user_id: str, query: str = "") -> list[str]:
        """Lấy danh sách thông tin đã ghi nhớ về user."""
        user_mems = self._memories.get(user_id, [])
        if not user_mems:
            return []
        # Nếu query rỗng, trả về tất cả
        if not query:
            return list(user_mems)
            
        # Lọc cơ bản theo từ khóa hoặc trả về toàn bộ nếu ít
        q_lower = query.lower()
        matched = [m for m in user_mems if any(word in m.lower() for word in q_lower.split())]
        return matched if matched else list(user_mems)

    def consolidate(self, user_id: str):
        """
        Dùng LLM tóm tắt bớt và loại bỏ các memory dư thừa khi > 10 items.
        """
        memories = self._memories.get(user_id, [])
        if len(memories) <= 10:
            return
            
        print(f"[MemoryStore 🧠] Bắt đầu nén/tóm tắt {len(memories)} memories của user '{user_id}'...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là chuyên gia quản lý hồ sơ trí nhớ. Hãy gộp và tóm tắt danh sách các thông tin sau thành tối đa 5-7 gạch đầu dòng súc tích, giữ lại các sở thích, phong cách và thông tin quan trọng nhất. Trả về mỗi thông tin trên một dòng, bắt đầu bằng dấu gạch ngang '- '."),
            ("human", "Danh sách các thông tin:\n" + "\n".join(f"- {m}" for m in memories))
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({})
            lines = [line.lstrip("- ").strip() for line in response.content.strip().split("\n") if line.strip()]
            self._memories[user_id] = lines
            self._save_to_disk()
            print(f"[MemoryStore ✅] Đã tóm tắt thành công xuống còn {len(lines)} memories.")
        except Exception as e:
            print(f"[MemoryStore ⚠️] Lỗi tóm tắt memory: {e}")

    def extract_and_save(self, user_id: str, user_text: str):
        """
        Phân tích tin nhắn của user xem có thông tin cá nhân/sở thích nào cần nhớ lâu dài không.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Nhiệm vụ của bạn là phát hiện các thông tin cá nhân lâu dài của người dùng "
             "(ví dụ: nghề nghiệp, chuyên môn, sở thích trình bày, phong cách trả lời yêu thích, ràng buộc cá nhân).\n"
             "Nếu tin nhắn KHÔNG chứa thông tin cá nhân nào cần nhớ lâu dài, chỉ trả về chữ 'NONE'.\n"
             "Nếu CÓ, hãy tóm tắt từng điều dưới dạng danh sách ngắn gọn (mỗi dòng một ý bắt đầu bằng '- ')."),
            ("human", "{user_input}")
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({"user_input": user_text})
            content = response.content.strip()
            if content and "NONE" not in content.upper():
                lines = [line.lstrip("- ").strip() for line in content.split("\n") if line.strip()]
                for line in lines:
                    if line:
                        self.save(user_id, line)
        except Exception as e:
            print(f"[MemoryStore ⚠️] Lỗi trích xuất memory từ tin nhắn: {e}")
