"""
Minh hoa co che "tool calling" o muc protocol tho nhat - KHONG dung LangChain,
de thay ro: LLM chi sinh ra JSON, con code Python thuong moi la ben THUC THI tool.

Chay duoc ngay (MOCK_MODE = True) khong can API key, de ban thay co che truoc.
Doi MOCK_MODE = False va cai `pip install anthropic --break-system-packages`
roi dat bien moi truong ANTHROPIC_API_KEY de chay voi model that.
"""

import inspect
import re
from datetime import datetime
from typing import Callable, get_type_hints

MOCK_MODE = True


# ============================================================
# BUOC 1: "@tool" toi gian - tu sinh JSON schema tu ham Python
#          (day chinh la thu @tool cua LangChain lam ben trong)
# ============================================================

TOOL_REGISTRY: dict[str, Callable] = {}   # ten (string) -> ham Python that
TOOL_SCHEMAS: list[dict] = []             # schema gui len LLM, KHONG chua code


def tool(func: Callable) -> Callable:
    """Decorator toi gian: dang ky ham + tu sinh schema tu type hints + docstring."""
    name = func.__name__
    doc = (func.__doc__ or "").strip()
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
    properties, required = {}, []
    for param_name, param in sig.parameters.items():
        py_type = hints.get(param_name, str)
        properties[param_name] = {"type": type_map.get(py_type, "string")}
        if param.default is inspect._empty:
            required.append(param_name)

    schema = {
        "name": name,
        "description": doc,
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }

    TOOL_REGISTRY[name] = func   # <-- so tra cuu: ten string -> ham that
    TOOL_SCHEMAS.append(schema)
    return func


# ============================================================
# BUOC 2: Dinh nghia vai tool that (giong tinh than code cua ban)
# ============================================================

@tool
def calculate(expression: str) -> str:
    """Tinh toan bieu thuc toan hoc don gian. Vi du: '50 + 75'."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


@tool
def get_current_datetime() -> str:
    """Lay ngay gio he thong hien tai."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# BUOC 3: Ham goi LLM - day la "hop den" model, boc lai de thay
#          ro input/output cua no CHI la JSON, khong hon
# ============================================================

def call_llm_real(messages: list[dict]) -> dict:
    """Goi API that (Anthropic). Tu dien key de chay."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=TOOL_SCHEMAS,   # <-- gui SCHEMA, khong gui code ham that
        messages=messages,
    )
    return response.model_dump()


def call_llm_mock(messages: list[dict]) -> dict:
    """
    MOCK de demo co che ma khong can API key.
    Gia lap: model "quyet dinh" goi calculate neu thay phep tinh trong cau hoi,
    nguoc lai tra loi truc tiep. Day CHINH LA viec model that lam - chi khac
    la model that dung trong so (weights) de quyet dinh, con o day ta hard-code
    logic quyet dinh do de nhin thay ro no dang lam gi.
    """
    last_msg = messages[-1]["content"]

    if isinstance(last_msg, str) and any(op in last_msg for op in "+-*/"):
        match = re.search(r"[\d\.\+\-\*/\s]{3,}", last_msg)
        expr = match.group().strip() if match else "0"
        return {
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Nguoi dung hoi phep tinh, can goi calculate."},
                {"type": "tool_use", "id": "call_1", "name": "calculate", "input": {"expression": expr}},
            ],
        }

    return {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": f"(mock) Day la cau tra loi cuoi cho: {last_msg}"}],
    }


call_llm = call_llm_mock if MOCK_MODE else call_llm_real


# ============================================================
# BUOC 4: Vong lap Agent - noi "thuc thi tool" THAT SU xay ra
# ============================================================

def run_agent(user_question: str, max_turns: int = 5) -> str:
    messages = [{"role": "user", "content": user_question}]

    for turn in range(max_turns):
        print(f"\n--- Luot {turn + 1}: gui {len(messages)} messages len LLM ---")
        response = call_llm(messages)

        # Model chi TRA VE JSON - chua co gi duoc thuc thi o day
        assistant_content = response["content"]
        messages.append({"role": "assistant", "content": assistant_content})

        if response["stop_reason"] != "tool_use":
            final_text = next(b["text"] for b in assistant_content if b["type"] == "text")
            return final_text

        # ---------- DAY LA DOAN "THUC THI TOOL" THAT SU ----------
        tool_results = []
        for block in assistant_content:
            if block["type"] != "tool_use":
                continue

            name, args, call_id = block["name"], block["input"], block["id"]
            print(f"  -> LLM yeu cau goi: {name}({args})")

            # (a) TRA DICT: ten string -> ham Python that
            func = TOOL_REGISTRY.get(name)

            # (b) GOI HAM THAT - dong nay la luc code thuc su chay
            output = func(**args) if func else f"Error: unknown tool {name}"

            print(f"  <- Ket qua that: {output}")
            tool_results.append({"type": "tool_result", "tool_use_id": call_id, "content": str(output)})
        # -----------------------------------------------------------

        # Dua ket qua tool tro lai lam message tiep theo, quay lai dau vong lap
        messages.append({"role": "user", "content": tool_results})

    return "Da vuot qua so luot toi da ma chua co cau tra loi cuoi."


if __name__ == "__main__":
    print("\n>>> Cau hoi 1: co phep tinh, se kich hoat tool_use")
    print(run_agent("Tinh giup toi 50 + 75 la bao nhieu?"))

    print("\n" + "=" * 60)
    print(">>> Cau hoi 2: khong co phep tinh, tra loi truc tiep")
    print(run_agent("Xin chao, ban khoe khong?"))