# Phase 1 – Detailed Checklist & Self‑Assessment

---

## 1️⃣ Core Knowledge Areas (Expanded)

### 📦 Pydantic
- [ ] **Define a `BaseModel`** with typed fields and default values. Example:
  ```python
  class User(BaseModel):
      id: int
      name: str = Field(..., description="User's full name")
  ```
- [ ] **Use `Field`** to add constraints (min/max length, regex) and documentation. Verify via `User.schema()`.
- [ ] **Generate JSON Schema** with `model.schema_json()` and compare against expected OpenAPI output.
- [ ] **Validate nested models** and write custom validators using `@validator`. Example: email validation with regex and a nested address model:

```python
from pydantic import BaseModel, EmailStr, validator

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class User(BaseModel):
    email: EmailStr
    address: Address

    @validator('email')
    def email_must_be_company(cls, v):
        """Ensure email belongs to the company domain."""
        if not v.endswith('@example.com'):
            raise ValueError('Email must be from @example.com domain')
        return v
```
- [ ] **Parse external data** using `parse_obj` / `parse_raw` and handle `ValidationError` gracefully.

### 🏗 OOP (Python)
- [ ] **Explain inheritance** and demonstrate `super().__init__()` with a concrete example (e.g., `BaseRepository` → `QdrantRepository`).
- [ ] **Create an Abstract Base Class** (`abc.ABC`, `@abstractmethod`) for a service interface (e.g., `DocumentParser`). Show how concrete classes must implement required methods.
- [ ] **Choose composition vs inheritance** appropriately; illustrate with a `Service` that uses a `Repository` instance.
- [ ] **Encapsulate attributes** using `@property` and setters, adding validation logic.
- [ ] **Implement useful `__repr__` / `__str__`** for debugging complex objects (include class name and key fields).

### ⚡️ AsyncIO
- [ ] **Write `async def` functions** and always `await` I/O‑bound calls. Show a simple async file read with `aiofiles`.
- [ ] **Understand the event loop**: `asyncio.run(main())` vs `loop = asyncio.get_event_loop()`; explain when to use each.
- [ ] **Use `asyncio.gather`** to run multiple LLM calls concurrently. Include `return_exceptions=True` to collect errors without aborting the whole batch.
- [ ] **Apply `asyncio.Semaphore`** to limit parallel external calls (e.g., max 5 concurrent LLM requests). Provide a snippet.
- [ ] **Wrap synchronous libraries** (like `PyMuPDF`) with `run_in_executor` or use async equivalents (`httpx`, `aiofiles`).

### 🌐 FastAPI
- [ ] **Define routes** with proper HTTP verbs (`@app.get`, `@app.post`). Add path‑parameter examples.
- [ ] **Use Dependency Injection** (`Depends`) for DB clients, LLM clients, and configuration objects. Demonstrate how this makes unit testing straightforward.
- [ ] **Return appropriate response types**: `JSONResponse`, `StreamingResponse` for chat streams, and explicit status codes (`status.HTTP_201_CREATED`).
- [ ] **Document endpoints** automatically via Pydantic models; verify generated OpenAPI UI includes request/response schemas.
- [ ] **Run the server** with `uvicorn main:app --reload` and access Swagger UI at `/docs`.

### 📄 JSON Schema / OpenAPI
- [ ] **Inspect generated OpenAPI schema** (`app.openapi()`) and verify that all routes, request bodies, and responses are present.
- [ ] **Export the schema** to a JSON file for downstream tooling (`openapi.json`).
- [ ] **Check required/optional fields**: Ensure `required` list matches your Pydantic model definitions.

### 🤖 LLM Fundamentals
- [ ] **Call an LLM API** (Groq/Anthropic/OpenAI) with proper authentication headers. Include a minimal wrapper that retries on transient errors.
- [ ] **Count tokens** using the provider’s tokenizer; confirm payload stays under the model’s context limit.
- [ ] **Implement Chunking**: split long documents into overlapping chunks (e.g., 1024 tokens with 200‑token overlap) before sending to the model.
- [ ] **Build a Recursive Map‑Reduce summarisation**: map stage processes chunks, reduce stage combines summaries, and recursion continues until a final short summary is produced.
- [ ] **Stream responses** with FastAPI’s `StreamingResponse` and `async for` token generation.
- [ ] **Use few‑shot prompting**, chain‑of‑thought, and enforce JSON‑structured output with `response_format` or explicit parsing.
- [ ] **Cache embeddings/prompts** (e.g., using `functools.lru_cache` or an external Redis cache) to avoid redundant API calls.

---

## 2️⃣ Practical Artifacts to Verify (Expanded)

- [ ] **`llm_service.py`**
  - Contains an **async client wrapper** (`httpx.AsyncClient`) with automatic retries and exponential back‑off.
  - Implements **recursive map‑reduce** summarisation with clear type hints and docstrings.
  - Handles errors (`httpx.HTTPStatusError`, `TimeoutError`) and raises a custom `LLMError`.

- [ ] **`main.py`** runs FastAPI with two essential endpoints:
  - `/chat` – **StreamingResponse** that yields tokens from the LLM service using `async for`.
  - `/health` – Simple `JSONResponse` returning `{ "status": "ok" }` and a **200** status.
  - Includes **CORS** settings for frontend integration.

- [ ] **Unit tests** (`pytest`) all pass:
  ```bash
  pytest -q
  ```
  - Tests cover model validation, async service calls (using `pytest-asyncio` and `respx` for HTTP mocking), and FastAPI route responses.

- [ ] **`test_api.py`** uses `httpx.AsyncClient` to:
  - Verify `/health` returns status 200 and correct JSON.
  - Mock the LLM endpoint and assert that `/chat` streams tokens correctly.

- [ ] **`.env`** file is loaded via `python-dotenv` in `main.py` (or a dedicated config module). It must contain at least:
  - `LLM_API_KEY`
  - `EMBEDDING_MODEL`
  - `DB_URL`
  - Verify missing keys raise a clear `RuntimeError`.

- [ ] **Documentation** (`README.md` or module docstrings) includes:
  - Quick‑start commands (`uvicorn main:app --reload`).
  - How to run tests (`pytest`).
  - Explanation of the architecture (ABC interfaces, service layer, FastAPI router).
  - Example curl commands for `/chat` and `/health`.

---

## 3️⃣ Self‑Quiz Questions (Expanded Guidance)

1. **Why do we need `await` inside an `async def`?** – Explain the event‑loop scheduling model and the difference between coroutine objects and their execution.
2. **How does `Chunk overlap` help maintain context?** – Discuss token continuity and avoiding dropped information at chunk boundaries.
3. **Describe the flow of a Recursive Map‑Reduce summarisation.** – Outline map → reduce → recursion steps with pseudocode.
4. **What happens if a downstream LLM request raises a 429 error?** – Detail exponential back‑off, retry limits, and fallback strategies.
5. **How does FastAPI’s Dependency Injection improve testability?** – Show how to inject a mock service in unit tests.
6. **Give an example of a Pydantic validator that enforces a regex pattern.** – Provide a code snippet using `@validator`.
7. **Explain the difference between a `BaseModel` and a plain `dict` in FastAPI responses.** – Mention automatic validation, OpenAPI schema generation, and serialization.
8. **What is the purpose of `gather(return_exceptions=True)` in concurrent LLM calls?** – Explain error collection without aborting other tasks.
9. **How would you convert a synchronous PDF parser to an async version?** – Suggest `run_in_executor` or using an async‑compatible library.
10. **When would you choose a Fixed‑Size chunker over a Recursive one?** – Compare simplicity vs handling very large documents.

---

## 4️⃣ How to Use This Detailed Checklist

1. **Tick** each ✅ after you have verified the corresponding implementation.
2. **Answer** the self‑quiz questions concisely; keep a short paragraph per answer.
3. If any ✅ remains unchecked, revisit the related code, tests, or docs.
4. Once every ✅ is checked **and** you are satisfied with your quiz answers, you have successfully **passed Phase 1** and can proceed to Phase 2.
