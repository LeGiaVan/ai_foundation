# Semantic Chunking & Chunk‑size Selection – Cheatsheet

---

## 1️⃣  What is **Semantic Chunking**?

- **Goal:** Create chunks that **preserve meaning** instead of just cutting at a fixed number of characters/tokens.
- **How:** Embed each *sentence* (or smaller unit) with a language‑model embedding, compute similarity between consecutive embeddings, and place a **breakpoint** where similarity drops sharply.
- The result: each chunk contains a coherent idea/topic, which yields **higher‑quality vectors** for similarity search or retrieval‑augmented generation (RAG).

> **Why it matters** – When you later query an LLM, the model receives a chunk that already contains a complete thought, so the answer is more accurate and the retrieval step needs fewer hops.

---

## 2️⃣  When to use **Semantic Chunking** vs. Fixed‑size / Recursive

| Situation | Recommended strategy |
|-----------|----------------------|
| ✅ **Legal, medical, contracts** – need **exact context** | **SemanticChunker** (or custom similarity‑based splitter) |
| ✅ **Long narrative books / research papers** – paragraphs are long but semantically coherent | **Recursive + overlap** (fast, good enough) |
| ✅ **FAQ, checklist, logs** – each line is an independent unit | **Fixed‑size** (or even line‑by‑line) |
| ✅ **Prototype / quick demo** | **Fixed‑size** (simple) |

---

## 3️⃣  Core Concepts & Terminology

- **Embedding model** – a transformer that maps a piece of text to a dense vector (e.g., `all-MiniLM-L6-v2`).
- **Similarity metric** – usually *cosine similarity*; values near 1.0 mean “very similar”.
- **Breakpoint threshold** – a similarity value (e.g., `< 0.75`) that signals a topic shift.
- **Chunk size limits** – after detecting a breakpoint you may still enforce a *max token* limit to stay within the LLM’s context window.

---

## 4️⃣  Setup (LangChain + Sentence‑Transformers)

```bash
# Install required packages (run inside your venv)
pip install langchain-community sentence-transformers
```

```python
# ---------------------------------------------------------------
# 4.1  Import & initialise the embedding model
# ---------------------------------------------------------------
from sentence_transformers import SentenceTransformer
from langchain_community.text_splitters import SemanticChunker

# Choose a lightweight, general‑purpose model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------
# 4.2  Build the SemanticChunker
# ---------------------------------------------------------------
semantic_splitter = SemanticChunker(
    embedding_function=embed_model.encode,   # callable that returns np.ndarray
    chunk_size=500,          # max tokens per chunk (adjust later)
    chunk_overlap=50,        # keep a little overlap (10‑20% of chunk_size)
    similarity_threshold=0.75,  # break when similarity < threshold
)
```

---

## 5️⃣  Full Example – From PDF → Semantic Chunks

```python
import fitz  # pymupdf – already installed in phase_2
from pathlib import Path

# ----- 5.1  Extract raw text from a PDF (reuse your existing helper) -----
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

# ----- 5.2  Load, split and inspect -----
pdf_path = Path("../phase_2/pdf.pdf")
raw_text = extract_text_from_pdf(str(pdf_path))

chunks = semantic_splitter.split_text(raw_text)

print(f"Created {len(chunks)} semantic chunks")
for i, ch in enumerate(chunks[:5]):  # show first 5 only
    print(f"--- Chunk {i+1} ({len(ch)} characters) ---")
    print(ch[:200].replace("\n", " ") + "...")
```

**What you will see** – each printed chunk ends at a natural topic boundary instead of mid‑sentence.

---

## 6️⃣  Choosing **Chunk Size** & **Overlap**

| Parameter | Recommended range | How to decide |
|-----------|-------------------|----------------|
| `chunk_size` (tokens) | **256‑512** for *semantic search*; **512‑1024** for *RAG‑QA* | Estimate the typical answer length for your use‑case. 256 tokens ≈ 150‑200 words. |
| `chunk_overlap` (tokens) | **10‑20 %** of `chunk_size` (e.g., 50 tokens when `chunk_size=500`) | Enough to keep the tail of a concept that might span two chunks. |
| `similarity_threshold` | **0.70 – 0.85** | Lower → more aggressive splitting (more chunks, finer granularity). Higher → fewer, larger chunks. Adjust on a sample document and inspect the output. |

### Quick heuristic for **token‑based sizing**
```python
# Rough conversion: 1 token ≈ 4 characters (English) or 1.3 words (approx.)
# If you target 512 tokens → ~2000 characters (~300‑350 words).
```

---

## 7️⃣  Trade‑offs & Performance

| Aspect | Fixed‑size / Recursive | Semantic Chunking |
|--------|----------------------|------------------|
| **Speed** | Very fast – pure string ops | Slower – requires embedding every sentence (CPU/GPU cost) |
| **Memory** | Low (only original text) | Higher – store embeddings (float32) for each sentence |
| **Chunk quality** | May cut mid‑idea → lower retrieval relevance | Preserves topic coherence → higher relevance, especially for long documents |
| **Control** | Exact token/character limit | Token limit *plus* similarity‑based breakpoints (you can still enforce a max token) |

---

## 8️⃣  Evaluation Checklist

1. **Visual inspection** – print first few chunks; ensure they end at logical boundaries.
2. **Token count** – verify `len(splitter.length_function(chunk)) <= chunk_size` for every chunk.
3. **Similarity sanity** – compute average cosine similarity between consecutive chunks; a sharp drop indicates proper breakpoints.
4. **Retrieval test** – pick a query, retrieve top‑k chunks, and manually confirm the answer resides wholly inside a returned chunk.

---

## 9️⃣  References & Further Reading

- **Pinecone – Chunking Strategies for LLM Applications** – https://www.pinecone.io/learn/chunking-strategies/
- **LangChain Docs – SemanticChunker** – https://python.langchain.com/docs/how_to/semantic-chunker/
- **Sentence‑Transformers** – https://www.sbert.net/
- **“How to Choose Chunk Size for Retrieval‑Augmented Generation”** – blog post by Cohere (search online for the latest guide).

---

## 🔧  Quick Starter Script (copy‑paste)

```python
# semantic_chunk_demo.py
from pathlib import Path
import fitz
from sentence_transformers import SentenceTransformer
from langchain_community.text_splitters import SemanticChunker

# ---------- 1. Load PDF ----------
pdf_path = Path("../phase_2/pdf.pdf")
with fitz.open(str(pdf_path)) as doc:
    raw = "\n".join(page.get_text("text") for page in doc)

# ---------- 2. Initialise model & splitter ----------
model = SentenceTransformer("all-MiniLM-L6-v2")
splitter = SemanticChunker(
    embedding_function=model.encode,
    chunk_size=500,
    chunk_overlap=50,
    similarity_threshold=0.75,
)

# ---------- 3. Split & display ----------
chunks = splitter.split_text(raw)
print(f"Created {len(chunks)} semantic chunks")
for i, c in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ({len(c)} chars) ---")
    print(c[:300].replace("\n", " ") + "...")
```

Run with:
```bash
python semantic_chunk_demo.py
```

You should see **4‑6** chunks for a typical 2‑3 page PDF, each ending at a natural topic transition.

---

### 🎯  Bottom‑line

- Use **Semantic Chunking** when *meaning* matters more than *speed* (legal docs, medical records, long‑form articles).
- Pick **chunk_size** based on the target model’s context window and the expected answer length.
- Always keep a modest **overlap** (10‑20%) to guarantee that the tail of a concept isn’t lost.
- Validate with visual checks and a simple retrieval test before feeding chunks into your downstream RAG pipeline.

---
