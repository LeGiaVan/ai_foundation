"""
config.py — Tập trung toàn bộ cấu hình qua pydantic-settings.
Nguyên tắc: KHÔNG hardcode bất kỳ key/value nào trong code.
Mọi giá trị đều đọc từ biến môi trường (.env) hoặc GitHub Secrets.

Scale-up hint:
  - Thêm field vào Settings khi cần config mới.
  - Khi deploy nhiều môi trường (dev/staging/prod), chỉ cần đổi file .env.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM ─────────────────────────────────────────────────────────────
    # Phase học: Dùng Groq (miễn phí). Phase production: Đổi sang anthropic/openai.
    llm_provider: str = "groq"                # "groq" | "anthropic" | "openai"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"   # Model mạnh, miễn phí trên Groq
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"    # Giữ llm_model để backward compatible
    llm_temperature: float = 0.0              # 0.0 = deterministic, quan trọng với tài chính
    llm_max_tokens: int = 4096

    # ── LangGraph ───────────────────────────────────────────────────────
    checkpointer_db_path: str = "./data/checkpoints.sqlite"  # Phase 1: SQLite
    max_agent_iterations: int = 10            # Chặn vòng lặp vô hạn

    # ── Thresholds nghiệp vụ (5C Framework) ────────────────────────────
    # Khi các chỉ số vượt ngưỡng này, hệ thống kích hoạt Human-in-the-Loop
    zscore_distress_threshold: float = 1.81   # Altman Z-Score < 1.81 → Nguy cơ phá sản
    dscr_minimum: float = 1.0                 # DSCR < 1.0 → Không đủ tiền trả nợ
    debt_equity_max: float = 3.0              # D/E > 3.0 → Đòn bẩy quá cao
    quick_ratio_minimum: float = 0.8          # Quick Ratio < 0.8 → Thanh khoản kém

    # ── Risk Score → Tự động xếp loại phán quyết ──────────────────────
    risk_score_reject_threshold: float = 70.0  # Score >= 70 → Từ chối tự động
    risk_score_review_threshold: float = 40.0  # 40 <= Score < 70 → Cần Human review

    # ── Langfuse Observability (Phase 3) ────────────────────────────────
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    enable_tracing: bool = True

    # ── Qdrant Vector DB (Phase 2) ───────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""              # Để trống khi dùng local; điền khi dùng Qdrant Cloud
    qdrant_collection: str = "financial_docs"

    # ── Embedding & Reranker (Phase 2) ──────────────────────────────────
    dense_model: str = "BAAI/bge-m3"                  # fastembed auto-download ~600MB
    reranker_model: str = "BAAI/bge-reranker-v2-m3"   # sentence-transformers ~1.1GB
    reranker_threshold: float = 0.3                    # Score tối thiểu sau rerank
    retrieval_top_k: int = 20                          # Số candidate trước khi rerank
    retrieval_top_n: int = 4                           # Số doc cuối gửi LLM

    # ── Chunking (Phase 2) ───────────────────────────────────────────────
    chunk_child_tokens: int = 150     # Child chunk nhỏ → dùng để embed & search
    chunk_parent_tokens: int = 1000   # Parent chunk lớn → dùng để gửi LLM (context đầy đủ)

    # ── Môi trường ───────────────────────────────────────────────────────
    environment: str = "development"          # "development" | "staging" | "production"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",                       # Bỏ qua biến môi trường không khai báo
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton pattern — chỉ parse .env một lần, cache lại.
    Dùng dependency injection trong FastAPI: Depends(get_settings)
    """
    return Settings()
