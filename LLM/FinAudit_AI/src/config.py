"""
config.py — Quản lý tập trung toàn bộ cấu hình hệ thống FinAudit AI.
Kế thừa triết lý từ FinRisk AI: Không hardcode key/value trong logic,
mọi cấu hình được load tự động qua pydantic-settings từ .env.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Cấu hình toàn hệ thống FinAudit AI."""

    # ── LLM Configuration ──────────────────────────────────────────────
    # Môi trường dev mặc định dùng Groq (nhanh, miễn phí)
    llm_provider: Literal["groq", "openai", "anthropic"] = "groq"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-haiku-latest"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

    # ── Phase 1: Block Classifier & Parser Settings ────────────────────
    # Ngưỡng tin cậy (confidence) của Rule-based heuristic. Nếu < threshold thì fallback LLM
    rule_confidence_threshold: float = 0.75

    # Tỷ lệ ô chứa số tối thiểu để coi bảng là bảng số liệu tài chính (Numeric density)
    high_numeric_density_threshold: float = 0.40

    # Số dòng tối thiểu để xem xét là bảng báo cáo tài chính hoàn chỉnh
    min_table_rows_for_statement: int = 3

    # Kích thước tối đa của đoạn văn bản cho 1 text block trước khi tách tiếp
    max_text_block_chars: int = 2500

    # ── Vision OCR API Fallback (Xử lý file PDF Scan dạng hình ảnh bằng Free API) ───
    enable_ocr_fallback: bool = True
    ocr_provider: Literal["auto", "gemini", "groq"] = "auto"
    gemini_api_key: str = ""
    gemini_vision_model: str = "gemini-flash-latest"
    groq_vision_model: str = "llama-3.2-11b-vision-preview"
    ocr_resolution: int = 150
    ocr_min_char_threshold: int = 50

    # ── Môi trường & Logging ────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
# lru_cache: lastest recently used, lần đầu nạp hết Setting vào RAM, sao đó ko cần gọi lại từ ổ đĩa nữa
def get_settings() -> Settings:
    """
    Singleton pattern — parse .env một lần duy nhất và cache lại trong bộ nhớ.
    Có thể dùng dependency injection trong API hoặc mock trong test suite.
    """
    return Settings()
