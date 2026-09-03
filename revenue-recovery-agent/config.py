"""
config.py
Gemini client via OpenAI-compatible endpoint + global safety thresholds.
"""

import os
from dataclasses import dataclass, field
from typing import FrozenSet

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gemini-2.0-flash")

if not GOOGLE_API_KEY:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    except Exception:
        pass

_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    global _client, GOOGLE_API_KEY
    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
                    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
            except Exception:
                pass
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file or Streamlit Secrets.")

    if _client is None:
        _client = OpenAI(
            base_url=GEMINI_BASE_URL,
            api_key=GOOGLE_API_KEY,
            timeout=60.0,
            max_retries=1,
        )
    return _client


@dataclass(frozen=True)
class SafetyThresholds:
    MAX_RETRY_ATTEMPTS: int = 2
    ESCALATION_VALUE_INR: float = 100_000.0
    HARD_STOP_ERROR_CODES: FrozenSet[str] = field(
        default_factory=lambda: frozenset(
            {"EXPIRED_CARD", "DO_NOT_HONOR", "ACCOUNT_CLOSED", "FRAUD_SUSPECTED", "CARD_REPORTED_LOST"}
        )
    )
    MIN_RETRY_DELAY_HOURS: int = 4
    MAX_RETRY_DELAY_HOURS: int = 72
    ALLOWED_CHANNELS: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"whatsapp", "sms", "email", "voice_hinglish"})
    )
    NUDGE_COOLDOWN_HOURS: int = 12


THRESHOLDS = SafetyThresholds()