"""
schemas.py
Strict typed contracts for every object flowing through the agent.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List

from pydantic import BaseModel, Field, field_validator


class TransactionType(str, Enum):
    RECURRING_PAYMENT = "recurring_payment"
    ABANDONED_CHECKOUT = "abandoned_checkout"
    B2B_INVOICE = "b2b_invoice"


class ErrorCode(str, Enum):
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    MANDATE_EXPIRED = "MANDATE_EXPIRED"
    EXPIRED_CARD = "EXPIRED_CARD"
    DO_NOT_HONOR = "DO_NOT_HONOR"
    ACCOUNT_CLOSED = "ACCOUNT_CLOSED"
    FRAUD_SUSPECTED = "FRAUD_SUSPECTED"
    CARD_REPORTED_LOST = "CARD_REPORTED_LOST"
    ISSUER_DOWN = "ISSUER_DOWN"
    NO_RESPONSE = "NO_RESPONSE"


class ActionType(str, Enum):
    RETRY = "retry"
    NUDGE = "nudge"
    RETRY_AND_NUDGE = "retry_and_nudge"
    ESCALATE = "escalate"
    NO_ACTION = "no_action"


class Channel(str, Enum):
    WHATSAPP = "whatsapp"
    SMS = "sms"
    EMAIL = "email"
    VOICE_HINGLISH = "voice_hinglish"


class TransactionEvent(BaseModel):
    transaction_id: str
    customer_id: str
    type: TransactionType
    amount_inr: float = Field(gt=0)
    error_code: ErrorCode
    retry_count: int = Field(ge=0, default=0)
    created_at: datetime
    due_date: Optional[datetime] = None
    customer_segment: str
    language_pref: str = "en"

    @field_validator("amount_inr")
    @classmethod
    def round_amount(cls, v: float) -> float:
        return round(v, 2)


class DiagnosisResult(BaseModel):
    transaction_id: str
    root_cause: str
    is_hard_stop: bool
    recoverable: bool
    recommended_action_pool: List[ActionType]
    notes: str


class InterventionDecision(BaseModel):
    transaction_id: str
    action: ActionType
    channel: Optional[Channel] = None
    delay_hours: Optional[int] = Field(default=None, ge=0)
    language: Optional[str] = None
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


class AuditLog(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    transaction_id: str
    actor: str
    event_type: str
    details: Dict[str, str | float | int | bool | None] = Field(default_factory=dict)


class BatchRecoveryReport(BaseModel):
    total_transactions: int
    total_money_at_risk_inr: float
    total_recovered_inr: float
    recovery_rate_pct: float
    escalations: int
    root_cause_breakdown: Dict[str, int]
    action_breakdown: Dict[str, int]
    hard_stops: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)