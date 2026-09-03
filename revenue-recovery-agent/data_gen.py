"""
data_gen.py
Synthetic transaction batch generator.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List

from schemas import TransactionEvent, TransactionType, ErrorCode

_ERROR_WEIGHTS = [
    (ErrorCode.INSUFFICIENT_FUNDS, 0.28),
    (ErrorCode.NETWORK_TIMEOUT, 0.15),
    (ErrorCode.MANDATE_EXPIRED, 0.12),
    (ErrorCode.NO_RESPONSE, 0.18),
    (ErrorCode.ISSUER_DOWN, 0.07),
    (ErrorCode.EXPIRED_CARD, 0.08),
    (ErrorCode.DO_NOT_HONOR, 0.06),
    (ErrorCode.ACCOUNT_CLOSED, 0.03),
    (ErrorCode.FRAUD_SUSPECTED, 0.02),
    (ErrorCode.CARD_REPORTED_LOST, 0.01),
]

_SEGMENTS = ["consumer_d2c", "sme_b2b", "enterprise_b2b"]
_LANGUAGES = ["en", "hi", "hinglish"]


def _weighted_error_code() -> ErrorCode:
    codes, weights = zip(*_ERROR_WEIGHTS)
    return random.choices(codes, weights=weights, k=1)[0]


def _amount_for_segment(segment: str) -> float:
    if segment == "consumer_d2c":
        return round(random.uniform(299, 4999), 2)
    if segment == "sme_b2b":
        return round(random.uniform(15000, 180000), 2)
    return round(random.uniform(80000, 450000), 2)


def generate_synthetic_batch(n: int = 35, seed: int | None = 42) -> List[TransactionEvent]:
    if seed is not None:
        random.seed(seed)

    batch = []
    for _ in range(n):
        segment = random.choice(_SEGMENTS)
        tx_type = (
            TransactionType.B2B_INVOICE
            if segment != "consumer_d2c"
            else random.choice([TransactionType.RECURRING_PAYMENT, TransactionType.ABANDONED_CHECKOUT])
        )
        created_at = datetime.utcnow() - timedelta(days=random.randint(0, 30))
        batch.append(
            TransactionEvent(
                transaction_id=f"TXN-{uuid.uuid4().hex[:8].upper()}",
                customer_id=f"CUST-{uuid.uuid4().hex[:6].upper()}",
                type=tx_type,
                amount_inr=_amount_for_segment(segment),
                error_code=_weighted_error_code(),
                retry_count=random.choices([0, 1, 2], weights=[0.55, 0.30, 0.15])[0],
                created_at=created_at,
                due_date=created_at + timedelta(days=random.randint(3, 21)),
                customer_segment=segment,
                language_pref=random.choice(_LANGUAGES),
            )
        )
    return batch