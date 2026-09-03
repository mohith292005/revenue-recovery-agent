"""
tools.py
Deterministic tools and the rule engine. Zero LLM calls in this module.
Audit log lives under data/ so it never triggers uvicorn --reload watchers
if you point --reload-exclude at that folder.
"""
import store
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List

from config import THRESHOLDS
from schemas import (
    TransactionEvent,
    DiagnosisResult,
    InterventionDecision,
    AuditLog,
    ActionType,
    ErrorCode,
    Channel,
)

AUDIT_LOG_DIR = Path("data")
AUDIT_LOG_DIR.mkdir(exist_ok=True)
AUDIT_LOG_PATH = AUDIT_LOG_DIR / "audit_log.jsonl"


# ---------------------------------------------------------------------------
# 1. Ingestion
# ---------------------------------------------------------------------------

def fetch_failed_transactions(dataset: List[TransactionEvent]) -> List[TransactionEvent]:
    return dataset


# ---------------------------------------------------------------------------
# 2. Diagnosis
# ---------------------------------------------------------------------------
import threading

_audit_log_lock = threading.Lock()

def log_audit_trail(entry: AuditLog) -> None:
    with _audit_log_lock:
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
_ROOT_CAUSE_MAP = {
    ErrorCode.INSUFFICIENT_FUNDS: (
        "Customer cash-flow timing issue", True, [ActionType.RETRY, ActionType.RETRY_AND_NUDGE],
    ),
    ErrorCode.NETWORK_TIMEOUT: (
        "Transient gateway/network failure", True, [ActionType.RETRY],
    ),
    ErrorCode.ISSUER_DOWN: (
        "Issuing bank system unavailability", True, [ActionType.RETRY],
    ),
    ErrorCode.MANDATE_EXPIRED: (
        "Recurring mandate/authorization lapsed", True, [ActionType.NUDGE],
    ),
    ErrorCode.NO_RESPONSE: (
        "Customer inaction (abandoned checkout / unpaid invoice)", True, [ActionType.NUDGE],
    ),
    ErrorCode.EXPIRED_CARD: (
        "Card expired — requires customer action", False, [ActionType.ESCALATE],
    ),
    ErrorCode.DO_NOT_HONOR: (
        "Bank declined categorically", False, [ActionType.ESCALATE],
    ),
    ErrorCode.ACCOUNT_CLOSED: (
        "Underlying account closed", False, [ActionType.ESCALATE],
    ),
    ErrorCode.FRAUD_SUSPECTED: (
        "Fraud flag raised by issuer", False, [ActionType.ESCALATE],
    ),
    ErrorCode.CARD_REPORTED_LOST: (
        "Card reported lost/stolen", False, [ActionType.ESCALATE],
    ),
}


def diagnose_failure_cause(tx: TransactionEvent) -> DiagnosisResult:
    root_cause, recoverable, pool = _ROOT_CAUSE_MAP[tx.error_code]
    is_hard_stop = tx.error_code in THRESHOLDS.HARD_STOP_ERROR_CODES

    if is_hard_stop:
        pool = [ActionType.ESCALATE]
        recoverable = False

    return DiagnosisResult(
        transaction_id=tx.transaction_id,
        root_cause=root_cause,
        is_hard_stop=is_hard_stop,
        recoverable=recoverable,
        recommended_action_pool=pool,
        notes=f"error_code={tx.error_code.value}, retry_count={tx.retry_count}, "
              f"amount=₹{tx.amount_inr:,.2f}",
    )


# ---------------------------------------------------------------------------
# 3. RULE ENGINE — final authority over every action
# ---------------------------------------------------------------------------

def validate_decision(
    tx: TransactionEvent,
    diagnosis: DiagnosisResult,
    proposed: InterventionDecision,
) -> Tuple[InterventionDecision, bool, str]:

    if diagnosis.is_hard_stop:
        if proposed.action != ActionType.ESCALATE:
            return (
                InterventionDecision(
                    transaction_id=tx.transaction_id,
                    action=ActionType.ESCALATE,
                    reasoning=f"Hard-stop error code {tx.error_code.value} overrides LLM proposal.",
                    confidence=1.0,
                ),
                True,
                f"Hard stop code {tx.error_code.value} forces escalation regardless of LLM output.",
            )
        return proposed, False, ""

    effective_retry_count = max(tx.retry_count, store.get_persisted_retry_count(tx.transaction_id))

    if effective_retry_count >= THRESHOLDS.MAX_RETRY_ATTEMPTS and proposed.action in (
        ActionType.RETRY, ActionType.RETRY_AND_NUDGE,
    ):
        return (
            InterventionDecision(
                transaction_id=tx.transaction_id,
                action=ActionType.ESCALATE,
                reasoning=f"Effective retry count ({effective_retry_count}) reached max attempts "
                          f"({THRESHOLDS.MAX_RETRY_ATTEMPTS}).",
                confidence=1.0,
            ),
            True,
            "Max retry attempts exceeded (persisted history + current batch).",
        )

    if tx.amount_inr > THRESHOLDS.ESCALATION_VALUE_INR and proposed.action != ActionType.ESCALATE:
        return (
            InterventionDecision(
                transaction_id=tx.transaction_id,
                action=ActionType.ESCALATE,
                reasoning=f"Amount ₹{tx.amount_inr:,.2f} exceeds escalation threshold "
                          f"₹{THRESHOLDS.ESCALATION_VALUE_INR:,.2f}.",
                confidence=1.0,
            ),
            True,
            "High-value transaction threshold exceeded.",
        )

    if proposed.action in (ActionType.RETRY, ActionType.RETRY_AND_NUDGE):
        delay = proposed.delay_hours if proposed.delay_hours is not None else THRESHOLDS.MIN_RETRY_DELAY_HOURS
        clamped = max(THRESHOLDS.MIN_RETRY_DELAY_HOURS, min(delay, THRESHOLDS.MAX_RETRY_DELAY_HOURS))
        if clamped != delay:
            proposed = proposed.model_copy(update={"delay_hours": clamped})
            return proposed, True, f"Retry delay clamped to allowed bounds ({clamped}h)."

    if proposed.channel is not None and proposed.channel.value not in THRESHOLDS.ALLOWED_CHANNELS:
        return (
            InterventionDecision(
                transaction_id=tx.transaction_id,
                action=ActionType.ESCALATE,
                reasoning=f"Proposed channel '{proposed.channel}' is not an allowed channel.",
                confidence=1.0,
            ),
            True,
            "Disallowed communication channel proposed by LLM.",
        )

    if proposed.action not in diagnosis.recommended_action_pool and proposed.action != ActionType.ESCALATE:
        return (
            InterventionDecision(
                transaction_id=tx.transaction_id,
                action=ActionType.ESCALATE,
                reasoning=f"Proposed action '{proposed.action.value}' is outside the "
                          f"permitted action pool for this diagnosis; escalating for safety.",
                confidence=1.0,
            ),
            True,
            "Proposed action outside allowed pool for diagnosed root cause.",
        )
    if proposed.action in (ActionType.NUDGE, ActionType.RETRY_AND_NUDGE):
        if store.is_within_nudge_cooldown(tx.transaction_id, THRESHOLDS.NUDGE_COOLDOWN_HOURS):
            return (
                InterventionDecision(
                    transaction_id=tx.transaction_id,
                    action=ActionType.NO_ACTION,
                    reasoning=f"A nudge was already sent within the last "
                              f"{THRESHOLDS.NUDGE_COOLDOWN_HOURS}h cooldown window; skipping to avoid spam.",
                    confidence=1.0,
                ),
                True,
                "Nudge cooldown window still active.",
            )
    return proposed, False, ""


# ---------------------------------------------------------------------------
# 4. Execution tools
# ---------------------------------------------------------------------------

def execute_smart_retry(tx: TransactionEvent, delay_hours: int) -> dict:
    base_success_odds = {
        ErrorCode.INSUFFICIENT_FUNDS: 0.55,
        ErrorCode.NETWORK_TIMEOUT: 0.85,
        ErrorCode.ISSUER_DOWN: 0.75,
    }
    odds = base_success_odds.get(tx.error_code, 0.4)
    succeeded = random.random() < odds

    return {
        "transaction_id": tx.transaction_id,
        "action": "smart_retry",
        "scheduled_delay_hours": delay_hours,
        "scheduled_for": (datetime.utcnow() + timedelta(hours=delay_hours)).isoformat(),
        "succeeded": succeeded,
        "recovered_amount_inr": tx.amount_inr if succeeded else 0.0,
    }


def dispatch_recovery_nudge(tx: TransactionEvent, channel: Channel, language: str) -> dict:
    templates = {
        "en": "Hi, your payment of ₹{amount:,.2f} could not be processed. "
              "Please retry or update your payment method to avoid service interruption.",
        "hi": "नमस्ते, आपका ₹{amount:,.2f} का भुगतान प्रोसेस नहीं हो सका। "
              "कृपया दोबारा प्रयास करें।",
        "hinglish": "Hi! Aapka ₹{amount:,.2f} ka payment fail ho gaya. "
                    "Please payment method update karke retry karein.",
    }
    message = templates.get(language, templates["en"]).format(amount=tx.amount_inr)

    nudge_conversion_odds = 0.25
    converted = random.random() < nudge_conversion_odds

    return {
        "transaction_id": tx.transaction_id,
        "customer_id": tx.customer_id,
        "action": "recovery_nudge",
        "channel": channel.value,
        "language": language,
        "message_preview": message,
        "dispatched_at": datetime.utcnow().isoformat(),
        "converted": converted,
        "recovered_amount_inr": tx.amount_inr if converted else 0.0,
    }


def escalate_to_human(tx: TransactionEvent, reason: str) -> dict:
    return {
        "transaction_id": tx.transaction_id,
        "action": "escalate_to_human",
        "reason": reason,
        "escalation_id": str(uuid.uuid4())[:8],
        "escalated_at": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# 5. Audit logging
# ---------------------------------------------------------------------------

def log_audit_trail(entry: AuditLog) -> None:
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")