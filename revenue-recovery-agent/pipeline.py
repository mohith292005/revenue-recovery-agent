"""
pipeline.py
Shared batch-processing pipeline. No rendering logic here.
"""
import store
from typing import Callable, List, Optional, Any, Dict

from config import THRESHOLDS
from schemas import TransactionEvent, AuditLog, BatchRecoveryReport, ActionType, Channel, InterventionDecision
from tools import (
    fetch_failed_transactions,
    diagnose_failure_cause,
    validate_decision,
    execute_smart_retry,
    dispatch_recovery_nudge,
    escalate_to_human,
    log_audit_trail,
)
from agent import get_llm_decision

OnEvent = Optional[Callable[[Dict[str, Any]], None]]


def process_transaction(tx: TransactionEvent) -> dict:
    # --- Idempotency guard ---
    if store.has_been_processed(tx.transaction_id):
        log_audit_trail(AuditLog(
            transaction_id=tx.transaction_id, actor="system", event_type="skipped_duplicate",
            details={"reason": "Transaction already processed previously; skipping to avoid double-counting."},
        ))
        return {
            "tx": tx,
            "diagnosis": diagnose_failure_cause(tx),  # for display only, not re-executed
            "final_decision": None,
            "recovered": 0.0,
            "escalated": False,
            "overridden": False,
            "override_reason": "SKIPPED — duplicate transaction_id already processed.",
            "skipped_duplicate": True,
        }

    diagnosis = diagnose_failure_cause(tx)
    log_audit_trail(AuditLog(
        transaction_id=tx.transaction_id, actor="rule_engine", event_type="diagnosis",
        details={"root_cause": diagnosis.root_cause, "is_hard_stop": diagnosis.is_hard_stop},
    ))

    # Fast-path: bypass remote LLM call if deterministic hard-stop conditions are met
    if (
        diagnosis.is_hard_stop
        or tx.retry_count >= THRESHOLDS.MAX_RETRY_ATTEMPTS
        or tx.amount_inr > THRESHOLDS.ESCALATION_VALUE_INR
    ):
        proposed = InterventionDecision(
            transaction_id=tx.transaction_id,
            action=ActionType.ESCALATE,
            reasoning="Deterministic guardrail trigger: Hard stop / threshold condition met; bypassing LLM call.",
            confidence=1.0,
        )
    elif len(diagnosis.recommended_action_pool) == 1:
        single_action = diagnosis.recommended_action_pool[0]
        proposed = InterventionDecision(
            transaction_id=tx.transaction_id,
            action=single_action,
            reasoning=f"Deterministic rule engine proposal: Standard intervention for {diagnosis.root_cause}.",
            confidence=0.95,
            delay_hours=4 if single_action == ActionType.RETRY else None,
            channel=Channel.WHATSAPP if single_action == ActionType.NUDGE else None,
            language=tx.language_pref,
        )
    else:
        # Nuanced multi-action cases (e.g. INSUFFICIENT_FUNDS, salary timing, custom trade-offs)
        proposed = get_llm_decision(tx, diagnosis)
    log_audit_trail(AuditLog(
        transaction_id=tx.transaction_id, actor="agent_llm", event_type="llm_proposal",
        details={"action": proposed.action.value, "confidence": proposed.confidence,
                 "reasoning": proposed.reasoning[:200]},
    ))

    final_decision, overridden, override_reason = validate_decision(tx, diagnosis, proposed)
    log_audit_trail(AuditLog(
        transaction_id=tx.transaction_id, actor="rule_engine", event_type="guardrail_decision",
        details={"final_action": final_decision.action.value, "overridden": overridden,
                 "override_reason": override_reason},
    ))

    recovered = 0.0
    escalated = False
    retried = False
    nudged = False

    if final_decision.action == ActionType.RETRY:
        result = execute_smart_retry(tx, final_decision.delay_hours or 24)
        recovered = result["recovered_amount_inr"]
        retried = True

    elif final_decision.action == ActionType.NUDGE:
        channel = final_decision.channel or Channel.WHATSAPP
        language = final_decision.language or tx.language_pref
        result = dispatch_recovery_nudge(tx, channel, language)
        recovered = result["recovered_amount_inr"]
        nudged = True

    elif final_decision.action == ActionType.RETRY_AND_NUDGE:
        retry_result = execute_smart_retry(tx, final_decision.delay_hours or 24)
        recovered = retry_result["recovered_amount_inr"]
        retried = True
        if recovered == 0.0:
            channel = final_decision.channel or Channel.WHATSAPP
            language = final_decision.language or tx.language_pref
            nudge_result = dispatch_recovery_nudge(tx, channel, language)
            recovered = nudge_result["recovered_amount_inr"]
            nudged = True

    elif final_decision.action == ActionType.ESCALATE:
        escalate_to_human(tx, final_decision.reasoning)
        escalated = True

    log_audit_trail(AuditLog(
        transaction_id=tx.transaction_id, actor="system", event_type="execution",
        details={"action": final_decision.action.value, "recovered_amount_inr": recovered,
                 "escalated": escalated},
    ))

    # --- Persist state for future idempotency / retry-count / cooldown checks ---
    store.record_processing(tx.transaction_id, retried=retried, nudged=nudged, recovered_inr=recovered)

    return {
        "tx": tx, "diagnosis": diagnosis, "final_decision": final_decision,
        "recovered": recovered, "escalated": escalated,
        "overridden": overridden, "override_reason": override_reason,
        "skipped_duplicate": False,
    }


def serialize_result(r: dict) -> dict:
    tx: TransactionEvent = r["tx"]
    return {
        "transaction_id": tx.transaction_id,
        "customer_id": tx.customer_id,
        "type": tx.type.value,
        "amount_inr": tx.amount_inr,
        "error_code": tx.error_code.value,
        "customer_segment": tx.customer_segment,
        "retry_count": tx.retry_count,
        "root_cause": r["diagnosis"].root_cause,
        "is_hard_stop": r["diagnosis"].is_hard_stop,
        "action": r["final_decision"].action.value if r["final_decision"] else "skipped_duplicate",
        "reasoning": r["final_decision"].reasoning if r["final_decision"] else "Duplicate transaction skipped.",
        "overridden": r["overridden"],
        "override_reason": r["override_reason"],
        "confidence": r["final_decision"].confidence if r.get("final_decision") else None,
        "recovered_inr": r["recovered"],
        "escalated": r["escalated"],
        "skipped_duplicate": r.get("skipped_duplicate", False),
    }

def build_report(results: List[dict]) -> BatchRecoveryReport:
    counted = [r for r in results if not r.get("skipped_duplicate", False)]

    total_at_risk = sum(r["tx"].amount_inr for r in counted)
    total_recovered = sum(r["recovered"] for r in counted)
    escalations = sum(1 for r in counted if r["escalated"])
    hard_stops = sum(1 for r in counted if r["diagnosis"].is_hard_stop)

    root_cause_breakdown: Dict[str, int] = {}
    action_breakdown: Dict[str, int] = {}
    for r in counted:
        rc = r["diagnosis"].root_cause
        ac = r["final_decision"].action.value if r["final_decision"] else "skipped_duplicate"
        root_cause_breakdown[rc] = root_cause_breakdown.get(rc, 0) + 1
        action_breakdown[ac] = action_breakdown.get(ac, 0) + 1

    return BatchRecoveryReport(
        total_transactions=len(counted),
        total_money_at_risk_inr=round(total_at_risk, 2),
        total_recovered_inr=round(total_recovered, 2),
        recovery_rate_pct=round((total_recovered / total_at_risk * 100) if total_at_risk else 0, 2),
        escalations=escalations,
        root_cause_breakdown=root_cause_breakdown,
        action_breakdown=action_breakdown,
        hard_stops=hard_stops,
    )

def run_batch_sync(transactions: List[TransactionEvent], on_event: OnEvent = None):
    transactions = fetch_failed_transactions(transactions)
    results = []
    for tx in transactions:
        result = process_transaction(tx)
        results.append(result)
        if on_event:
            on_event(serialize_result(result))
    report = build_report(results)
    return results, report