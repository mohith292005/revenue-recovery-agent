"""
agent.py
LLM orchestration via OpenRouter. Single model, no fallback. The LLM only
PROPOSES an intervention via forced tool call; it never executes anything.
Includes a credits-exhausted circuit breaker.
"""

import json
import logging
import time
from openai import OpenAI

from config import get_llm_client, PRIMARY_MODEL
from schemas import TransactionEvent, DiagnosisResult, InterventionDecision, ActionType, Channel

logger = logging.getLogger("agent")

_credits_exhausted = False


def reset_circuit_breaker():
    global _credits_exhausted
    _credits_exhausted = False


SYSTEM_PROMPT = """You are a financial revenue-recovery decision assistant operating under strict compliance controls.

RULES YOU MUST FOLLOW:
- You do NOT execute any action. You only PROPOSE one intervention by calling `propose_intervention`.
- You may only choose an action from the diagnosis's `recommended_action_pool`, unless you believe escalation is safer.
- Hard-stop error codes (EXPIRED_CARD, DO_NOT_HONOR, ACCOUNT_CLOSED, FRAUD_SUSPECTED, CARD_REPORTED_LOST) must ALWAYS be escalated.
- Transactions with retry_count >= 2 must ALWAYS be escalated.
- Transactions with amount_inr > 100000 must ALWAYS be escalated, regardless of root cause.
- Retry delay_hours must be between 4 and 72. Prefer salary-day timing (near month start/end) for INSUFFICIENT_FUNDS cases.
- Only use channels: whatsapp, sms, email, voice_hinglish.
- Be conservative: when uncertain, escalate rather than retry.

You must always respond by calling the `propose_intervention` tool exactly once. Do not respond in plain text.
"""

PROPOSE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "propose_intervention",
        "description": "Propose a single recovery intervention for a transaction. This is a PROPOSAL only.",
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_id": {"type": "string"},
                "action": {"type": "string", "enum": [a.value for a in ActionType]},
                "channel": {"type": ["string", "null"], "enum": [c.value for c in Channel] + [None]},
                "delay_hours": {"type": ["integer", "null"], "minimum": 0, "maximum": 72},
                "language": {"type": ["string", "null"], "enum": ["en", "hi", "hinglish", None]},
                "reasoning": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["transaction_id", "action", "reasoning", "confidence"],
        },
    },
}


def _build_user_prompt(tx: TransactionEvent, diagnosis: DiagnosisResult) -> str:
    return f"""Transaction details:
- transaction_id: {tx.transaction_id}
- type: {tx.type.value}
- amount_inr: {tx.amount_inr}
- error_code: {tx.error_code.value}
- retry_count: {tx.retry_count}
- customer_segment: {tx.customer_segment}
- language_pref: {tx.language_pref}
- due_date: {tx.due_date}

Deterministic diagnosis (already computed, do not re-derive):
- root_cause: {diagnosis.root_cause}
- is_hard_stop: {diagnosis.is_hard_stop}
- recoverable: {diagnosis.recoverable}
- recommended_action_pool: {[a.value for a in diagnosis.recommended_action_pool]}

Propose the single best intervention by calling propose_intervention."""


def _call_model(client: OpenAI, tx: TransactionEvent, diagnosis: DiagnosisResult):
    return client.chat.completions.create(
        model=PRIMARY_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(tx, diagnosis)},
        ],
        tools=[PROPOSE_TOOL_SCHEMA],
        tool_choice={"type": "function", "function": {"name": "propose_intervention"}},
        temperature=0.2,
        max_tokens=800,
    )


import time

def get_llm_decision(tx: TransactionEvent, diagnosis: DiagnosisResult) -> InterventionDecision:
    client = get_llm_client()

    for attempt in range(2):
        try:
            response = _call_model(client, tx, diagnosis)
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                raise ValueError("Model did not return a tool call.")

            args = json.loads(tool_calls[0].function.arguments)
            return InterventionDecision.model_validate(args)

        except Exception as e:
            err_str = str(e).lower()
            is_transient = (
                "503" in err_str or "unavailable" in err_str or "high demand" in err_str
                 or "overloaded" in err_str or "timeout" in err_str
                )

            if is_transient and attempt == 0:
                logger.warning(f"Model temporarily unavailable for {tx.transaction_id}, retrying in 5s...")
                time.sleep(5)
                continue

            logger.warning(f"Model call failed for {tx.transaction_id}: {e}")
            return InterventionDecision(
                transaction_id=tx.transaction_id,
                action=ActionType.ESCALATE,
                reasoning=f"LLM decisioning failed ({type(e).__name__}); defaulting to human escalation.",
                confidence=0.0,
            )