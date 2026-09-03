"""
threshold_analyzer.py
Post-batch analysis module. Uses the LLM to inspect override patterns,
escalation rates, and recovery outcomes from the most recent batch, then
PROPOSES adjustments to the SafetyThresholds. This is purely advisory:
suggestions are returned as structured data and NEVER auto-applied.
"""

import json
import logging
from typing import List

from config import get_llm_client, PRIMARY_MODEL, THRESHOLDS
from schemas import BatchRecoveryReport

logger = logging.getLogger("threshold_analyzer")

_SYSTEM_PROMPT = """You are a risk-calibration analyst for an AI Revenue Recovery system.

You will be given statistics from the most recent batch run — including override rates,
escalation rates, recovery rates by error code, and threshold breach counts.

Your job: analyse whether the current safety thresholds are OVER-RESTRICTIVE or
UNDER-RESTRICTIVE, and propose targeted adjustments. Be conservative — only suggest
a change when the data clearly supports it.

You must respond by calling the `propose_threshold_changes` tool exactly once.
Do not respond in plain text.
"""

_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "propose_threshold_changes",
        "description": "Propose threshold adjustments based on batch analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "overall_assessment": {
                    "type": "string",
                    "description": "2-3 sentence summary of the batch's risk posture and key findings.",
                },
                "suggestions": {
                    "type": "array",
                    "description": "List of threshold adjustment suggestions (empty if no changes are warranted).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "threshold_name": {
                                "type": "string",
                                "enum": [
                                    "MAX_RETRY_ATTEMPTS",
                                    "ESCALATION_VALUE_INR",
                                    "NUDGE_COOLDOWN_HOURS",
                                    "MIN_RETRY_DELAY_HOURS",
                                    "MAX_RETRY_DELAY_HOURS",
                                ],
                            },
                            "current_value": {"type": "string"},
                            "proposed_value": {"type": "string"},
                            "direction": {"type": "string", "enum": ["increase", "decrease", "no_change"]},
                            "reasoning": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "impact": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                        "required": [
                            "threshold_name",
                            "current_value",
                            "proposed_value",
                            "direction",
                            "reasoning",
                            "confidence",
                            "impact",
                        ],
                    },
                },
            },
            "required": ["overall_assessment", "suggestions"],
        },
    },
}


def _build_analysis_prompt(results: List[dict], report: BatchRecoveryReport) -> str:
    total = report.total_transactions
    if total == 0:
        return "No transactions to analyze."

    override_count = sum(1 for r in results if r.get("overridden"))
    override_rate = override_count / total * 100

    # Override reason breakdown
    override_reasons: dict[str, int] = {}
    for r in results:
        reason = r.get("override_reason", "")
        if r.get("overridden") and reason:
            key = reason[:80]
            override_reasons[key] = override_reasons.get(key, 0) + 1

    # Recovery rate by error code
    error_recovery: dict[str, dict] = {}
    for r in results:
        ec = r.get("error_code", "UNKNOWN")
        if ec not in error_recovery:
            error_recovery[ec] = {"count": 0, "recovered": 0.0, "at_risk": 0.0}
        error_recovery[ec]["count"] += 1
        error_recovery[ec]["at_risk"] += r.get("amount_inr", 0)
        error_recovery[ec]["recovered"] += r.get("recovered_inr", 0)

    ec_lines = []
    for ec, data in sorted(error_recovery.items(), key=lambda x: -x[1]["count"]):
        rate = (data["recovered"] / data["at_risk"] * 100) if data["at_risk"] else 0
        ec_lines.append(
            f"  {ec}: {data['count']} txns, ₹{data['at_risk']:,.0f} at risk, "
            f"₹{data['recovered']:,.0f} recovered ({rate:.1f}%)"
        )

    # High-value escalations (above threshold)
    high_val_escalations = sum(
        1 for r in results
        if r.get("escalated") and r.get("amount_inr", 0) > THRESHOLDS.ESCALATION_VALUE_INR
    )
    # Near-threshold transactions (80–100% of escalation threshold)
    near_threshold = sum(
        1 for r in results
        if THRESHOLDS.ESCALATION_VALUE_INR * 0.8 <= r.get("amount_inr", 0) <= THRESHOLDS.ESCALATION_VALUE_INR
    )

    max_retry_escalations = sum(
        1 for r in results
        if "Max retry" in (r.get("override_reason") or "") or "retry" in (r.get("override_reason") or "").lower()
    )

    return f"""BATCH ANALYSIS REPORT
=====================
Total transactions: {total}
Total at risk: ₹{report.total_money_at_risk_inr:,.2f}
Total recovered: ₹{report.total_recovered_inr:,.2f}
Recovery rate: {report.recovery_rate_pct:.2f}%
Total escalations: {report.escalations} ({report.escalations/total*100:.1f}% of batch)
Hard stops: {report.hard_stops}

OVERRIDE ANALYSIS
-----------------
Total overrides by rule engine: {override_count} ({override_rate:.1f}% of batch)
Override reason breakdown:
{chr(10).join(f'  "{k}": {v} times' for k, v in override_reasons.items()) or "  (none)"}

THRESHOLD BREACH DETAILS
------------------------
Current ESCALATION_VALUE_INR: ₹{THRESHOLDS.ESCALATION_VALUE_INR:,.2f}
  - Transactions escalated due to high value: {high_val_escalations}
  - Transactions in 80–100% of threshold (near-miss): {near_threshold}
Current MAX_RETRY_ATTEMPTS: {THRESHOLDS.MAX_RETRY_ATTEMPTS}
  - Transactions escalated due to retry count: {max_retry_escalations}
Current NUDGE_COOLDOWN_HOURS: {THRESHOLDS.NUDGE_COOLDOWN_HOURS}h
Current MIN_RETRY_DELAY_HOURS: {THRESHOLDS.MIN_RETRY_DELAY_HOURS}h
Current MAX_RETRY_DELAY_HOURS: {THRESHOLDS.MAX_RETRY_DELAY_HOURS}h

RECOVERY BY ERROR CODE
----------------------
{chr(10).join(ec_lines)}

ACTION BREAKDOWN
----------------
{chr(10).join(f'  {k}: {v}' for k, v in report.action_breakdown.items())}

ROOT CAUSE BREAKDOWN
--------------------
{chr(10).join(f'  {k}: {v}' for k, v in report.root_cause_breakdown.items())}

Based on this data, propose targeted threshold adjustments. If thresholds look well-calibrated,
return an empty suggestions list and explain why in overall_assessment.
"""


def analyze_and_suggest(results: List[dict], report: BatchRecoveryReport) -> dict:
    """
    Run post-batch LLM analysis and return threshold suggestions.
    Always returns a dict; never raises — errors are returned as error keys.
    """
    if not results or report.total_transactions == 0:
        return {
            "overall_assessment": "No batch data available to analyze.",
            "suggestions": [],
        }

    client = get_llm_client()
    prompt = _build_analysis_prompt(results, report)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=[_TOOL_SCHEMA],
                tool_choice={"type": "function", "function": {"name": "propose_threshold_changes"}},
                temperature=0.2,
                max_tokens=1200,
            )
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                raise ValueError("Model returned no tool call.")

            args = json.loads(tool_calls[0].function.arguments)
            # Inject current threshold values for display convenience
            current_map = {
                "MAX_RETRY_ATTEMPTS": str(THRESHOLDS.MAX_RETRY_ATTEMPTS),
                "ESCALATION_VALUE_INR": f"₹{THRESHOLDS.ESCALATION_VALUE_INR:,.0f}",
                "NUDGE_COOLDOWN_HOURS": f"{THRESHOLDS.NUDGE_COOLDOWN_HOURS}h",
                "MIN_RETRY_DELAY_HOURS": f"{THRESHOLDS.MIN_RETRY_DELAY_HOURS}h",
                "MAX_RETRY_DELAY_HOURS": f"{THRESHOLDS.MAX_RETRY_DELAY_HOURS}h",
            }
            for s in args.get("suggestions", []):
                if not s.get("current_value"):
                    s["current_value"] = current_map.get(s["threshold_name"], "—")
            return args

        except Exception as e:
            err = str(e).lower()
            is_transient = any(k in err for k in ("503", "unavailable", "overloaded", "timeout", "high demand"))
            if is_transient and attempt < 2:
                import time
                time.sleep(4 * (attempt + 1))
                continue
            logger.error(f"Threshold analysis LLM call failed: {e}")
            return {
                "overall_assessment": f"Analysis failed ({type(e).__name__}). Try again after running a batch.",
                "suggestions": [],
                "error": str(e),
            }

    return {
        "overall_assessment": "Analysis timed out after 3 attempts. Please retry.",
        "suggestions": [],
    }
