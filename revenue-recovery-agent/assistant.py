"""
assistant.py
Read-only AI assistant for querying batch results, transaction details, and
the audit trail via natural language. Uses OpenRouter with forced tool
calling, same as the core agent — but every tool here is READ-ONLY. This
assistant can never trigger a retry, nudge, or escalation; it can only
explain and query what has already happened.
"""
import time
import json
import logging
from typing import List, Dict, Any

from openai import OpenAI

from config import get_llm_client, PRIMARY_MODEL
from schemas import BatchRecoveryReport

logger = logging.getLogger("assistant")

SYSTEM_PROMPT = """You are an analyst assistant for an AI Revenue Recovery system.

You have READ-ONLY access to the current batch's transaction data, the summary
report, and the audit trail. You cannot trigger any action (no retries, nudges,
or escalations) — you can only look up and explain what has already happened.

Guidelines:
- Always use the provided tools to look up real data before answering. Never
  guess or fabricate transaction IDs, amounts, or outcomes.
- When explaining a decision, reference the actual root cause, the LLM's
  original proposal, and whether the rule engine overrode it (and why).
- Be concise and precise — this is a financial ops tool, not a casual chatbot.
- If asked something outside your data (e.g. general finance advice), politely
  clarify you can only answer questions about this batch's data.
- Cite actual transaction IDs, amounts, and reasons when relevant.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_batch_report",
            "description": "Get the current batch's summary metrics: total money at risk, total recovered, recovery rate, escalations, root cause breakdown, action breakdown.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_transactions",
            "description": "Search/filter processed transactions by action, whether they were overridden by the rule engine, minimum amount, or error code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["retry", "nudge", "retry_and_nudge", "escalate", "no_action"], "description": "Filter by final action taken."},
                    "overridden_only": {"type": "boolean", "description": "If true, only return transactions where the rule engine overrode the LLM's proposal."},
                    "min_amount_inr": {"type": "number", "description": "Only return transactions with amount >= this value."},
                    "error_code": {"type": "string", "description": "Filter by exact error code, e.g. INSUFFICIENT_FUNDS."},
                    "limit": {"type": "integer", "description": "Max results to return, default 10."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transaction_detail",
            "description": "Get full details for one specific transaction by its transaction_id, including root cause, LLM reasoning, and whether it was overridden.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string"},
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_audit_trail",
            "description": "Get the raw step-by-step audit log entries (diagnosis, LLM proposal, guardrail decision, execution) for a specific transaction_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string"},
                },
                "required": ["transaction_id"],
            },
        },
    },
]


def _execute_tool(name: str, args: dict, results: List[dict], report: BatchRecoveryReport | None) -> Any:
    """Dispatches to the actual read-only data lookups. `results` is the
    live list of serialized transaction results from app.py's state."""

    if name == "get_batch_report":
        if report is None:
            return {"error": "No completed batch yet."}
        return report.model_dump()

    if name == "search_transactions":
        filtered = results
        if args.get("action"):
            filtered = [r for r in filtered if r.get("action") == args["action"]]
        if args.get("overridden_only"):
            filtered = [r for r in filtered if r.get("overridden")]
        if args.get("min_amount_inr") is not None:
            filtered = [r for r in filtered if r.get("amount_inr", 0) >= args["min_amount_inr"]]
        if args.get("error_code"):
            filtered = [r for r in filtered if r.get("error_code") == args["error_code"]]
        limit = args.get("limit", 10)
        return filtered[:limit]

    if name == "get_transaction_detail":
        tx_id = args.get("transaction_id")
        match = next((r for r in results if r.get("transaction_id") == tx_id), None)
        return match if match else {"error": f"Transaction {tx_id} not found in current batch results."}

    if name == "get_audit_trail":
        tx_id = args.get("transaction_id")
        from tools import AUDIT_LOG_PATH
        entries = []
        if AUDIT_LOG_PATH.exists():
            with AUDIT_LOG_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("transaction_id") == tx_id:
                        entries.append(entry)
        return entries if entries else {"error": f"No audit entries found for {tx_id}."}

    return {"error": f"Unknown tool: {name}"}


import time

def chat(
    message: str,
    history: List[Dict[str, str]],
    results: List[dict],
    report: BatchRecoveryReport | None,
) -> str:
    client: OpenAI = get_llm_client()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": message}
    ]

    for attempt in range(3):
        try:
            for _ in range(4):
                response = client.chat.completions.create(
                    model=PRIMARY_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    temperature=0.3,
                    max_tokens=500,
                )
                msg = response.choices[0].message

                if not msg.tool_calls:
                    return msg.content or "I couldn't generate a response."

                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        **(
                            {"extra_content": {"google": {"thought_signature": tc.model_extra["extra_content"]["google"]["thought_signature"]}}}
                            if getattr(tc, "model_extra", None) and tc.model_extra.get("extra_content", {}).get("google", {}).get("thought_signature")
                            else {}
                        ),
                    }
                    for tc in msg.tool_calls
                ],
            })

                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    tool_result = _execute_tool(tc.function.name, args, results, report)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, default=str),
                    })

            return "I looked into this but wasn't able to finish — try rephrasing your question."

        except Exception as e:
            err_str = str(e).lower()
            is_transient = "503" in err_str or "unavailable" in err_str or "high demand" in err_str or "overloaded" in err_str or "timeout" in err_str

            if is_transient and attempt < 2:
                import time
                wait = 3 * (attempt + 1)
                logger.warning(f"Model temporarily unavailable, retrying in {wait}s (attempt {attempt+1}/3)...")
                time.sleep(wait)
                continue

            import traceback
            logger.error(f"Assistant chat failed: {e}")
            logger.error(traceback.format_exc())
            return f"Sorry, I hit an error looking that up ({type(e).__name__}). Try again in a moment."