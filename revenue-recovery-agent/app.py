"""
app.py
FastAPI live dashboard server. Runs the recovery batch in a background
thread and streams per-transaction results to connected browsers over
a WebSocket, so the dashboard updates in real time as the agent works.
"""
import logging
logger = logging.getLogger("app")
from agent import reset_circuit_breaker
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import assistant
import threshold_analyzer
from typing import List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config import THRESHOLDS
from data_gen import generate_synthetic_batch
from pipeline import process_transaction, serialize_result, build_report, OnEvent
from schemas import BatchRecoveryReport
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# In-memory batch state (fine for a single-instance demo/dev server)
# ---------------------------------------------------------------------------
class BatchState:
    def __init__(self):
        self.running: bool = False
        self.results: List[dict] = []          # serialized results, in order
        self.report: BatchRecoveryReport | None = None
        self.total_expected: int = 0

state = BatchState()

# Thread-safe queue: worker thread pushes serialized events here,
# an asyncio task drains it and fans out to all connected websockets.
_event_queue: asyncio.Queue[dict] | None = None   # initialised lazily inside the event loop
_active_sockets: List[WebSocket] = []
_main_loop: asyncio.AbstractEventLoop | None = None

app = FastAPI(title="AI Revenue Recovery Agent — Live Dashboard")


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.post("/api/assistant/chat")
async def assistant_chat(req: ChatRequest):
    reply = assistant.chat(
        message=req.message,
        history=req.history,
        results=state.results,
        report=state.report,
    )
    return {"reply": reply}

_broadcast_task_started = False

async def _ensure_broadcast_task():
    """Lazily starts the broadcast loop the first time it's needed,
    creating the Queue inside the running event loop to avoid
    'attached to a different event loop' errors on Python 3.10+."""
    global _main_loop, _broadcast_task_started, _event_queue
    if not _broadcast_task_started:
        _event_queue = asyncio.Queue()
        _main_loop = asyncio.get_running_loop()
        asyncio.create_task(_broadcast_loop())
        _broadcast_task_started = True


def _worker_run_batch(n: int, seed: int | None):
    reset_circuit_breaker()
    state.running = True
    state.results = []
    state.report = None
    state.total_expected = n

    try:
        transactions = generate_synthetic_batch(n=n, seed=seed)

        def on_event(serialized: dict):
            state.results.append(serialized)
            if _main_loop is not None:
                _main_loop.call_soon_threadsafe(_event_queue.put_nowait, serialized)

        raw_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_tx = {executor.submit(process_transaction, tx): tx for tx in transactions}
            for future in as_completed(future_to_tx):
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Transaction processing failed unexpectedly: {e}")
                    continue
                raw_results.append(result)
                on_event(serialize_result(result))

        state.report = build_report(raw_results)

        if _main_loop is not None:
            _main_loop.call_soon_threadsafe(
                _event_queue.put_nowait, {"__event__": "batch_complete", "report": state.report.model_dump()}
            )

    except Exception as e:
        logger.error(f"Batch run failed: {e}")
        if _main_loop is not None:
            _main_loop.call_soon_threadsafe(
                _event_queue.put_nowait, {"__event__": "batch_error", "error": str(e)}
            )

    finally:
        state.running = False  # ALWAYS resets, no matter what happened above


async def _broadcast_loop():
    """Drains the event queue and pushes messages to all connected sockets."""
    while True:
        event = await _event_queue.get()  # type: ignore[union-attr]  # always set before this runs
        dead = []
        for ws in _active_sockets:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in _active_sockets:
                _active_sockets.remove(ws)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.post("/api/batch/run")
async def run_batch(n: int = 35, seed: int | None = 42):
    await _ensure_broadcast_task()
    if state.running:
        return JSONResponse({"error": "A batch is already running."}, status_code=409)

    thread = threading.Thread(target=_worker_run_batch, args=(n, seed), daemon=True)
    thread.start()
    return {"status": "started", "n": n}


@app.get("/api/batch/results")
async def get_results():
    return {"running": state.running, "count": len(state.results), "results": state.results}


@app.get("/api/batch/report")
async def get_report():
    if state.report is None:
        return JSONResponse({"error": "No completed batch yet."}, status_code=404)
    return state.report.model_dump()


@app.get("/api/batch/status")
async def get_status():
    return {
        "running": state.running,
        "processed": len(state.results),
        "total_expected": state.total_expected,
    }
@app.post("/api/batch/reset")
async def reset_batch_state():
    reset_circuit_breaker()
    state.running = False
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Threshold endpoints
# ---------------------------------------------------------------------------

@app.get("/api/thresholds/current")
async def get_current_thresholds():
    """Return the live (read-only) SafetyThresholds values."""
    return {
        "MAX_RETRY_ATTEMPTS": THRESHOLDS.MAX_RETRY_ATTEMPTS,
        "ESCALATION_VALUE_INR": THRESHOLDS.ESCALATION_VALUE_INR,
        "NUDGE_COOLDOWN_HOURS": THRESHOLDS.NUDGE_COOLDOWN_HOURS,
        "MIN_RETRY_DELAY_HOURS": THRESHOLDS.MIN_RETRY_DELAY_HOURS,
        "MAX_RETRY_DELAY_HOURS": THRESHOLDS.MAX_RETRY_DELAY_HOURS,
        "HARD_STOP_ERROR_CODES": sorted(THRESHOLDS.HARD_STOP_ERROR_CODES),
        "ALLOWED_CHANNELS": sorted(THRESHOLDS.ALLOWED_CHANNELS),
    }


@app.post("/api/thresholds/analyze")
async def analyze_thresholds():
    """Run LLM-powered post-batch threshold analysis. Advisory only — never modifies config."""
    if state.report is None:
        return JSONResponse(
            {"error": "No completed batch available. Run a batch first."},
            status_code=404,
        )
    # Run the (blocking) LLM call in a thread so we don't block the event loop
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        threshold_analyzer.analyze_and_suggest,
        state.results,
        state.report,
    )
    return result


# ---------------------------------------------------------------------------
# Customer Risk Profile endpoint
# ---------------------------------------------------------------------------

def _compute_customer_profiles(results: List[dict]) -> List[Dict[str, Any]]:
    """Aggregate serialized batch results by customer_id and compute risk scores."""
    customers: Dict[str, Dict[str, Any]] = {}

    for r in results:
        cid = r.get("customer_id", "UNKNOWN")
        if cid not in customers:
            customers[cid] = {
                "customer_id": cid,
                "transaction_count": 0,
                "total_failed_amount_inr": 0.0,
                "total_recovered_inr": 0.0,
                "escalation_count": 0,
                "override_count": 0,
                "retry_count_sum": 0,
                "error_codes": set(),
                "actions_taken": set(),
                "transaction_ids": [],
                "segments": set(),
            }
        c = customers[cid]
        c["transaction_count"] += 1
        c["total_failed_amount_inr"] += r.get("amount_inr", 0)
        c["total_recovered_inr"] += r.get("recovered_inr", 0)
        if r.get("escalated"):
            c["escalation_count"] += 1
        if r.get("overridden"):
            c["override_count"] += 1
        c["retry_count_sum"] += r.get("retry_count", 0)
        if r.get("error_code"):
            c["error_codes"].add(r["error_code"])
        if r.get("action"):
            c["actions_taken"].add(r["action"])
        c["transaction_ids"].append(r.get("transaction_id", ""))
        if r.get("customer_segment"):
            c["segments"].add(r["customer_segment"])

    profiles = []
    for c in customers.values():
        n = c["transaction_count"]
        at_risk = c["total_failed_amount_inr"]
        recovered = c["total_recovered_inr"]
        recovery_rate = recovered / at_risk if at_risk > 0 else 0.0

        # Risk score formula (0-100)
        escalation_component = (c["escalation_count"] / n) * 40
        override_component = (c["override_count"] / n) * 20
        unrecovered_component = (1 - recovery_rate) * 30
        retry_component = min(10, (c["retry_count_sum"] / n) * 10)
        risk_score = round(
            min(100, escalation_component + override_component + unrecovered_component + retry_component)
        )

        if risk_score <= 25:
            risk_label = "Low"
        elif risk_score <= 50:
            risk_label = "Medium"
        elif risk_score <= 75:
            risk_label = "High"
        else:
            risk_label = "Critical"

        profiles.append({
            "customer_id": c["customer_id"],
            "transaction_count": n,
            "total_failed_amount_inr": round(at_risk, 2),
            "total_recovered_inr": round(recovered, 2),
            "recovery_rate_pct": round(recovery_rate * 100, 1),
            "escalation_count": c["escalation_count"],
            "override_count": c["override_count"],
            "error_codes": sorted(c["error_codes"]),
            "actions_taken": sorted(c["actions_taken"]),
            "transaction_ids": c["transaction_ids"],
            "segment": next(iter(c["segments"]), "unknown"),
            "risk_score": risk_score,
            "risk_label": risk_label,
        })

    return sorted(profiles, key=lambda x: -x["risk_score"])


@app.get("/api/customers/profiles")
async def get_customer_profiles():
    """Return per-customer aggregated risk profiles from the current batch."""
    if not state.results:
        return []
    return _compute_customer_profiles(state.results)

# ---------------------------------------------------------------------------
# WebSocket for live streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await _ensure_broadcast_task()
    await websocket.accept()
    _active_sockets.append(websocket)
    try:
        for r in state.results:
            await websocket.send_json(r)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _active_sockets:
            _active_sockets.remove(websocket)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")