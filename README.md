# ⚡ AI Revenue Recovery Agent

**An autonomous revenue recovery operations engine** — combining a tool-calling LLM agent, dual-layer safety guardrails, and a real-time Streamlit dashboard to detect, triage, and recover failed transactions without manual intervention.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://revenue-recovery-agent.streamlit.app/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35%2B-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

**[🌐 Try the live demo →](https://revenue-recovery-agent.streamlit.app/)**
*(Hosted on Streamlit Community Cloud — no auth required, runs on free-tier Gemini 2.5 Flash)*

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment](#deployment-streamlit-community-cloud)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why This Exists

Failed and disputed payments are one of the most expensive silent leaks in any revenue pipeline — most teams either retry blindly (burning customer goodwill) or triage manually (burning engineering time). This agent sits in between: it applies deterministic rules for the obvious cases, escalates to an LLM for judgment calls, and never acts outside the boundaries defined by its guardrails.

## Key Features

- **📊 Live Recovery Dashboard** — Real-time batch transaction streaming with glowing KPI cards, a dynamic SVG gradient recovery gauge, and interactive breakdown charts.
- **🛡️ Dual-Layer Safety Guardrails** — Hard-stop error code enforcement, maximum retry limits, value-based escalation triggers, and cooldown controls that sit outside the agent's own reasoning loop.
- **🤖 Deterministic Fast-Paths** — Clear-cut cases (e.g. known non-retryable error codes) are resolved by rules, not LLM calls — faster and cheaper, with the model reserved for ambiguous cases.
- **👤 Customer Risk Profiles** — Risk scoring (0–100) aggregating exposure, recovery yield, and escalation frequency, with live search and sort.
- **🎛️ Advisory Threshold Tuner** — Inspects batch friction points and override patterns to *propose* calibration adjustments, without ever auto-modifying production config.
- **💬 Recovery Assistant** — A read-only conversational agent with direct access to batch transactions, audit logs, and override rationales, so operators can ask questions in plain English instead of digging through logs.

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│  Transaction     │────▶│  Rule-Based Fast-Path │────▶│   Resolved / Retry  │
│  Batch Stream    │     │  (deterministic)      │     │                     │
└─────────────────┘     └──────────┬───────────┘     └────────────────────┘
                                    │ ambiguous case
                                    ▼
                         ┌──────────────────────┐
                         │   LLM Agent (Gemini)  │
                         │   + Safety Guardrails │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             Retry / Escalate  Risk Profile     Audit Log Entry
                                Update
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │  Streamlit Live Dashboard   │
                      │  + Recovery Assistant Chat   │
                      └───────────────────────────┘
```

Every LLM decision passes through the guardrail layer *before* it's allowed to act — hard-stop codes and retry caps are enforced in code, not left to model judgment.

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash / 2.0 Flash |
| Agent orchestration | Python, custom tool-calling loop |
| Backend/API | FastAPI, Uvicorn, WebSockets |
| Dashboard | Streamlit, Altair |
| Data | Pandas |
| Config & validation | Pydantic, python-dotenv |
| Deployment | Streamlit Community Cloud |

## Project Structure

```
revenue-recovery-agent/
├── streamlit_app.py                 # Root entry point — forwards to the app package
├── revenue-recovery-agent/          # Core application package
│   └── streamlit_app.py             # Actual dashboard + agent logic
├── .streamlit/                      # Streamlit theme/config
├── .env.example                     # Environment variable template
├── requirements.txt                 # Python dependencies
└── README.md
```

## Quick Start

### 1. Clone & Set Up

```bash
git clone https://github.com/mohith292005/revenue-recovery-agent.git
cd revenue-recovery-agent

# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
PRIMARY_MODEL=gemini-2.0-flash
APP_REFERER=https://your-app.example.com
APP_TITLE=AI Revenue Recovery Agent
```

### 3. Run

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

## Configuration

| Variable | Description | Example |
|---|---|---|
| `GOOGLE_API_KEY` | API key for Gemini access | `AIza...` |
| `PRIMARY_MODEL` | Gemini model used by the agent | `gemini-2.0-flash` |
| `APP_REFERER` | Referer URL sent with API requests | `https://your-app.example.com` |
| `APP_TITLE` | Display title for the app | `AI Revenue Recovery Agent` |

## Deployment (Streamlit Community Cloud)

1. Push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and create a **New App**.
3. Select this repository and branch.
4. Set the **Main file path** to `streamlit_app.py`.
5. Under **Advanced Settings → Secrets**, add:

   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key_here"
   PRIMARY_MODEL = "gemini-2.0-flash"
   ```

6. Click **Deploy**.

## Roadmap

- [ ] Pluggable payment-provider adapters (Stripe, Razorpay) beyond the simulated batch stream
- [ ] Persistent audit-log storage (currently in-memory per session)
- [ ] Configurable guardrail thresholds via the dashboard, with approval workflow before they take effect
- [ ] Unit tests for the rule-based fast-path and guardrail layer


---

Built by [Mohith](https://github.com/mohith292005) as part of an AI engineering portfolio.