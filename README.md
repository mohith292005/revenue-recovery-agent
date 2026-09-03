# ⚡ AI Revenue Recovery Agent

An intelligent, autonomous revenue recovery operations engine featuring dual-layer guardrails, deterministic rule fast-paths, adaptive safety threshold tuning, and a real-time live dashboard built with Streamlit.

---

## 🌟 Key Features

- **📊 Live Recovery Dashboard**: Real-time batch transaction streaming, glowing KPI cards, dynamic SVG gradient recovery gauge, and interactive breakdown charts.
- **🛡️ Dual-Layer Safety Guardrails**: Hard-stop error code enforcement, maximum retry limits, value-based escalation triggers, and cooldown controls.
- **👤 Customer Risk Profiles**: Risk scoring (0–100) aggregating exposure, recovery yield, and escalation frequency with live search and sort.
- **🎛️ Advisory Threshold Tuner**: Inspects batch friction points and override patterns to propose calibration adjustments without auto-modifying production config.
- **💬 Recovery Assistant**: Read-only conversational agent with direct access to batch transactions, audit logs, and override rationales.

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone <your-repo-url>
cd "AI revenue agent"

# Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and add your Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GOOGLE_API_KEY=your_gemini_api_key_here
PRIMARY_MODEL=gemini-2.0-flash
```

### 3. Run the Application
```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501` in your browser.

---

## ☁️ Live Cloud Deployment (Streamlit Community Cloud)

1. Push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and create a **New App**.
3. Select this repository and branch.
4. Set the **Main file path** to `streamlit_app.py`.
5. Under **Advanced Settings > Secrets**, configure:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key_here"
   PRIMARY_MODEL = "gemini-2.0-flash"
   ```
6. Click **Deploy!**
