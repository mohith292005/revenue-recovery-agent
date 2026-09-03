"""
streamlit_app.py
Root entry point for live deployment on Streamlit Community Cloud, HuggingFace Spaces, or local execution.
Automatically forwards execution to revenue-recovery-agent/streamlit_app.py.
"""

import os
import sys
from pathlib import Path

# Path to the revenue-recovery-agent package
AGENT_DIR = Path(__file__).resolve().parent / "revenue-recovery-agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# Ensure working directory is set so relative database and asset paths resolve properly
os.chdir(str(AGENT_DIR))

# Execute the main streamlit application
app_path = AGENT_DIR / "streamlit_app.py"
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, str(app_path), "exec"), globals())
