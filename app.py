"""
MenkarAI — Streamlit UI for AI Video Assistant with RAG.
Mirrors main.py pipeline without modifying existing modules.
"""

from __future__ import annotations

import html
import io
import os
import re
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Heavy ML imports are intentionally deferred (lazy) and loaded only when the
# pipeline actually runs. Loading torch/whisper/chromadb at module level would
# cause Streamlit to timeout on low-RAM cloud hosts before rendering anything.

st.set_page_config(
    page_title="MenkarAI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

CURRENT_YEAR = datetime.now().year

PIPELINE_STEPS = [
    ("audio",      "Audio",      "Download & chunk"),
    ("transcribe", "Transcribe", "Whisper / Sarvam"),
    ("title",      "Title",      "AI title"),
    ("summary",    "Summary",    "Meeting summary"),
    ("extract",    "Insights",   "Actions & decisions"),
    ("rag",        "RAG Index",  "Vector store"),
]

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --primary:      #0a1f44;
            --primary-mid:  #1d4ed8;
            --accent:       #2563eb;
            --accent-light: #eff6ff;
            --accent-hover: #1e40af;
            --bg:           #f0f4f8;
            --surface:      #ffffff;
            --surface-2:    #f8fafc;
            --text:         #0f172a;
            --text-muted:   #475569;
            --border:       #cbd5e1;
            --border-light: #e2e8f0;
            --success:      #059669;
            --success-bg:   #ecfdf5;
            --success-dark: #047857;
            --active-bg:    #eff6ff;
            --orange:       #d97706;
            --orange-bg:    #fffbeb;
            --bottom-h:     58px;
            --radius:       10px;
            --shadow-sm:    0 1px 3px rgba(15,23,42,0.08), 0 1px 2px rgba(15,23,42,0.06);
            --shadow-md:    0 4px 16px rgba(15,23,42,0.10), 0 2px 4px rgba(15,23,42,0.06);
        }

        /* ── Hide Streamlit chrome ── */
        header[data-testid="stHeader"]    { display: none !important; }
        section[data-testid="stSidebar"],
        [data-testid="collapsedControl"]  { display: none !important; }
        #MainMenu, footer                 { visibility: hidden !important; }

        /* ── Root app ── */
        .stApp {
            background: var(--bg);
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            color: var(--text);
            font-size: 15px;
        }

        /* ── Page padding ── */
        .block-container {
            max-width: 100% !important;
            padding-top:    0.75rem !important;
            padding-left:   2rem  !important;
            padding-right:  2rem  !important;
            padding-bottom: calc(var(--bottom-h) + 1rem) !important;
        }

        /* ── Compact header ── */
        .main-header {
            background: linear-gradient(125deg, #0a1f44 0%, #1d4ed8 55%, #3b82f6 100%);
            padding: 0.75rem 2rem;
            border-radius: var(--radius);
            text-align: center;
            margin-bottom: 0.75rem;
            box-shadow: var(--shadow-md);
        }
        .main-header h1 {
            color: #ffffff !important;
            font-size: 1.65rem !important;
            font-weight: 800 !important;
            margin: 0 !important;
            letter-spacing: -0.02em;
            line-height: 1.2 !important;
        }
        .main-header p {
            color: #bfdbfe !important;
            font-size: 0.875rem !important;
            margin: 0.2rem 0 0 !important;
            line-height: 1.4 !important;
            font-weight: 400 !important;
        }

        /* ── Feature cards row ── */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.65rem;
            margin-bottom: 0.75rem;
        }
        @media (max-width: 900px) { .feature-grid { grid-template-columns: repeat(2,1fr); } }

        .feature-card {
            background: var(--surface);
            border: 1px solid var(--border-light);
            border-top: 3px solid var(--accent);
            border-radius: var(--radius);
            padding: 0.7rem 0.95rem;
            box-shadow: var(--shadow-sm);
            transition: box-shadow 0.15s;
        }
        .feature-card:hover { box-shadow: var(--shadow-md); }
        .feature-card h4 {
            color: var(--primary-mid);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 0 0 0.3rem 0;
        }
        .feature-card p {
            color: var(--text-muted);
            font-size: 0.85rem;
            margin: 0;
            line-height: 1.4;
            font-weight: 400;
        }

        /* ── Input card ── */
        .input-card {
            background: var(--surface);
            border: 2px solid var(--accent);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            box-shadow: var(--shadow-md);
        }
        .input-card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--primary);
            margin: 0 0 0.65rem 0;
            text-align: center;
            letter-spacing: -0.01em;
        }
        .field-lbl {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text);
            margin: 0.35rem 0 0.2rem;
            letter-spacing: 0.01em;
        }

        /* ── Text input ── */
        [data-testid="stTextInput"] input {
            height: 44px !important;
            min-height: 44px !important;
            padding: 0 0.875rem !important;
            font-size: 0.9rem !important;
            font-weight: 400 !important;
            border: 1.5px solid var(--border) !important;
            border-radius: 8px !important;
            background: var(--surface) !important;
            color: var(--text) !important;
            transition: border-color 0.15s, box-shadow 0.15s;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
        }
        [data-testid="stTextInput"] input::placeholder { color: #94a3b8 !important; }
        [data-testid="stTextInput"] > label { display: none !important; }

        /* Selectbox */
        [data-testid="stSelectbox"] div[data-baseweb="select"] {
            border-radius: 8px !important;
            min-height: 44px !important;
            font-size: 0.9rem !important;
        }
        [data-testid="stSelectbox"] > label { display: none !important; }

        /* File uploader */
        [data-testid="stFileUploader"] {
            border: 2px dashed var(--accent) !important;
            border-radius: 8px !important;
            padding: 0.5rem 0.75rem !important;
            background: var(--accent-light) !important;
            font-size: 0.875rem !important;
        }

        /* Radio */
        [data-testid="stRadio"] > label { display: none !important; }
        [data-testid="stRadio"] > div {
            flex-direction: row !important;
            gap: 1.75rem !important;
            justify-content: center;
        }
        [data-testid="stRadio"] label span {
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            color: var(--text) !important;
        }

        /* Run button */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"] {
            height: 44px !important;
            min-height: 44px !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            background: linear-gradient(135deg, var(--primary-mid) 0%, #3b82f6 100%) !important;
            color: #fff !important;
            border: none !important;
            width: 100% !important;
            letter-spacing: 0.01em;
            box-shadow: 0 2px 8px rgba(37,99,235,0.35) !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, var(--accent-hover) 0%, var(--accent) 100%) !important;
            box-shadow: 0 4px 14px rgba(37,99,235,0.45) !important;
        }

        /* ── Fixed bottom bar ── */
        .bottom-bar {
            position: fixed;
            bottom: 0; left: 0; right: 0;
            height: var(--bottom-h);
            background: var(--primary);
            color: #cbd5e1;
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 1.75rem;
            font-size: 0.8rem;
            gap: 0.5rem;
            border-top: 2px solid var(--accent);
        }
        .bottom-bar .contact { color: #fff; font-weight: 700; font-size: 0.85rem; }
        .bottom-bar a        { color: #93c5fd; text-decoration: none; }
        .bottom-bar .api-info{ color: #94a3b8; font-size: 0.75rem; }
        .key-ok   { color: #34d399; font-weight: 700; }
        .key-miss { color: #fbbf24; font-weight: 700; }

        /* ── Progress tracker ── */
        .progress-wrapper {
            background: var(--surface);
            border: 1px solid var(--border-light);
            border-radius: var(--radius);
            padding: 0.85rem 1.1rem;
            margin: 0.6rem 0;
            box-shadow: var(--shadow-sm);
        }
        .progress-title {
            font-size: 0.9rem; font-weight: 700;
            color: var(--primary); margin: 0 0 0.65rem 0;
        }
        .progress-steps {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 0.45rem;
        }
        @media (max-width: 900px) { .progress-steps { grid-template-columns: repeat(3,1fr); } }

        .progress-step {
            text-align: center;
            padding: 0.45rem 0.3rem;
            border-radius: 8px;
            border: 1px solid var(--border-light);
            background: var(--surface-2);
            font-size: 0.7rem;
        }
        .progress-step.done   { background: var(--success-bg); border-color: #6ee7b7; }
        .progress-step.active { background: var(--active-bg);  border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(37,99,235,0.15); }

        .step-circle {
            width: 28px; height: 28px; border-radius: 50%;
            margin: 0 auto 0.3rem;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 0.75rem;
            background: var(--border); color: var(--text-muted);
        }
        .progress-step.done   .step-circle { background: var(--success); color: #fff; }
        .progress-step.active .step-circle {
            background: var(--accent); color: #fff;
            animation: pulse 1.4s ease-in-out infinite;
        }
        @keyframes pulse {
            0%   { box-shadow: 0 0 0 0   rgba(37,99,235,0.45); }
            70%  { box-shadow: 0 0 0 8px rgba(37,99,235,0); }
            100% { box-shadow: 0 0 0 0   rgba(37,99,235,0); }
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .step-circle.spinner { background: var(--accent); position: relative; }
        .step-circle.spinner::after {
            content: ''; position: absolute;
            width: 13px; height: 13px;
            border: 2px solid rgba(255,255,255,0.35);
            border-top-color: #fff; border-radius: 50%;
            animation: spin 0.75s linear infinite;
        }
        .step-name { font-weight: 700; color: var(--primary); font-size: 0.72rem; }
        .step-desc { color: var(--text-muted); font-size: 0.63rem; margin-top: 0.1rem; }
        .progress-step.done .step-name { color: var(--success-dark); }

        /* ── Overview title card ── */
        .title-card {
            background: linear-gradient(125deg, #0a1f44 0%, #1d4ed8 55%, #3b82f6 100%);
            border-radius: 12px;
            padding: 1.1rem 1.5rem;
            margin-bottom: 1rem;
            color: #fff;
            box-shadow: var(--shadow-md);
        }
        .title-card .tc-label {
            font-size: 0.7rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.1em;
            color: #93c5fd; margin-bottom: 0.35rem;
        }
        .title-card .tc-title {
            font-size: 1.4rem; font-weight: 800;
            line-height: 1.25; color: #fff;
            letter-spacing: -0.02em;
        }

        /* ── Stats row ── */
        .stats-row {
            display: flex; gap: 0.85rem;
            margin-bottom: 1rem; flex-wrap: wrap;
        }
        .stat-box {
            background: var(--surface);
            border: 1px solid var(--border-light);
            border-radius: var(--radius);
            padding: 0.65rem 1.1rem;
            flex: 1; min-width: 100px; text-align: center;
            box-shadow: var(--shadow-sm);
        }
        .stat-val {
            font-size: 1.35rem; font-weight: 800;
            color: var(--primary-mid); line-height: 1.2;
            letter-spacing: -0.02em;
        }
        .stat-lbl {
            font-size: 0.68rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.06em;
            color: var(--text-muted); margin-top: 0.2rem;
        }

        /* ── Result panel wrapper ── */
        .panel-header {
            font-size: 0.9rem; font-weight: 700;
            color: var(--primary);
            padding: 0.6rem 0 0.45rem 0;
            border-bottom: 2px solid var(--accent);
            margin-bottom: 0.65rem;
            letter-spacing: 0.01em;
        }
        .result-panel-wrap {
            background: var(--surface);
            border: 1px solid var(--border-light);
            border-radius: var(--radius);
            padding: 1rem 1.25rem 0.6rem;
            margin-bottom: 0.75rem;
            box-shadow: var(--shadow-sm);
        }
        /* Markdown inside panels */
        .result-panel-wrap p, .result-panel-wrap li {
            font-size: 0.925rem !important;
            line-height: 1.7 !important;
            color: var(--text) !important;
        }
        .result-panel-wrap h1, .result-panel-wrap h2, .result-panel-wrap h3 {
            color: var(--primary) !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
            margin-top: 0.5rem !important;
        }
        .result-panel-wrap strong { color: var(--primary-mid) !important; font-weight: 700 !important; }
        .result-panel-wrap ul, .result-panel-wrap ol { padding-left: 1.25rem !important; }

        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, var(--success-dark), var(--success)) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            height: 40px !important;
            box-shadow: 0 2px 8px rgba(5,150,105,0.3) !important;
        }

        /* ── Tabs ── */
        [data-testid="stTabs"] button {
            font-size: 0.9rem !important;
            font-weight: 600 !important;
        }

        /* ── Chat ── */
        .chat-user, .chat-assistant {
            padding: 0.8rem 1.1rem;
            border-radius: 0 8px 8px 0;
            margin: 0.45rem 0;
            line-height: 1.6;
            font-size: 0.925rem;
        }
        .chat-user      { background: var(--accent-light); border-left: 4px solid var(--accent); }
        .chat-assistant { background: var(--success-bg);   border-left: 4px solid var(--success); }

        /* ── Secondary button ── */
        .stButton > button[kind="secondary"],
        .stButton > button[data-testid="stBaseButton-secondary"] {
            font-size: 0.875rem !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            border: 1.5px solid var(--border) !important;
            color: var(--text) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _init_session_state() -> None:
    for key, default in {
        "pipeline_result": None,
        "chat_history":    [],
        "source_label":    "",
        "is_processing":   False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _save_uploaded_file(uploaded_file) -> str:
    path = os.path.join(DOWNLOAD_DIR, os.path.basename(uploaded_file.name))
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def _check_api_keys(language: str) -> list[str]:
    missing = []
    if language == "hinglish" and not os.getenv("SARVAM_API_KEY"):
        missing.append("SARVAM_API_KEY")
    if not os.getenv("MISTRAL_API_KEY"):
        missing.append("MISTRAL_API_KEY")
    return missing


# ─────────────────────────────────────────────────────────────────────────────
# Fixed bottom bar (contact + API status)
# ─────────────────────────────────────────────────────────────────────────────
def _render_bottom_bar() -> None:
    mk = bool(os.getenv("MISTRAL_API_KEY"))
    sk = bool(os.getenv("SARVAM_API_KEY"))
    wm = html.escape(os.getenv("WHISPER_MODEL", "base"))
    st.markdown(
        f"""
        <div class="bottom-bar">
            <span class="contact">
                Swapnil Menkar &nbsp;|&nbsp;
                Mobile: <a href="tel:+918149005578">8149005578</a>
            </span>
            <span class="api-info">
                MISTRAL <span class="{'key-ok' if mk else 'key-miss'}">{'✓' if mk else '✗'}</span>&nbsp;
                SARVAM <span class="{'key-ok' if sk else 'key-miss'}">{'✓' if sk else '✗'}</span>&nbsp;
                WHISPER={wm}
            </span>
            <span>&copy; {CURRENT_YEAR} MenkarAI. All rights reserved.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Progress tracker
# ─────────────────────────────────────────────────────────────────────────────
def _progress_html(states: dict, title: str) -> str:
    rows = []
    for i, (key, name, desc) in enumerate(PIPELINE_STEPS, 1):
        s = states.get(key, "pending")
        css = s if s in ("done", "active") else ""
        if s == "done":
            circ = '<div class="step-circle">✓</div>'
        elif s == "active":
            circ = '<div class="step-circle spinner"></div>'
        else:
            circ = f'<div class="step-circle">{i}</div>'
        rows.append(
            f'<div class="progress-step {css}">{circ}'
            f'<div class="step-name">{html.escape(name)}</div>'
            f'<div class="step-desc">{html.escape(desc)}</div></div>'
        )
    return (
        f'<div class="progress-wrapper">'
        f'<div class="progress-title">{html.escape(title)}</div>'
        f'<div class="progress-steps">{"".join(rows)}</div></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline (stepped)
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline_stepped(source: str, language: str, slot, bar) -> dict:
    states = {k: "pending" for k, _, _ in PIPELINE_STEPS}
    total = len(PIPELINE_STEPS)

    def _step(key: str, idx: int, label: str) -> None:
        # mark previous active as done
        for k, _, _ in PIPELINE_STEPS:
            if states.get(k) == "active" and k != key:
                states[k] = "done"
        states[key] = "active"
        slot.markdown(_progress_html(states, label), unsafe_allow_html=True)
        bar.progress(min(idx / total, 1.0))

    _step("audio", 0, "Step 1/6 — Processing audio…")
    # ── Lazy imports: loaded here so the home page renders without touching torch/whisper
    from utils.audio_processor import process_input
    from core.transcriber_with_sarvam import transcribe_all
    from core.summarizer import generate_title, summarize
    from core.extractor import extract_action_items, extract_key_decisions, extract_questions
    from core.rag_engine import build_rag_chain
    chunks = process_input(source)
    states["audio"] = "done"

    _step("transcribe", 1, "Step 2/6 — Transcribing…")
    transcript = transcribe_all(chunks, language)
    states["transcribe"] = "done"

    _step("title", 2, "Step 3/6 — Generating title…")
    title = generate_title(transcript)
    states["title"] = "done"

    _step("summary", 3, "Step 4/6 — Creating summary…")
    summary = summarize(transcript)
    states["summary"] = "done"

    _step("extract", 4, "Step 5/6 — Extracting insights…")
    action_items  = extract_action_items(transcript)
    key_decisions = extract_key_decisions(transcript)
    open_questions = extract_questions(transcript)
    states["extract"] = "done"

    _step("rag", 5, "Step 6/6 — Building RAG index…")
    rag_chain = build_rag_chain(transcript)
    states["rag"] = "done"

    slot.markdown(_progress_html(states, "Analysis complete ✓"), unsafe_allow_html=True)
    bar.progress(1.0)

    return {
        "title": title, "transcript": transcript, "summary": summary,
        "action_items": action_items, "key_decisions": key_decisions,
        "open_questions": open_questions, "rag_chain": rag_chain,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reusable result panel  (renders markdown properly)
# ─────────────────────────────────────────────────────────────────────────────
def _result_panel(title: str, content: str) -> None:
    st.markdown(f'<div class="result-panel-wrap"><div class="panel-header">{html.escape(title)}</div>', unsafe_allow_html=True)
    st.markdown(content or "_No content._")
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PDF generation
# ─────────────────────────────────────────────────────────────────────────────
def _md_to_plain(text: str) -> str:
    """Strip markdown and normalise unicode to ASCII-safe Latin-1 for fpdf2 Helvetica."""
    # Remove markdown syntax
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*(.+?)\*",     r"\1", text, flags=re.DOTALL)
    text = re.sub(r"#{1,6}\s*",     "",    text)
    text = re.sub(r"`(.+?)`",        r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)

    # Replace common unicode punctuation with ASCII equivalents
    replacements = {
        "\u2014": "--",   # em dash
        "\u2013": "-",    # en dash
        "\u2018": "'",    "\u2019": "'",   # smart single quotes
        "\u201c": '"',    "\u201d": '"',   # smart double quotes
        "\u2022": "*",    "\u2023": "*",   # bullet variants
        "\u25cf": "*",    "\u25e6": "*",
        "\u2026": "...",  # ellipsis
        "\u00b7": ".",    # middle dot
        "\u2019": "'",    "\u0060": "'",
        "\u00a0": " ",    # non-breaking space
        "\u2192": "->",   "\u21d2": "=>",  # arrows
        "\u00d7": "x",    "\u00f7": "/",
        "\u00e9": "e",    "\u00e8": "e",   "\u00ea": "e",
        "\u00e0": "a",    "\u00e2": "a",   "\u00fc": "u",
        "\u00f6": "o",    "\u00e4": "a",
        "\u20b9": "Rs.",  # Rupee sign
        "\u20ac": "EUR",  # Euro
        "\u00a3": "GBP",  # Pound
        "\u00a5": "JPY",  # Yen
        "\u00a9": "(c)",  "\u00ae": "(R)", "\u2122": "(TM)",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Final pass: drop any remaining non-Latin-1 characters
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text.strip()


def _generate_pdf(result: dict) -> bytes:
    from fpdf import FPDF

    # ── Colour palette (mirrors the app CSS variables) ─────────────────────
    C_NAVY      = (10,  31,  68)   # --primary     #0a1f44
    C_BLUE      = (29,  78, 216)   # --primary-mid #1d4ed8
    C_ACCENT    = (37,  99, 235)   # --accent      #2563eb
    C_ACCENT_L  = (239, 246, 255)  # --accent-light#eff6ff
    C_BG        = (240, 244, 248)  # --bg          #f0f4f8
    C_TEXT      = (15,  23,  42)   # --text        #0f172a
    C_MUTED     = (71,  85, 105)   # --text-muted  #475569
    C_BORDER    = (203, 213, 225)  # --border      #cbd5e1
    C_SUCCESS   = (5,  150, 105)   # --success     #059669
    C_SUCC_BG   = (236, 253, 245)  # --success-bg  #ecfdf5
    C_WHITE     = (255, 255, 255)
    C_LT_BLUE   = (191, 219, 254)  # #bfdbfe
    C_SKY       = (147, 197, 253)  # #93c5fd
    C_SLATE     = (148, 163, 184)  # #94a3b8

    PAGE_W = 210
    PAGE_H = 297
    MARGIN = 15
    HDR_H  = 24   # header banner height (mm)
    FTR_H  = 14   # footer banner height (mm)
    CW     = PAGE_W - 2 * MARGIN   # usable content width

    class _PDF(FPDF):
        # ── Page header (navy banner with brand + tagline + page number) ──
        def header(self):
            # Navy background banner
            self.set_fill_color(*C_NAVY)
            self.rect(0, 0, PAGE_W, HDR_H, style="F")
            # Blue accent strip at banner bottom
            self.set_fill_color(*C_ACCENT)
            self.rect(0, HDR_H - 1.5, PAGE_W, 1.5, style="F")

            # Brand name "MenkarAI" – left
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(*C_WHITE)
            self.set_xy(MARGIN, 6)
            self.cell(70, 9, "MenkarAI", align="L")

            # Tagline – right
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(*C_LT_BLUE)
            self.set_xy(PAGE_W - MARGIN - 92, 7)
            self.cell(92, 5, "AI Video Assistant -- Analysis Report", align="R")

            # Page number – bottom-right of banner
            self.set_font("Helvetica", "B", 7.5)
            self.set_text_color(*C_SKY)
            self.set_xy(PAGE_W - MARGIN - 18, 15)
            self.cell(18, 5, f"Page {self.page_no()}", align="R")

            # Position cursor just below banner
            self.set_xy(MARGIN, HDR_H + 5)

        # ── Page footer (navy banner with contact + gen-info + copyright) ─
        def footer(self):
            fy = PAGE_H - FTR_H
            # Navy background banner
            self.set_fill_color(*C_NAVY)
            self.rect(0, fy, PAGE_W, FTR_H, style="F")
            # Blue accent strip at banner top
            self.set_fill_color(*C_ACCENT)
            self.rect(0, fy, PAGE_W, 1.5, style="F")

            # Contact info – left (bold white)
            self.set_font("Helvetica", "B", 7.5)
            self.set_text_color(*C_WHITE)
            self.set_xy(MARGIN, fy + 4)
            self.cell(72, 5, "Swapnil Menkar  |  Mobile: 8149005578", align="L")

            # Generated info – centre
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*C_SLATE)
            self.set_xy(PAGE_W / 2 - 38, fy + 4)
            self.cell(76, 5,
                      f"Generated by MenkarAI  |  "
                      f"{datetime.now().strftime('%d %b %Y, %I:%M %p')}",
                      align="C")

            # Copyright – right
            self.set_xy(PAGE_W - MARGIN - 54, fy + 4)
            self.cell(54, 5,
                      f"(c) {datetime.now().year} MenkarAI. All rights reserved.",
                      align="R")

    # ── PDF instance ──────────────────────────────────────────────────────
    pdf = _PDF(orientation="P", unit="mm", format="A4")
    # Bottom auto-break margin keeps content above the footer banner
    pdf.set_auto_page_break(auto=True, margin=FTR_H + 6)
    # Top margin = just below the header banner
    pdf.set_margins(MARGIN, HDR_H + 5, MARGIN)
    pdf.add_page()

    # ── Title card (dark navy gradient card mimicking webpage) ────────────
    tc_y = pdf.get_y()
    tc_h = 24
    # Navy card background
    pdf.set_fill_color(*C_NAVY)
    pdf.rect(MARGIN, tc_y, CW, tc_h, style="F")
    # Blue left accent bar
    pdf.set_fill_color(*C_ACCENT)
    pdf.rect(MARGIN, tc_y, 4, tc_h, style="F")
    # "CONTENT TITLE" micro-label
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(*C_SKY)
    pdf.set_xy(MARGIN + 7, tc_y + 3.5)
    pdf.cell(CW - 7, 4, "CONTENT TITLE", align="L")
    # Main title text
    title_text = _md_to_plain(result.get("title", "Analysis Report"))
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*C_WHITE)
    pdf.set_xy(MARGIN + 7, tc_y + 9)
    pdf.cell(CW - 7, 9, title_text, align="L")
    pdf.set_y(tc_y + tc_h + 5)

    # ── Stats row (4 metric boxes) ────────────────────────────────────────
    transcript     = result.get("transcript",    "")
    action_lines   = [l for l in result.get("action_items",  "").split("\n") if l.strip()]
    decision_lines = [l for l in result.get("key_decisions", "").split("\n") if l.strip()]
    stats = [
        ("Words",        f"{len(transcript.split()):,}"),
        ("Characters",   f"{len(transcript):,}"),
        ("Action Items", str(len(action_lines))),
        ("Decisions",    str(len(decision_lines))),
    ]
    gap  = 2.5
    sw   = (CW - gap * 3) / 4   # box width
    sy   = pdf.get_y()
    sh   = 16
    for i, (lbl, val) in enumerate(stats):
        sx = MARGIN + i * (sw + gap)
        # White card with border
        pdf.set_fill_color(*C_WHITE)
        pdf.set_draw_color(*C_BORDER)
        pdf.rect(sx, sy, sw, sh, style="FD")
        # Blue top accent strip
        pdf.set_fill_color(*C_ACCENT)
        pdf.rect(sx, sy, sw, 2, style="F")
        # Metric value
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*C_BLUE)
        pdf.set_xy(sx, sy + 3)
        pdf.cell(sw, 7, val, align="C")
        # Label
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_text_color(*C_MUTED)
        pdf.set_xy(sx, sy + 10)
        pdf.cell(sw, 5, lbl.upper(), align="C")
    pdf.set_y(sy + sh + 5)

    # ── Generation meta line ──────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*C_MUTED)
    pdf.set_x(MARGIN)
    pdf.cell(
        0, 5,
        f"Analysis generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}  "
        f"|  Words: {len(transcript.split()):,}",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(4)

    # ── Section helper ────────────────────────────────────────────────────
    def _section(label: str, body: str) -> None:
        # Blue left accent bar + light-blue label header
        bar_y = pdf.get_y()
        pdf.set_fill_color(*C_ACCENT)
        pdf.rect(MARGIN, bar_y, 3.5, 8.5, style="F")
        pdf.set_fill_color(*C_ACCENT_L)
        pdf.set_text_color(*C_BLUE)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_xy(MARGIN + 3.5, bar_y)
        pdf.cell(CW - 3.5, 8.5, f"  {label}", fill=True,
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        # Body text
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*C_TEXT)
        pdf.set_x(MARGIN)
        pdf.multi_cell(CW, 5.8, _md_to_plain(body))
        pdf.ln(5)

    _section("SUMMARY",        result.get("summary",        ""))
    _section("ACTION ITEMS",   result.get("action_items",   ""))
    _section("KEY DECISIONS",  result.get("key_decisions",  ""))
    _section("OPEN QUESTIONS", result.get("open_questions", ""))

    pdf.add_page()
    _section("FULL TRANSCRIPT", transcript)

    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────────────
# Home page — single viewport
# ─────────────────────────────────────────────────────────────────────────────
def _render_home() -> tuple[str | None, str, str, bool]:
    # ── Compact header ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="main-header">
            <h1>MenkarAI</h1>
            <p>AI Video Assistant &mdash; Transcribe &middot; Summarize &middot; Extract Insights &middot; RAG Chat</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Cloud-mode notice ────────────────────────────────────────────────────
    if os.getenv("RENDER", "").lower() in ("1", "true", "yes"):
        st.info(
            "☁️ **Cloud mode** — This deployment uses Sarvam AI for transcription "
            "(cloud API, no local model). Please select **Hinglish — Sarvam AI** "
            "as the language. English/Whisper (local model) is not available on "
            "this instance due to memory constraints.",
            icon="ℹ️",
        )

    # ── Feature cards (original text) ───────────────────────────────────────
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <h4>① Audio Input</h4>
                <p>YouTube URL or local audio/video file upload</p>
            </div>
            <div class="feature-card">
                <h4>② Transcription</h4>
                <p>Whisper (English) or Sarvam AI (Hinglish → English)</p>
            </div>
            <div class="feature-card">
                <h4>③ AI Analysis</h4>
                <p>Title, summary, action items, decisions & open questions</p>
            </div>
            <div class="feature-card">
                <h4>④ RAG Chat</h4>
                <p>Ask questions grounded in your transcript context</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Input card — two-column layout so nothing is cut off ────────────────
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-title">Start Your Analysis</div>', unsafe_allow_html=True)

    left, right = st.columns([1.45, 1], gap="large")

    source: str | None = None
    source_label = ""
    language = "english"
    run_clicked = False

    with left:
        input_mode = st.radio(
            "source_mode",
            ["YouTube URL", "Upload local file"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if input_mode == "YouTube URL":
            st.markdown('<p class="field-lbl">YouTube URL</p>', unsafe_allow_html=True)
            url = st.text_input(
                "yt_url",
                placeholder="https://www.youtube.com/watch?v=...",
                label_visibility="collapsed",
            )
            if url.strip():
                source = url.strip()
                source_label = source
        else:
            st.markdown('<p class="field-lbl">Audio / Video file</p>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "file_up",
                type=["mp3", "mp4", "wav", "m4a", "webm", "ogg", "flac", "avi", "mkv"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                source = _save_uploaded_file(uploaded)
                source_label = uploaded.name

    with right:
        st.markdown('<p class="field-lbl">Language</p>', unsafe_allow_html=True)
        language = st.selectbox(
            "lang_sel",
            options=["english", "hinglish"],
            index=0,
            label_visibility="collapsed",
            format_func=lambda x: (
                "English — Whisper (local)" if x == "english"
                else "Hinglish — Sarvam AI"
            ),
        )

        st.markdown('<p class="field-lbl">&nbsp;</p>', unsafe_allow_html=True)
        btn_slot = st.empty()
        run_clicked = btn_slot.button(
            "▶  Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("is_processing", False),
        )

    st.markdown("</div>", unsafe_allow_html=True)   # close input-card
    return source, source_label, language, run_clicked, btn_slot


# ─────────────────────────────────────────────────────────────────────────────
# Results page
# ─────────────────────────────────────────────────────────────────────────────
def _render_results(result: dict) -> None:
    col_back, col_src = st.columns([1, 5])
    with col_back:
        if st.button("← New Analysis", type="secondary"):
            st.session_state.pipeline_result = None
            st.session_state.chat_history = []
            st.session_state.source_label = ""
            st.rerun()
    with col_src:
        if st.session_state.source_label:
            st.caption(f"Source: {st.session_state.source_label}")

    tab_overview, tab_transcript, tab_insights, tab_chat = st.tabs(
        ["Overview", "Transcript", "Insights", "RAG Chat"]
    )

    with tab_overview:
        # Title card
        st.markdown(
            f'<div class="title-card">'
            f'<div class="tc-label">Meeting Title</div>'
            f'<div class="tc-title">{html.escape(result["title"])}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Stats row
        words    = len(result["transcript"].split())
        chars    = len(result["transcript"])
        actions  = result["action_items"].count("\n1") + result["action_items"].count("1.")
        actions  = max(actions, result["action_items"].count("\n-"), 1)
        decisions = result["key_decisions"].count("\n-") + 1

        st.markdown(
            f'<div class="stats-row">'
            f'<div class="stat-box"><div class="stat-val">{words:,}</div><div class="stat-lbl">Words</div></div>'
            f'<div class="stat-box"><div class="stat-val">{chars:,}</div><div class="stat-lbl">Characters</div></div>'
            f'<div class="stat-box"><div class="stat-val">{actions}</div><div class="stat-lbl">Action Items</div></div>'
            f'<div class="stat-box"><div class="stat-val">{decisions}</div><div class="stat-lbl">Decisions</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Download PDF button
        try:
            pdf_bytes = _generate_pdf(result)
            safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", result["title"])[:40] or "report"
            st.download_button(
                label="⬇  Download PDF Report",
                data=pdf_bytes,
                file_name=f"MenkarAI_{safe_title}.pdf",
                mime="application/pdf",
                use_container_width=False,
            )
        except Exception as e:
            st.warning(f"PDF generation unavailable: {e}")

        st.divider()
        _result_panel("Summary", result["summary"])

    with tab_transcript:
        _result_panel("Full Transcript", result["transcript"])

    with tab_insights:
        # Download PDF also accessible from Insights
        try:
            pdf_bytes = _generate_pdf(result)
            safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", result["title"])[:40] or "report"
            st.download_button(
                "⬇  Download PDF Report",
                data=pdf_bytes,
                file_name=f"MenkarAI_{safe_title}.pdf",
                mime="application/pdf",
            )
        except Exception:
            pass

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            _result_panel("Action Items", result["action_items"])
        with c2:
            _result_panel("Key Decisions", result["key_decisions"])
        _result_panel("Open Questions", result["open_questions"])

    with tab_chat:
        st.markdown("Chat with your meeting — answers are grounded via RAG.")
        for msg in st.session_state.chat_history:
            css   = "chat-user" if msg["role"] == "user" else "chat-assistant"
            label = "You"       if msg["role"] == "user" else "Assistant"
            st.markdown(
                f'<div class="{css}"><strong>{label}:</strong> {html.escape(msg["content"])}</div>',
                unsafe_allow_html=True,
            )
        with st.form("chat_form", clear_on_submit=True):
            q = st.text_input("Q", placeholder="Ask anything about the transcript…", label_visibility="collapsed")
            sent = st.form_submit_button("Send", use_container_width=True)
        if sent and q.strip():
            try:
                with st.spinner("Thinking…"):
                    from core.rag_engine import ask_question
                    ans = ask_question(result["rag_chain"], q.strip())
                st.session_state.chat_history.extend([
                    {"role": "user",      "content": q.strip()},
                    {"role": "assistant", "content": ans},
                ])
                st.rerun()
            except Exception as exc:
                st.error(f"Chat failed: {exc}")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    _init_session_state()
    _render_bottom_bar()

    result = st.session_state.pipeline_result
    if result is not None:
        _render_results(result)
        return

    source, source_label, language, run_clicked, btn_slot = _render_home()

    if run_clicked:
        if not source:
            st.error("Please enter a YouTube URL or upload an audio/video file.")
        else:
            missing = _check_api_keys(language)
            if missing:
                st.error(f"Missing env vars: {', '.join(missing)} — configure your .env (see bottom bar).")
            else:
                # Grey out the button immediately — visible for the entire pipeline run
                btn_slot.button(
                    "⏳  Processing…",
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                    key="_btn_disabled",
                )
                slot = st.empty()
                bar  = st.progress(0)
                st.session_state.is_processing = True
                try:
                    result = run_pipeline_stepped(source, language, slot, bar)
                    st.session_state.pipeline_result = result
                    st.session_state.source_label    = source_label
                    st.session_state.chat_history    = []
                    time.sleep(0.2)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Pipeline failed: {exc}")
                finally:
                    st.session_state.is_processing = False


if __name__ == "__main__":
    main()
