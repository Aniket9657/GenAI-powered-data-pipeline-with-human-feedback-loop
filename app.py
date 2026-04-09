# app.py — GenAI Image Description Processor
# Requirements: streamlit, requests, Pillow
# Run: streamlit run app.py

import json
import re
import sqlite3
import datetime
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

# ─── Config ───────────────────────────────────────────────────────────────────

DB_PATH = "responses.db"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# ─── Database ─────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the SQLite database and responses table if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_text     TEXT    NOT NULL,
            processed_json TEXT   NOT NULL,
            timestamp     TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_response(user_text: str, processed_json: str) -> int:
    """Insert a processed response into the database; return the new row id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "INSERT INTO responses (user_text, processed_json, timestamp) VALUES (?, ?, ?)",
        (user_text, processed_json, datetime.datetime.utcnow().isoformat()),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def fetch_recent(limit: int = 5) -> list[tuple]:
    """Return the most recent `limit` rows from the responses table."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, user_text, processed_json, timestamp FROM responses ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return rows

# ─── Ollama / GenAI ───────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a text processing engine. Analyze the user's image description below and return ONLY a JSON object — no explanation, no markdown, no extra text.

User description: "{user_text}"

Return exactly this JSON structure:
{{
  "cleaned_text": "<grammatically corrected, slang-free version of the description>",
  "objects": ["<object1>", "<object2>", "..."],
  "emotion": "<single word capturing dominant emotional tone>",
  "theme": "<single word capturing central theme>",
  "quality_score": <integer 1-10 based on clarity, relevance, descriptiveness>
}}

Rules:
- cleaned_text: fix grammar, spelling, remove slang/emojis, keep meaning intact
- objects: list every distinct physical object or element mentioned or clearly implied
- emotion: ONE word only (e.g. serene, joyful, melancholic, awe, tense)
- theme: ONE word only (e.g. nature, solitude, celebration, decay, growth)
- quality_score: 1 = incomprehensible, 10 = exceptionally clear and descriptive
- Output ONLY the JSON object. Nothing else."""


def call_ollama(user_text: str) -> dict:
    """
    Send the user's description to Ollama and parse the JSON response.
    Returns a dict on success, or raises an exception on failure.
    """
    prompt = PROMPT_TEMPLATE.format(user_text=user_text.replace('"', '\\"'))

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temp for deterministic JSON
            "num_predict": 512,
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()

    raw_text = response.json().get("response", "")
    return parse_json_safely(raw_text)


def parse_json_safely(raw: str) -> dict:
    """
    Extract and parse a JSON object from raw LLM output.
    Handles cases where the model wraps JSON in markdown fences or adds prose.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to extract the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return a structured error object so the app never crashes
    return {
        "cleaned_text": raw.strip()[:500] if raw.strip() else "Could not process text.",
        "objects": [],
        "emotion": "unknown",
        "theme": "unknown",
        "quality_score": 0,
        "_parse_error": "JSON extraction failed — raw model output shown in cleaned_text.",
    }

# ─── UI Helpers ───────────────────────────────────────────────────────────────

def render_result(result: dict) -> None:
    """Display the structured JSON result in a readable Streamlit layout."""
    score = result.get("quality_score", 0)

    # Quality score colour
    if score >= 8:
        score_color = "🟢"
    elif score >= 5:
        score_color = "🟡"
    else:
        score_color = "🔴"

    st.markdown("### Processed output")

    col1, col2, col3 = st.columns(3)
    col1.metric("Emotion", result.get("emotion", "—").capitalize())
    col2.metric("Theme", result.get("theme", "—").capitalize())
    col3.metric("Quality score", f"{score_color} {score} / 10")

    st.markdown("**Cleaned text**")
    st.info(result.get("cleaned_text", "—"))

    st.markdown("**Objects detected**")
    objects = result.get("objects", [])
    if objects:
        st.write(" · ".join(f"`{o}`" for o in objects))
    else:
        st.write("_None detected_")

    if "_parse_error" in result:
        st.warning(f"⚠️ {result['_parse_error']}")

    with st.expander("Raw JSON"):
        st.code(json.dumps(result, indent=2), language="json")


def render_history(rows: list[tuple]) -> None:
    """Render recent database entries in an expander."""
    if not rows:
        st.write("_No history yet._")
        return

    for row_id, user_text, processed_json_str, timestamp in rows:
        with st.expander(f"#{row_id} — {timestamp[:19].replace('T', ' ')} UTC"):
            st.markdown(f"**Input:** {user_text[:200]}{'…' if len(user_text) > 200 else ''}")
            try:
                data = json.loads(processed_json_str)
                st.json(data)
            except Exception:
                st.code(processed_json_str)

# ─── Main App ─────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="GenAI Image Description Processor",
        page_icon="🖼️",
        layout="wide",
    )

    # Initialise database on every cold start
    init_db()

    st.title("🖼️ GenAI Image Description Processor")
    st.caption("Upload an image, describe it, and let a local LLM (Ollama / llama3) extract structured insights.")

    st.divider()

    # ── Input section ────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        uploaded_file = st.file_uploader(
            "Upload an image (optional — for visual reference only)",
            type=["jpg", "jpeg", "png", "webp", "gif"],
            help="The image is displayed for reference. The LLM processes your text description.",
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_container_width=True)

    with col_right:
        user_text = st.text_area(
            "Describe the image",
            height=160,
            placeholder="e.g. omg this tree is sooo pretty lol its like super old and the leafs r all orange n stuff!!!",
            help="Write anything — the app will clean and analyse your description.",
        )

        submitted = st.button("Process description ▶", type="primary", use_container_width=True)

    st.divider()

    # ── Processing ────────────────────────────────────────────────────────────
    if submitted:
        if not user_text.strip():
            st.error("Please enter a text description before submitting.")
        else:
            with st.spinner("Sending to Ollama (llama3)… this may take 10–30 seconds."):
                try:
                    result = call_ollama(user_text.strip())
                    processed_json_str = json.dumps(result)
                    row_id = save_response(user_text.strip(), processed_json_str)
                    st.success(f"✅ Processed and saved (record #{row_id})")
                    render_result(result)

                except requests.exceptions.ConnectionError:
                    st.error(
                        "**Ollama is not running.**\n\n"
                        "Start it with: `ollama serve`\n\n"
                        "Then make sure the model is pulled: `ollama pull llama3`"
                    )
                except requests.exceptions.Timeout:
                    st.error(
                        "**Request timed out.** Ollama took longer than 60 seconds. "
                        "Try a shorter description or check your system resources."
                    )
                except requests.exceptions.HTTPError as e:
                    st.error(f"**Ollama API error:** {e}\n\nEnsure the model `llama3` is available: `ollama pull llama3`")
                except Exception as e:
                    st.error(f"**Unexpected error:** {e}")

    # ── History section ───────────────────────────────────────────────────────
    st.markdown("### Recent responses (last 5)")
    rows = fetch_recent(limit=5)
    render_history(rows)


if __name__ == "__main__":
    main()
