# app.py — GenAI Human Creativity Data Engine
# Run:  streamlit run app.py
# Deps: pip install streamlit requests pandas Pillow

import json
import re
import sqlite3
import datetime
import io
import requests
import streamlit as st
import pandas as pd
from PIL import Image

# ─── Config ───────────────────────────────────────────────────────────────────

DB_PATH         = "creativity.db"
GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

FALLBACK_MODELS = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# Prompts shown to users when they view an image
IMAGE_PROMPTS = [
    "What do you feel when you look at this image?",
    "Write a short poem inspired by this image.",
    "Describe what you see in your own words.",
    "What emotions does this image evoke in you?",
    "Tell a short story inspired by this artwork.",
    "What does this image remind you of?",
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def safe_str(value) -> str:
    text = str(value) if not isinstance(value, str) else value
    return text.encode("ascii", errors="ignore").decode("ascii")


def safe_dumps(obj, indent=None) -> str:
    return json.dumps(obj, ensure_ascii=True, indent=indent, default=str)


def extract_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ─── Live model fetching ──────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_models(token: str) -> list:
    try:
        resp = requests.get(
            GROQ_MODELS_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if not resp.ok:
            return FALLBACK_MODELS
        exclude = ["whisper", "tts", "vision", "guard"]
        models  = [
            m["id"] for m in resp.json().get("data", [])
            if not any(kw in m["id"].lower() for kw in exclude)
        ]
        return sorted(models) if models else FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


# ─── Database ─────────────────────────────────────────────────────────────────

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # Images table — stores uploaded painting/artwork
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT,
                prompt      TEXT,
                uploaded_at TEXT
            )
        """)
        # Responses table — stores human creativity submissions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id        INTEGER,
                response_type   TEXT,
                raw_text        TEXT,
                cleaned_text    TEXT,
                objects         TEXT,
                emotion         TEXT,
                theme           TEXT,
                quality_score   INTEGER,
                creativity_score INTEGER,
                sentiment       TEXT,
                keywords        TEXT,
                model_used      TEXT,
                created_at      TEXT,
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)


def save_image(name: str, prompt: str) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO images (name, prompt, uploaded_at) VALUES (?, ?, ?)",
            (name, prompt, datetime.datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def get_images() -> list:
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            "SELECT id, name, prompt, uploaded_at FROM images ORDER BY id DESC"
        ).fetchall()


def save_response(image_id: int, response_type: str, raw_text: str,
                  result: dict, model: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO responses (
                image_id, response_type, raw_text,
                cleaned_text, objects, emotion, theme,
                quality_score, creativity_score, sentiment,
                keywords, model_used, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            image_id,
            safe_str(response_type),
            safe_str(raw_text[:1000]),
            safe_str(result.get("cleaned_text", "")),
            safe_dumps(result.get("objects", [])),
            safe_str(result.get("emotion", "")),
            safe_str(result.get("theme", "")),
            int(result.get("quality_score", 0)),
            int(result.get("creativity_score", 0)),
            safe_str(result.get("sentiment", "")),
            safe_dumps(result.get("keywords", [])),
            safe_str(model),
            datetime.datetime.utcnow().isoformat(),
        ))


def get_responses(image_id: int = None) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        if image_id:
            return conn.execute(
                "SELECT * FROM responses WHERE image_id = ? ORDER BY id DESC",
                (image_id,)
            ).fetchall()
        return conn.execute(
            "SELECT * FROM responses ORDER BY id DESC"
        ).fetchall()


def get_stats() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        total    = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        images   = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        avg_q    = conn.execute("SELECT AVG(quality_score) FROM responses").fetchone()[0]
        avg_c    = conn.execute("SELECT AVG(creativity_score) FROM responses").fetchone()[0]
        return {
            "total_responses": total,
            "total_images":    images,
            "avg_quality":     round(avg_q or 0, 1),
            "avg_creativity":  round(avg_c or 0, 1),
        }


def delete_response(response_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM responses WHERE id = ?", (response_id,))


# ─── Groq API ─────────────────────────────────────────────────────────────────

def call_groq(messages: list, token: str, model: str,
              max_tokens: int = 800, temperature: float = 0.1) -> str:
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages,
              "max_tokens": max_tokens, "temperature": temperature},
        timeout=30,
    )
    if resp.status_code == 401:
        raise PermissionError("Invalid Groq API key. Get one free at console.groq.com")
    if resp.status_code == 429:
        raise TimeoutError("Rate limit hit. Wait a moment and retry.")
    if not resp.ok:
        raise Exception(f"Groq error {resp.status_code}: {resp.text[:300]}")
    return safe_str(resp.json()["choices"][0]["message"]["content"])


def groq_json(prompt: str, token: str, model: str) -> str:
    return call_groq(
        messages=[
            {"role": "system", "content":
                "You are a data processing engine for a creativity analysis system. "
                "Return valid JSON only. No markdown. No explanation."},
            {"role": "user", "content": prompt},
        ],
        token=token, model=model, max_tokens=800, temperature=0.1,
    )


# ─── GenAI Processing ─────────────────────────────────────────────────────────

PROCESS_PROMPT = """\
You are analysing a human creativity response to an artwork or image.

Response type: {response_type}
User's raw response: "{raw_text}"
Image context: "{image_context}"

Return ONLY this JSON object:
{{
  "cleaned_text": "<corrected, coherent version of the response>",
  "objects": ["<visual element 1>", "<visual element 2>", "<visual element 3>"],
  "emotion": "<single dominant emotion word>",
  "theme": "<single central theme word>",
  "quality_score": <integer 1-10>,
  "creativity_score": <integer 1-10>,
  "sentiment": "<positive | negative | neutral>",
  "keywords": ["<word1>", "<word2>", "<word3>"],
  "language": "<detected language>"
}}

Scoring rules:
- quality_score: clarity, grammar, coherence (1=gibberish, 10=excellent)
- creativity_score: originality, imagination, expressiveness (1=generic, 10=highly creative)
- emotion: one word e.g. joy, awe, melancholy, nostalgia, wonder, fear
- theme: one word e.g. nature, love, solitude, hope, decay, light
- objects: visual elements seen or referenced in the response
- Return raw JSON only"""


def process_response(raw_text: str, response_type: str,
                     image_context: str, token: str, model: str) -> dict:
    raw  = groq_json(
        PROCESS_PROMPT.format(
            response_type=response_type,
            raw_text=safe_str(raw_text).replace('"', "'"),
            image_context=safe_str(image_context),
        ),
        token, model,
    )
    data = extract_json(raw)

    result = {
        "cleaned_text":    safe_str(data.get("cleaned_text",    raw_text)),
        "objects":         [safe_str(o) for o in data.get("objects",   [])],
        "emotion":         safe_str(data.get("emotion",         "unknown")),
        "theme":           safe_str(data.get("theme",           "unknown")),
        "quality_score":   0,
        "creativity_score": 0,
        "sentiment":       safe_str(data.get("sentiment",       "neutral")),
        "keywords":        [safe_str(k) for k in data.get("keywords", [])],
        "language":        safe_str(data.get("language",        "English")),
    }

    for score_key in ("quality_score", "creativity_score"):
        try:
            result[score_key] = max(0, min(10, int(data.get(score_key, 0))))
        except (ValueError, TypeError):
            result[score_key] = 0

    return result


# ─── Score bar helper ─────────────────────────────────────────────────────────

def score_color(score: int) -> str:
    if score >= 8: return "🟢"
    if score >= 5: return "🟡"
    return "🔴"


def sentiment_label(s: str) -> str:
    s = s.lower()
    if s == "positive": return "🟢 Positive"
    if s == "negative": return "🔴 Negative"
    return "🟡 Neutral"


# ─── Tab: Post Image ──────────────────────────────────────────────────────────

def tab_post_image():
    st.subheader("Post an Artwork")
    st.markdown(
        "Upload a painting or image to share with your audience. "
        "Choose a prompt to guide their creative response."
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded = st.file_uploader(
            "Upload artwork (PNG, JPG, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            key="post_img",
        )
        if uploaded:
            st.image(Image.open(uploaded), caption=uploaded.name,
                     use_container_width=True)

    with col2:
        image_name = st.text_input(
            "Image title",
            placeholder="e.g. Starry Night, Forest at Dawn...",
        )
        selected_prompt = st.selectbox(
            "Audience prompt",
            IMAGE_PROMPTS,
            help="This is the question your audience will see alongside the image.",
        )
        custom_prompt = st.text_input(
            "Or write a custom prompt (optional)",
            placeholder="e.g. What story does this painting tell you?",
        )
        final_prompt = custom_prompt.strip() if custom_prompt.strip() else selected_prompt

        st.info(f"**Audience will see:** *{final_prompt}*")

    if st.button("Post Image", type="primary", use_container_width=True):
      if not uploaded:
        st.warning("Please upload an image first.")
        return
    if not image_name.strip():
        st.warning("Please give the image a title.")
        return

    # ✅ SAFE way to read image
    img_bytes = uploaded.getvalue()

    if not img_bytes:
        st.error("Image data is empty or corrupted.")
        return

    image_id = save_image(image_name.strip(), final_prompt)

    st.session_state["posted_images"] = st.session_state.get("posted_images", {})
    st.session_state["posted_images"][image_id] = {
        "name":   image_name.strip(),
        "prompt": final_prompt,
        "bytes":  img_bytes,
    }

    st.success(
        f"Image **{image_name}** posted! "
        f"Go to the **Respond** tab to submit creative responses."
    )


# ─── Tab: Respond ─────────────────────────────────────────────────────────────

def tab_respond(token: str, model: str):
    st.subheader("Respond to an Artwork")

    images = get_images()
    if not images:
        st.info("No images posted yet. Go to **Post Image** to add one.")
        return

    # Image selector
    image_options = {f"#{row[0]} — {row[1]}": row for row in images}
    selected_label = st.selectbox("Select an image to respond to", list(image_options.keys()))
    selected_row   = image_options[selected_label]
    image_id       = selected_row[0]
    image_name     = selected_row[1]
    image_prompt   = selected_row[2]

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        # Show image if it was uploaded in this session
        posted = st.session_state.get("posted_images", {})
        if image_id in posted:
            st.image(
                Image.open(io.BytesIO(posted[image_id]["bytes"])),
                caption=image_name,
                use_container_width=True,
            )
        else:
            st.info(
                f"Image **{image_name}** was posted in a previous session. "
                "Re-upload it in **Post Image** to preview it here."
            )

    with col2:
        st.markdown(f"### {image_name}")
        st.markdown(f"> *{image_prompt}*")
        st.divider()

        response_type = st.selectbox(
            "Response type",
            ["Poem", "Emotion / Feeling", "Description", "Story", "Free thought"],
            help="What kind of creative response are you submitting?",
        )

        user_response = st.text_area(
            "Your response",
            height=160,
            placeholder=(
                "Write your poem, description, or feeling here...\n"
                "Express yourself freely — any language is welcome!"
            ),
        )

        if st.button("Submit & Process", type="primary", use_container_width=True):
            if not token.strip():
                st.error("Add your Groq API key in the sidebar.")
                return
            if not user_response.strip():
                st.warning("Please write a response before submitting.")
                return

            with st.spinner("Processing your response with AI..."):
                try:
                    result = process_response(
                        raw_text=user_response.strip(),
                        response_type=response_type,
                        image_context=f"{image_name}: {image_prompt}",
                        token=token.strip(),
                        model=model,
                    )
                    save_response(image_id, response_type, user_response.strip(),
                                  result, model)
                    st.success("Response submitted and processed!")

                    st.divider()
                    st.markdown("### AI Analysis of your response")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Emotion",    result["emotion"].capitalize())
                    c2.metric("Theme",      result["theme"].capitalize())
                    c3.metric("Quality",
                              f"{score_color(result['quality_score'])} {result['quality_score']}/10")
                    c4.metric("Creativity",
                              f"{score_color(result['creativity_score'])} {result['creativity_score']}/10")

                    m1, m2 = st.columns(2)
                    m1.markdown(f"**Sentiment:** {sentiment_label(result['sentiment'])}")
                    m2.markdown(f"**Language detected:** `{result['language']}`")

                    if result["keywords"]:
                        st.markdown(
                            "**Keywords:** "
                            + "  ·  ".join(f"`{k}`" for k in result["keywords"])
                        )
                    if result["objects"]:
                        st.markdown(
                            "**Visual elements:** "
                            + "  ·  ".join(f"`{o}`" for o in result["objects"])
                        )

                    st.markdown("**Cleaned & structured version:**")
                    st.info(result["cleaned_text"])

                    with st.expander("Structured JSON output"):
                        st.code(safe_dumps(result, indent=2), language="json")

                except PermissionError as e:
                    st.error(str(e))
                except TimeoutError as e:
                    st.error(str(e))
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach Groq API. Check your connection.")
                except Exception as e:
                    st.error(f"Error: {safe_str(str(e))}")


# ─── Tab: Dataset ─────────────────────────────────────────────────────────────

def tab_dataset():
    st.subheader("Collected Dataset")
    st.markdown(
        "All processed human responses — ready for AI training and analysis."
    )

    responses = get_responses()
    if not responses:
        st.info("No responses collected yet.")
        return

    # Build display DataFrame
    col_names = [
        "id", "image_id", "response_type", "raw_text",
        "cleaned_text", "objects", "emotion", "theme",
        "quality_score", "creativity_score", "sentiment",
        "keywords", "model_used", "created_at"
    ]
    df = pd.DataFrame(responses, columns=col_names)

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        type_filter = st.multiselect(
            "Filter by type",
            df["response_type"].unique().tolist(),
        )
    with f2:
        emotion_filter = st.multiselect(
            "Filter by emotion",
            df["emotion"].unique().tolist(),
        )
    with f3:
        min_quality = st.slider("Min quality score", 0, 10, 0)

    filtered = df.copy()
    if type_filter:
        filtered = filtered[filtered["response_type"].isin(type_filter)]
    if emotion_filter:
        filtered = filtered[filtered["emotion"].isin(emotion_filter)]
    filtered = filtered[filtered["quality_score"] >= min_quality]

    st.markdown(f"**{len(filtered)} records** (of {len(df)} total)")

    display_cols = [
        "id", "image_id", "response_type", "emotion", "theme",
        "quality_score", "creativity_score", "sentiment",
        "cleaned_text", "created_at"
    ]
    st.dataframe(filtered[display_cols], use_container_width=True)

    # Expand individual record
    st.divider()
    st.markdown("**Inspect a record**")
    record_id = st.number_input("Enter record ID", min_value=1, step=1)
    row = filtered[filtered["id"] == record_id]
    if not row.empty:
        r = row.iloc[0]
        st.markdown(f"**Raw text:** {r['raw_text']}")
        st.markdown(f"**Cleaned:** {r['cleaned_text']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quality",    r["quality_score"])
        c2.metric("Creativity", r["creativity_score"])
        c3.metric("Emotion",    str(r["emotion"]).capitalize())
        c4.metric("Theme",      str(r["theme"]).capitalize())

        if st.button("Delete this record", type="secondary"):
            delete_response(int(r["id"]))
            st.success("Record deleted.")
            st.rerun()

    st.divider()

    # Export
    e1, e2 = st.columns(2)
    with e1:
        csv = filtered.to_csv(index=False).encode("utf-8-sig", errors="replace")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="creativity_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e2:
        # Export as clean training JSONL
        training_records = []
        for _, r in filtered.iterrows():
            training_records.append({
                "image_id":        int(r["image_id"]),
                "response_type":   r["response_type"],
                "cleaned_text":    r["cleaned_text"],
                "emotion":         r["emotion"],
                "theme":           r["theme"],
                "quality_score":   int(r["quality_score"]),
                "creativity_score": int(r["creativity_score"]),
                "sentiment":       r["sentiment"],
                "objects":         r["objects"],
                "keywords":        r["keywords"],
            })
        jsonl = "\n".join(safe_dumps(rec) for rec in training_records)
        st.download_button(
            "Download JSONL (for training)",
            data=jsonl.encode("utf-8", errors="replace"),
            file_name="creativity_dataset.jsonl",
            mime="application/jsonl",
            use_container_width=True,
        )


# ─── Tab: Analytics ───────────────────────────────────────────────────────────

def tab_analytics():
    st.subheader("Analytics Dashboard")

    stats = get_stats()

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total responses",  stats["total_responses"])
    c2.metric("Images posted",    stats["total_images"])
    c3.metric("Avg quality",      f"{stats['avg_quality']} / 10")
    c4.metric("Avg creativity",   f"{stats['avg_creativity']} / 10")

    responses = get_responses()
    if not responses:
        st.info("No data yet.")
        return

    col_names = [
        "id", "image_id", "response_type", "raw_text",
        "cleaned_text", "objects", "emotion", "theme",
        "quality_score", "creativity_score", "sentiment",
        "keywords", "model_used", "created_at"
    ]
    df = pd.DataFrame(responses, columns=col_names)

    st.divider()

    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**Responses by type**")
        rc = df["response_type"].value_counts()
        if not rc.empty:
            st.bar_chart(rc)

        st.markdown("**Emotion distribution**")
        ec = df["emotion"].value_counts()
        if not ec.empty:
            st.bar_chart(ec)

        st.markdown("**Sentiment breakdown**")
        sc = df["sentiment"].value_counts()
        if not sc.empty:
            st.bar_chart(sc)

    with ch2:
        st.markdown("**Theme distribution**")
        tc = df["theme"].value_counts()
        if not tc.empty:
            st.bar_chart(tc)

        st.markdown("**Quality scores**")
        st.bar_chart(df["quality_score"].value_counts().sort_index())

        st.markdown("**Creativity scores**")
        st.bar_chart(df["creativity_score"].value_counts().sort_index())

    st.divider()

    # Quality vs creativity scatter (simulated with line chart)
    st.markdown("**Quality vs Creativity per response**")
    score_df = df[["quality_score", "creativity_score"]].reset_index(drop=True)
    score_df.columns = ["Quality", "Creativity"]
    st.line_chart(score_df)

    # Top responses
    st.divider()
    st.markdown("**Top 5 most creative responses**")
    top = df.nlargest(5, "creativity_score")[
        ["response_type", "emotion", "theme",
         "creativity_score", "quality_score", "cleaned_text"]
    ]
    st.dataframe(top, use_container_width=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="GenAI Human Creativity Engine",
        page_icon="🎨",
        layout="wide",
    )
    init_db()

    st.title("🎨 GenAI Human Creativity Data Engine")
    st.caption(
        "Collect, process, and structure human creativity data "
        "from image-based interactions — powered by Groq AI."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Setup")
        st.markdown(
            "**Free Groq API key:**\n"
            "1. [console.groq.com](https://console.groq.com)\n"
            "2. Sign up (no credit card)\n"
            "3. API Keys → Create Key\n"
            "4. Paste below"
        )
        token = st.text_input(
            "Groq API key",
            type="password",
            placeholder="gsk_xxxxxxxxxxxxxxxxxxxx",
        )

        if token:
            st.success("Key set")
        else:
            st.warning("Key required for AI processing")

        st.divider()

        # Live model list
        if token:
            with st.spinner("Loading models..."):
                models = fetch_models(token)
        else:
            models = FALLBACK_MODELS

        default_idx = models.index("llama3-8b-8192") if "llama3-8b-8192" in models else 0
        model = st.selectbox(
            "AI model",
            options=models,
            index=default_idx,
            help="Fetched live from Groq API",
        )
        st.caption(f"{len(models)} models available")

        st.divider()

        # Dataset stats in sidebar
        stats = get_stats()
        st.markdown("**Dataset stats**")
        st.markdown(f"- Responses: **{stats['total_responses']}**")
        st.markdown(f"- Images: **{stats['total_images']}**")
        st.markdown(f"- Avg quality: **{stats['avg_quality']}/10**")
        st.markdown(f"- Avg creativity: **{stats['avg_creativity']}/10**")

        st.divider()

        # Pipeline overview
        st.markdown("**Data pipeline**")
        st.markdown(
            "```\n"
            "Image Upload\n"
            "    ↓\n"
            "User Response\n"
            "    ↓\n"
            "GenAI Processing\n"
            "    ↓\n"
            "Structured JSON\n"
            "    ↓\n"
            "SQLite Storage\n"
            "    ↓\n"
            "Export for Training\n"
            "```"
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Post Image",
        "Respond",
        "Dataset",
        "Analytics",
    ])

    with tab1:
        tab_post_image()

    with tab2:
        tab_respond(token, model)

    with tab3:
        tab_dataset()

    with tab4:
        tab_analytics()


if __name__ == "__main__":
    main()