import streamlit as st
import json
import time
from datetime import datetime, timezone
import math
import re
from sklearn.feature_extraction.text import CountVectorizer
from difflib import get_close_matches


from datetime import datetime, timezone

def normalize_docs(docs: list[dict]) -> list[dict]:
    out = []
    for d in docs:
        # convert unix timestamp -> ISO string (UTC)
        ts = d.get("timestamp", None)
        if isinstance(ts, (int, float)):
            ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        else:
            ts_iso = str(ts) if ts is not None else ""

        out.append({
            "id": d.get("id"),
            "title": d.get("title", "(no title)"),
            # map content -> text (this is the big one)
            "text": d.get("text", ""),
            "timestamp": ts_iso,
            # keep trust/pogo if your ranking uses them
            "trust": float(d.get("trust", 0.5)),
            "pogo": int(d.get("pogo", 0)),
            # optional fields if your older pipeline expects them
            "source": d.get("source", "unknown"),
            "source_type": d.get("source_type", "unknown"),
            "location": d.get("location", "")
        })
    return out

EMERGENCY_TOPICS = {"earthquake", "cyclone", "tsunami","flood"}

def build_emergency_phrases(docs, min_freq=2):
    texts = []

    for d in docs:
        combined = f"{d['title']} {d['text']}".lower()
        if any(t in combined for t in EMERGENCY_TOPICS):
            texts.append(combined)

    if not texts:
        return set()

    vectorizer = CountVectorizer(
        ngram_range=(1, 3),
        stop_words="english",
        min_df=min_freq
    )
    vectorizer.fit(texts)
    return set(vectorizer.get_feature_names_out())

st.title("Emergency-Aware Search Engine")



if "click_time" not in st.session_state:
    st.session_state.click_time = {}

if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id = None

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "results"  # or "doc"


def load():
     with open("documents_2000_upgraded.json", "r", encoding="utf-8") as f:
        return json.load(f)

DOC_PATH = "documents_2000_upgraded.json"

def save_docs(docs):
    with open(DOC_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)


raw_docs = load()
docs = normalize_docs(raw_docs)


docs = normalize_docs(docs)
def build_vocabulary(docs):
    vocab = set()
    for d in docs:
        text = f"{d['title']} {d['text']}".lower()
        words = re.findall(r"[a-z]+", text)
        vocab.update(words)
    return vocab

VOCAB = build_vocabulary(docs)


EMERGENCY_PHRASES = build_emergency_phrases(docs)


def score_doc(d, emergency=False):
    base = 1.0

    # trust
    base *= d.get("trust", 0.5)

    # pogo penalty
    pogo = d.get("pogo", 0)
    penalty = min(0.5, 0.05 * pogo)
    base *= (1 - penalty)

    # emergency boost
    if emergency:
        base *= 1.3

    return base

def autocorrect_query(query, vocab, cutoff=0.8):
    corrected_words = []
    for word in query.lower().split():
        if word in vocab:
            corrected_words.append(word)
        else:
            matches = get_close_matches(word, vocab, n=1, cutoff=cutoff)
            corrected_words.append(matches[0] if matches else word)
    return " ".join(corrected_words)



query = st.text_input("Search", placeholder="Enter your query..")

results = []

if query:
    q_raw = query.lower().strip()
    q = autocorrect_query(q_raw, VOCAB)
    if q != q_raw:
        st.info(f"Did you mean: **{q}** ?")


    emergency = any(w in q for w in EMERGENCY_PHRASES)

    if emergency:
        st.error("🚨 Emergency Mode ON")
    else:
        st.success("✅ Normal Mode")

    for d in docs:
        text = (d["title"] + " " + d["text"]).lower()
        if q in text:
            d["_score"] = score_doc(d, emergency)
            results.append(d)

    results = sorted(results, key=lambda x: x["_score"], reverse=True)

RESULTS_PER_PAGE = 10
page = st.number_input(
    "Page",
    min_value=1,
    max_value=max(1, math.ceil(len(results) / RESULTS_PER_PAGE)),
    step=1
)

start = (page - 1) * RESULTS_PER_PAGE
end = start + RESULTS_PER_PAGE

if st.session_state.view_mode == "results" and query:
    st.write(f"Found {len(results)} result(s)")

    for i, d in enumerate(results[start:end], start=start + 1):
        st.markdown(f"### {i}. {d['title']}")

        preview = d.get("text", "")[:250]
        st.write(preview)

        if st.button("Open", key=f"open_{d['id']}"):
            st.session_state.click_time[d["id"]] = time.time()
            st.session_state.current_doc_id = d["id"]
            st.session_state.view_mode = "doc"
            st.rerun()

        st.caption(f"Time: {d['timestamp']}")
        st.divider()

if st.session_state.view_mode == "doc":
    doc_id = st.session_state.current_doc_id
    doc = next(d for d in docs if d["id"] == doc_id)

    st.subheader(doc["title"])
    st.write(doc["text"])

    if st.button("← Back to results"):
        end_time = time.time()
        start_time = st.session_state.click_time.get(doc_id, end_time)
        dwell_time = end_time - start_time

        POGO_THRESHOLD = 8

        if dwell_time < POGO_THRESHOLD:
            doc["pogo"] = doc.get("pogo", 0) + 1
            raw_doc = next(rd for rd in raw_docs if rd["id"] == doc_id)
            raw_doc["pogo"] = raw_doc.get("pogo", 0) + 1
            save_docs(docs)
            st.warning("Quick return detected (pogo-sticking)")
        else:
            st.success("User engaged — no pogo")

        st.session_state.current_doc_id = None
        st.session_state.view_mode = "results"
        st.rerun()

