import streamlit as st
import json
import time
from datetime import datetime, timezone
import math
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer


mode = st.sidebar.selectbox(
    "Select Mode",
    ["Search Engine", "Admin Panel"]
)

def load():
     with open("documents_2000_upgraded.json", "r", encoding="utf-8") as f:
          return json.load(f)

def save_docs(docs):
    with open("documents_2000.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)


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

EMERGENCY_TOPICS = {"earthquake", "cyclone", "tsunami"}

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


raw_docs = load()
docs = normalize_docs(raw_docs)

EMERGENCY_PHRASES = build_emergency_phrases(docs)


if mode == "Search Engine":
     st.title("Emergency-Aware Search Engine")

     if "click_time" not in st.session_state:
         st.session_state.click_time = {}

     if "current_doc_id" not in st.session_state:
         st.session_state.current_doc_id = None

     if "view_mode" not in st.session_state:
         st.session_state.view_mode = "results"  # or "doc"

     query = st.text_input("Search", placeholder="Enter your query..")

     results = []

     if query:
         q = query.lower().strip()
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
     if st.session_state.view_mode == "results" and query:
         st.write(f"Found {len(results)} result(s)")

         for i, d in enumerate(results[:10], start=1):
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

     
elif mode == "Admin Panel":
     st.title("🛠 Admin Panel (Emergency Updates)")
     title = st.text_input("Title")
     content = st.text_area("Content / Update")
     source_type = st.selectbox(
          "Source Type",
          ["Official", "Reliable", "Unverified"]
     )

     if st.button("Add Update"):
          if title and content:
               new_doc = {
                    "title": title,
                    "text": content,
                    "snippet": content[:150],
                    "source_type": source_type,
                    "timestamp": str(pd.Timestamp.now())
               }
               
               docs.append(new_doc)
               save_docs(docs)
               
               st.success("✅ Update added successfully")
               st.rerun()
          else:
               st.warning("Please fill all fields")

     # ---------- Document database view ----------
     df_docs = pd.DataFrame(docs)

     st.subheader("📂 Document Database (Read-only)")

     # ---------- SIDEBAR FILTERS ----------
     st.sidebar.markdown("### 🔎 Filters")

     # Column selection
     all_columns = df_docs.columns.tolist()
     default_cols = [c for c in ["title", "source_type", "timestamp"] if c in all_columns]

     selected_cols = st.sidebar.multiselect(
         "Columns to display",
         options=all_columns,
         default=default_cols
     )

     if not selected_cols:
         st.warning("Please select at least one column.")
         st.stop()

     # Title search
     search_term = st.sidebar.text_input("Search title")

     # Source type filter
     if "source_type" in df_docs.columns:
         source_filter = st.sidebar.multiselect(
             "Source type",
             options=df_docs["source_type"].dropna().unique()
         )
     else:
         source_filter = []

     # Emergency-only filter
     emergency_only = st.sidebar.checkbox("Emergency documents only")

     # Timestamp handling
     if "timestamp" in df_docs.columns:
         df_docs["timestamp"] = pd.to_datetime(df_docs["timestamp"], errors="coerce")

         min_date = df_docs["timestamp"].min().date()
         max_date = df_docs["timestamp"].max().date()

         date_range = st.sidebar.date_input(
             "Date range",
             value=(min_date, max_date)
         )
     else:
         date_range = None

     # Rows per page
     page_size = st.sidebar.selectbox(
         "Rows per page",
         [10, 25, 50, 100],
         index=1
     )

     # ---------- APPLY FILTERS ----------
     df_view = df_docs[selected_cols].copy()

     if search_term and "title" in df_docs.columns:
         df_view = df_view[
             df_docs["title"]
             .str.lower()
             .str.contains(search_term.lower(), na=False)
         ]

     if source_filter:
         df_view = df_view[df_docs["source_type"].isin(source_filter)]

     if emergency_only:
         emergency_pattern = "|".join(EMERGENCY_WORDS)
         df_view = df_view[
             df_docs["text"]
             .str.lower()
             .str.contains(emergency_pattern, na=False)
         ]

     if date_range and len(date_range) == 2:
         start_date, end_date = date_range
         df_view = df_view[
             (df_view["timestamp"].dt.date >= start_date) &
             (df_view["timestamp"].dt.date <= end_date)
         ]

     # Sorting
     if "timestamp" in df_view.columns:
         df_view = df_view.sort_values("timestamp", ascending=False)

     # ---------- PAGINATION ----------
     total_pages = max(1, (len(df_view) - 1) // page_size + 1)

     page = st.number_input(
         "Page",
         min_value=1,
         max_value=total_pages,
         step=1
     )

     start = (page - 1) * page_size
     end = start + page_size

     st.dataframe(df_view.iloc[start:end], use_container_width=True)

     st.caption(
         f"Showing rows {start + 1} to {min(end, len(df_view))} of {len(df_view)}"
     )
