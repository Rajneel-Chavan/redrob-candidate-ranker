import streamlit as st
import json
import csv
import io
import gzip
import sys
import os

# Import scoring logic from rank.py
sys.path.insert(0, os.path.dirname(__file__))
from rank import score_candidate, is_honeypot

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Redrob Candidate Ranker")
st.markdown("**Intelligent Candidate Discovery & Ranking** — Data & AI Challenge submission by Rajneel Chavan")
st.markdown("---")

st.markdown("""
Upload a `.jsonl` or `.jsonl.gz` file of candidate profiles.
The system will rank all candidates and return the **top 100** as a downloadable CSV.
""")

uploaded_file = st.file_uploader(
    "Upload candidates file (.jsonl or .jsonl.gz)",
    type=["jsonl", "gz"]
)

if uploaded_file is not None:
    with st.spinner("Reading candidates..."):
        raw = uploaded_file.read()
        if uploaded_file.name.endswith(".gz"):
            raw = gzip.decompress(raw)
        lines = raw.decode("utf-8").strip().split("\n")

    st.info(f"Loaded **{len(lines):,}** candidates. Scoring now...")

    progress = st.progress(0)
    results = []
    errors = 0

    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            candidate = json.loads(line)
            score, reasoning = score_candidate(candidate)
            results.append({
                "candidate_id": candidate.get("candidate_id", f"CAND_{i}"),
                "score": score,
                "reasoning": reasoning
            })
        except Exception:
            errors += 1
        if i % 1000 == 0:
            progress.progress(min(i / len(lines), 1.0))

    progress.progress(1.0)

    # Sort and take top 100
    results.sort(key=lambda x: x["score"], reverse=True)
    top100 = results[:100]

    # Rescale scores to [0.10, 0.99]
    if len(top100) > 1:
        min_s = top100[-1]["score"]
        max_s = top100[0]["score"]
        if max_s > min_s:
            for r in top100:
                r["score"] = 0.10 + (r["score"] - min_s) / (max_s - min_s) * 0.89

    st.success(f"✅ Ranked {len(results):,} candidates in seconds. Showing top 100.")
    if errors:
        st.warning(f"{errors} lines skipped due to parse errors.")

    # Build CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["rank", "candidate_id", "score", "reasoning"])
    for rank, r in enumerate(top100, 1):
        writer.writerow([rank, r["candidate_id"], f"{r['score']:.4f}", r["reasoning"]])

    csv_bytes = output.getvalue().encode("utf-8")

    st.download_button(
        label="⬇️ Download ranked CSV",
        data=csv_bytes,
        file_name="rajneel_chavan.csv",
        mime="text/csv"
    )

    # Show table
    st.markdown("### Top 10 Preview")
    import pandas as pd
    df = pd.DataFrame([
        {"Rank": i+1, "Candidate ID": r["candidate_id"], "Score": f"{r['score']:.4f}"}
        for i, r in enumerate(top100[:10])
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    "**GitHub:** [Rajneel-Chavan/redrob-candidate-ranker](https://github.com/Rajneel-Chavan/redrob-candidate-ranker) "
    "· Built for India.Runs Data & AI Challenge"
)
