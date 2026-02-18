import streamlit as st
import pandas as pd

from utils import extract_text, extract_keywords, calculate_tfidf_similarity, keyword_overlap_analysis

st.set_page_config(
    page_title="Resume Screener",
    page_icon="\U0001f4c4",
    layout="wide",
)

st.title("\U0001f4c4 Resume Screening & Ranking Tool")
st.markdown("Upload resumes and a job description to find the best-matching candidates.")

# ── Sidebar: Inputs ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Inputs")

    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the job description here:",
        height=250,
        placeholder="Enter the full job description including required skills, qualifications, and responsibilities...",
    )

    # ── Keyword Editor ──
    st.subheader("Keywords")

    if st.button("Extract Keywords from JD", use_container_width=True):
        if job_description.strip():
            auto_keywords = sorted(extract_keywords(job_description))
            st.session_state["edited_keywords"] = ", ".join(auto_keywords)
        else:
            st.warning("Enter a job description first.")

    edited_keywords_str = st.text_area(
        "Edit keywords (comma-separated):",
        value=st.session_state.get("edited_keywords", ""),
        height=120,
        placeholder="python, machine learning, docker, sql, ...",
        help="Click 'Extract Keywords from JD' to auto-fill, then add or remove as needed.",
    )
    # Sync edits back to session state
    st.session_state["edited_keywords"] = edited_keywords_str

    st.subheader("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    st.subheader("Settings")
    threshold = st.slider(
        "Shortlist threshold (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="Candidates scoring above this threshold will be shortlisted.",
    )

    process_btn = st.button("Analyze Resumes", type="primary", use_container_width=True)

# ── Main Area: Results ────────────────────────────────────────────────────────

if process_btn:
    if not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()
    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()

    # Step 1: Extract text
    with st.spinner("Extracting text from resumes..."):
        results = []
        for file in uploaded_files:
            name, text = extract_text(file)
            results.append({"name": name, "text": text})

    valid_results = [r for r in results if r["text"].strip()]
    failed_files = [r["name"] for r in results if not r["text"].strip()]

    if failed_files:
        st.warning(f"Could not extract text from: {', '.join(failed_files)}")

    if not valid_results:
        st.error("No text could be extracted from any uploaded file.")
        st.stop()

    # Parse user-edited keywords (if any)
    custom_kw = None
    kw_str = st.session_state.get("edited_keywords", "").strip()
    if kw_str:
        custom_kw = {k.strip().lower() for k in kw_str.split(",") if k.strip()}

    # Step 2: Score resumes
    with st.spinner("Analyzing resumes..."):
        resume_texts = [r["text"] for r in valid_results]
        resume_names = [r["name"] for r in valid_results]

        tfidf_scores = calculate_tfidf_similarity(job_description, resume_texts)

        keyword_results = []
        for text in resume_texts:
            keyword_results.append(
                keyword_overlap_analysis(job_description, text, custom_jd_keywords=custom_kw)
            )

    # Step 3: Build results table
    df = pd.DataFrame({
        "Candidate": resume_names,
        "Match Score (%)": [round(s * 100, 1) for s in tfidf_scores],
        "Keyword Overlap (%)": [round(kr["overlap_score"] * 100, 1) for kr in keyword_results],
        "Matched Keywords": [", ".join(sorted(kr["matched_keywords"])) for kr in keyword_results],
        "Missing Keywords": [", ".join(sorted(kr["missing_keywords"])) for kr in keyword_results],
    })
    df = df.sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"
    df["Status"] = df["Match Score (%)"].apply(
        lambda x: "\u2705 Shortlisted" if x >= threshold else "\u274c Rejected"
    )

    # ── Summary Metrics ──
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Resumes", len(valid_results))
    with col2:
        shortlisted = len(df[df["Match Score (%)"] >= threshold])
        st.metric("Shortlisted", shortlisted)
    with col3:
        st.metric("Average Score", f"{df['Match Score (%)'].mean():.1f}%")

    # ── Ranking Table ──
    st.subheader("Ranking Results")
    st.dataframe(
        df[["Candidate", "Match Score (%)", "Keyword Overlap (%)", "Status"]],
        use_container_width=True,
        column_config={
            "Match Score (%)": st.column_config.ProgressColumn(
                "Match Score (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "Keyword Overlap (%)": st.column_config.ProgressColumn(
                "Keyword Overlap (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
        },
    )

    # ── Per-Candidate Details ──
    st.subheader("Candidate Details")
    for _, row in df.iterrows():
        candidate = row["Candidate"]
        score = row["Match Score (%)"]
        status_icon = "\u2705" if score >= threshold else "\u274c"

        with st.expander(f"{status_icon} {candidate} — {score}%"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Matched Keywords:**")
                if row["Matched Keywords"]:
                    st.success(row["Matched Keywords"])
                else:
                    st.info("No keyword matches found.")
            with col_b:
                st.markdown("**Missing Keywords:**")
                if row["Missing Keywords"]:
                    st.error(row["Missing Keywords"])
                else:
                    st.success("All key terms present!")

            st.markdown("**Resume Text Preview:**")
            original_idx = resume_names.index(candidate)
            preview = resume_texts[original_idx][:1000]
            st.text_area(
                "", value=preview, height=150, disabled=True,
                key=f"preview_{candidate}",
            )

    # ── CSV Export ──
    st.subheader("Export Results")
    csv = df.to_csv(index=True)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="resume_screening_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
