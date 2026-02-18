import streamlit as st
import pandas as pd

from utils import (
    extract_text,
    extract_keywords,
    extract_contact_info,
    extract_experience_years,
    calculate_tfidf_similarity,
    keyword_overlap_analysis,
)

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
        height=200,
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
        height=100,
        placeholder="python, machine learning, docker, sql, ...",
        help="Click 'Extract Keywords from JD' to auto-fill, then add or remove as needed.",
    )
    st.session_state["edited_keywords"] = edited_keywords_str

    # ── Must-Have Keywords ──
    st.subheader("Must-Have Skills")
    must_have_str = st.text_input(
        "Required skills (comma-separated):",
        placeholder="python, sql, aws",
        help="Candidates missing ANY of these will be flagged as 'Missing Required'.",
    )

    # ── Upload ──
    st.subheader("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    # ── Settings ──
    st.subheader("Settings")
    threshold = st.slider(
        "Shortlist threshold (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="Candidates scoring above this threshold will be shortlisted.",
    )
    min_experience = st.slider(
        "Minimum experience (years)",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help="Filter out candidates with less than this many years of experience.",
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

    # Parse must-have keywords
    must_have_keywords = set()
    if must_have_str.strip():
        must_have_keywords = {k.strip().lower() for k in must_have_str.split(",") if k.strip()}

    # Step 2: Score resumes + extract info
    with st.spinner("Analyzing resumes..."):
        resume_texts = [r["text"] for r in valid_results]
        resume_names = [r["name"] for r in valid_results]

        tfidf_scores = calculate_tfidf_similarity(job_description, resume_texts)

        keyword_results = []
        contact_infos = []
        experience_years = []
        must_have_results = []

        for text in resume_texts:
            keyword_results.append(
                keyword_overlap_analysis(job_description, text, custom_jd_keywords=custom_kw)
            )
            contact_infos.append(extract_contact_info(text))
            experience_years.append(extract_experience_years(text))

            # Check must-have keywords
            if must_have_keywords:
                text_lower = text.lower()
                missing_must = {kw for kw in must_have_keywords if kw not in text_lower}
                must_have_results.append(missing_must)
            else:
                must_have_results.append(set())

    # Step 3: Build results table
    df = pd.DataFrame({
        "Candidate": resume_names,
        "Email": [ci["email"] for ci in contact_infos],
        "Phone": [ci["phone"] for ci in contact_infos],
        "Experience (yrs)": experience_years,
        "Match Score (%)": [round(s * 100, 1) for s in tfidf_scores],
        "Keyword Overlap (%)": [round(kr["overlap_score"] * 100, 1) for kr in keyword_results],
        "Matched Keywords": [", ".join(sorted(kr["matched_keywords"])) for kr in keyword_results],
        "Missing Keywords": [", ".join(sorted(kr["missing_keywords"])) for kr in keyword_results],
        "Missing Must-Haves": [", ".join(sorted(m)) if m else "" for m in must_have_results],
    })

    # Determine status
    def get_status(row):
        if row["Missing Must-Haves"]:
            return "\u26a0\ufe0f Missing Required"
        if min_experience > 0 and row["Experience (yrs)"] < min_experience:
            return "\u274c Low Experience"
        if row["Match Score (%)"] >= threshold:
            return "\u2705 Shortlisted"
        return "\u274c Rejected"

    df["Status"] = df.apply(get_status, axis=1)
    df = df.sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"

    # ── Summary Metrics ──
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Resumes", len(valid_results))
    with col2:
        shortlisted = len(df[df["Status"].str.contains("Shortlisted")])
        st.metric("Shortlisted", shortlisted)
    with col3:
        st.metric("Average Score", f"{df['Match Score (%)'].mean():.1f}%")
    with col4:
        flagged = len(df[df["Status"].str.contains("Missing Required")])
        st.metric("Missing Required", flagged)

    # ── Score Distribution Chart ──
    st.subheader("Score Distribution")
    chart_df = df[["Candidate", "Match Score (%)"]].copy()
    chart_df = chart_df.sort_values("Match Score (%)", ascending=True)
    colors = [
        "#2ecc71" if s >= threshold else "#e74c3c"
        for s in chart_df["Match Score (%)"]
    ]
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
        x=chart_df["Match Score (%)"].values,
        y=chart_df["Candidate"].values,
        orientation="h",
        marker_color=colors,
        text=[f"{s}%" for s in chart_df["Match Score (%)"]],
        textposition="outside",
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="gray",
                  annotation_text=f"Threshold ({threshold}%)")
    fig.update_layout(
        xaxis_title="Match Score (%)",
        yaxis_title="",
        height=max(300, len(chart_df) * 40),
        margin=dict(l=0, r=40, t=10, b=40),
        xaxis=dict(range=[0, 105]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Ranking Table ──
    st.subheader("Ranking Results")
    display_cols = ["Candidate", "Email", "Phone", "Experience (yrs)",
                    "Match Score (%)", "Keyword Overlap (%)", "Status"]
    st.dataframe(
        df[display_cols],
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
        status = row["Status"]
        icon = "\u2705" if "Shortlisted" in status else "\u26a0\ufe0f" if "Missing Required" in status else "\u274c"

        with st.expander(f"{icon} {candidate} — {score}% | {status}"):
            # Contact + experience row
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.markdown(f"**Email:** {row['Email'] or 'Not found'}")
            with info_col2:
                st.markdown(f"**Phone:** {row['Phone'] or 'Not found'}")
            with info_col3:
                exp = row["Experience (yrs)"]
                st.markdown(f"**Experience:** {exp:.0f} years" if exp else "**Experience:** Not detected")

            # Must-have warning
            if row["Missing Must-Haves"]:
                st.error(f"**Missing required skills:** {row['Missing Must-Haves']}")

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
                    st.warning(row["Missing Keywords"])
                else:
                    st.success("All key terms present!")

            st.markdown("**Resume Text Preview:**")
            original_idx = resume_names.index(candidate)
            preview = resume_texts[original_idx][:1000]
            st.text_area(
                "", value=preview, height=150, disabled=True,
                key=f"preview_{candidate}",
            )

    # ── Candidate Comparison ──
    st.subheader("Compare Candidates")
    candidates_list = df["Candidate"].tolist()
    selected = st.multiselect(
        "Select candidates to compare side-by-side:",
        candidates_list,
        default=candidates_list[:2] if len(candidates_list) >= 2 else candidates_list,
    )

    if selected:
        cols = st.columns(len(selected))
        for i, cand in enumerate(selected):
            row = df[df["Candidate"] == cand].iloc[0]
            with cols[i]:
                st.markdown(f"### {cand}")
                status = row["Status"]
                st.markdown(f"**Status:** {status}")
                st.metric("Match Score", f"{row['Match Score (%)']:.1f}%")
                st.metric("Keyword Overlap", f"{row['Keyword Overlap (%)']:.1f}%")
                exp = row["Experience (yrs)"]
                st.markdown(f"**Experience:** {exp:.0f} yrs" if exp else "**Experience:** N/A")
                st.markdown(f"**Email:** {row['Email'] or 'N/A'}")
                st.markdown(f"**Phone:** {row['Phone'] or 'N/A'}")

                if row["Missing Must-Haves"]:
                    st.error(f"Missing: {row['Missing Must-Haves']}")

                if row["Matched Keywords"]:
                    st.success(f"**Matched:** {row['Matched Keywords']}")
                if row["Missing Keywords"]:
                    st.warning(f"**Missing:** {row['Missing Keywords']}")

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
