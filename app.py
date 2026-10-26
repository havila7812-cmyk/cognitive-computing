"""
app.py
------
Streamlit dashboard for the Explainable Career Recommendation System.

Run:
    python -m streamlit run app.py
"""

import io
import pathlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import career_data as cd
from recommender import (
    CareerMatch,
    StudentProfile,
    recommend,
    validate_profile,
    load_dataset,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Explainable Career Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e0e7ff;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #a5b4fc;
        margin: 0 0 1rem 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(165,180,252,0.15);
        color: #a5b4fc;
        border: 1px solid rgba(165,180,252,0.3);
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.8rem;
        margin-right: 0.4rem;
    }

    /* Score cards */
    .score-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .score-rank {
        font-size: 0.75rem;
        color: #6366f1;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .score-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0.3rem 0;
    }
    .score-value {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .score-label {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.1rem;
    }
    .skill-tag {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 20px;
        padding: 0.2rem 0.65rem;
        font-size: 0.78rem;
        margin: 0.15rem;
    }
    .gap-tag {
        display: inline-block;
        background: rgba(251,191,36,0.1);
        color: #fbbf24;
        border: 1px solid rgba(251,191,36,0.25);
        border-radius: 20px;
        padding: 0.2rem 0.65rem;
        font-size: 0.78rem;
        margin: 0.15rem;
    }
    .suggest-tag {
        display: inline-block;
        background: rgba(52,211,153,0.1);
        color: #34d399;
        border: 1px solid rgba(52,211,153,0.25);
        border-radius: 20px;
        padding: 0.2rem 0.65rem;
        font-size: 0.78rem;
        margin: 0.15rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        border-bottom: 2px solid #6366f1;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    .info-box {
        background: rgba(99,102,241,0.08);
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        color: #94a3b8;
        margin: 0.8rem 0;
    }
    .warning-box {
        background: rgba(251,191,36,0.08);
        border-left: 3px solid #fbbf24;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        color: #fcd34d;
        margin: 0.8rem 0;
    }
    .medal-1 { color: #fbbf24; }
    .medal-2 { color: #94a3b8; }
    .medal-3 { color: #b45309; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Dataset loading (cached)
# ---------------------------------------------------------------------------
DATASET_PATH = (
    pathlib.Path(__file__).resolve().parent
    / "data" / "processed" / "career_dataset.csv"
)


@st.cache_data(show_spinner=False)
def _load_df() -> pd.DataFrame:
    return load_dataset(DATASET_PATH)


@st.cache_data(show_spinner=False)
def _skills_list(df_json: str) -> list[str]:
    df = pd.read_json(io.StringIO(df_json))
    return cd.all_skills(df)


@st.cache_data(show_spinner=False)
def _software_list(df_json: str) -> list[str]:
    df = pd.read_json(io.StringIO(df_json))
    return cd.all_software(df)


@st.cache_data(show_spinner=False)
def _interests_list(df_json: str) -> list[str]:
    df = pd.read_json(io.StringIO(df_json))
    return cd.all_interests(df)


@st.cache_data(show_spinner=False)
def _styles_list(df_json: str) -> list[str]:
    df = pd.read_json(io.StringIO(df_json))
    return cd.all_work_styles(df)


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
dataset_error = None
df = None

try:
    df = _load_df()
    df_json = df.to_json()
    skills_options   = _skills_list(df_json)
    software_options = _software_list(df_json)
    interests_options = _interests_list(df_json)
    styles_options   = _styles_list(df_json)
except FileNotFoundError as e:
    dataset_error = str(e)
except Exception as e:
    dataset_error = f"Unexpected error loading dataset: {e}"

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🎯 Explainable Career Recommendation System</div>
        <div class="hero-subtitle">
            Powered by O*NET occupational data · Rule-based reasoning · Transparent scoring
        </div>
        <span class="hero-badge">📊 Knowledge Representation</span>
        <span class="hero-badge">⚙️ Rule-Based Reasoning</span>
        <span class="hero-badge">🔍 Explainable AI</span>
        <span class="hero-badge">🎓 Cognitive Computing</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if dataset_error:
    st.error(
        f"**Dataset not found.** Run `python scripts/prepare_dataset.py` first.\n\n{dataset_error}"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: Student Profile Form
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 👤 Student Profile")
    st.markdown("*Fill in any combination of fields below.*")

    st.markdown("### 🛠️ Technical Skills")
    selected_skills = st.multiselect(
        "Select skills you know",
        options=skills_options,
        help="Choose from the list — these are skills directly from O*NET occupational data.",
        key="tech_skills",
    )

    st.markdown("### 💻 Software & Tools")
    selected_software = st.multiselect(
        "Select software / tools you know",
        options=software_options,
        help="Software technologies and tools drawn from O*NET data.",
        key="soft_skills",
    )

    st.markdown("### 🌟 Career Interests")
    selected_interests = st.multiselect(
        "Select your career interests",
        options=interests_options,
        help="Holland occupational interest categories from O*NET.",
        key="interests",
    )

    st.markdown("### 🎯 Work Styles")
    selected_styles = st.multiselect(
        "Select your preferred work styles",
        options=styles_options,
        help="Work-style preferences drawn from O*NET data.",
        key="styles",
    )

    st.markdown("### 🎓 Optional Details")
    education = st.selectbox(
        "Highest education level (optional)",
        ["", "High School / Secondary", "Diploma / Associate Degree",
         "Bachelor's Degree", "Master's Degree", "Doctoral Degree"],
        key="education",
    )
    goals = st.text_area(
        "Career goals (optional, for your reference)",
        placeholder="E.g., I want to work in cybersecurity or data science...",
        max_chars=300,
        key="goals",
    )

    st.markdown("---")
    run_btn = st.button("🚀 Get Recommendations", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
col_info, col_stat = st.columns([3, 1])
with col_info:
    st.markdown(
        """
        <div class="info-box">
        <b>How it works:</b> Select your skills, interests and work styles in the sidebar,
        then click <b>Get Recommendations</b>.  The system compares your profile against
        O*NET occupational data using a transparent weighted scoring formula and explains
        every recommendation.<br><br>
        <b>Disclaimer:</b> Scores represent a <em>profile match percentage</em>, not a
        probability of employment or success.  O*NET data is primarily US-based.
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_stat:
    st.metric("Occupations in dataset", len(df))
    st.metric("Skills catalogued", len(skills_options))

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if run_btn or st.session_state.get("last_results"):

    # Build profile
    profile = StudentProfile(
        technical_skills=[s.lower() for s in selected_skills],
        software_skills=[s.lower() for s in selected_software],
        career_interests=[i.lower() for i in selected_interests],
        work_styles=[s.lower() for s in selected_styles],
        education_level=education or None,
        career_goals=goals or None,
    )

    # Validate
    errors = validate_profile(profile)
    if errors:
        st.markdown(
            f'<div class="warning-box">⚠️ {errors[0]}</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Analysing your profile against O*NET data…"):
            results: list[CareerMatch] = recommend(profile, dataset=df, top_n=len(df))

        if run_btn:
            st.session_state["last_results"] = results

        if not results:
            st.warning("No recommendations generated. Please add more profile information.")
        else:
            # ------------------------------------------------------------------
            # Section 1: Top 3 recommendation cards
            # ------------------------------------------------------------------
            st.markdown('<div class="section-header">🏆 Top 3 Career Matches</div>', unsafe_allow_html=True)
            medals = ["🥇", "🥈", "🥉"]
            top3_cols = st.columns(3)

            for idx, match in enumerate(results[:3]):
                with top3_cols[idx]:
                    bar_color = ["#6366f1", "#8b5cf6", "#a78bfa"][idx]
                    pct = match.overall_score

                    st.markdown(
                        f"""
                        <div class="score-card">
                            <div class="score-rank">{medals[idx]} Rank #{match.rank}</div>
                            <div class="score-title">{match.occupation_title}</div>
                            <div class="score-value">{pct:.1f}<span style="font-size:1rem;color:#475569">%</span></div>
                            <div class="score-label">Profile Match Score</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Mini breakdown
                    sub1, sub2, sub3 = st.columns(3)
                    sub1.metric("Skills", f"{match.skill_score:.0f}%")
                    sub2.metric("Interests", f"{match.interest_score:.0f}%")
                    sub3.metric("Styles", f"{match.style_score:.0f}%")

                    # Matched skills tags
                    if match.matched_skills:
                        tags_html = "".join(
                            f'<span class="skill-tag">{s.title()}</span>'
                            for s in match.matched_skills[:6]
                        )
                        extra = len(match.matched_skills) - 6
                        if extra > 0:
                            tags_html += f'<span class="skill-tag">+{extra} more</span>'
                        st.markdown(f"**Matched skills:**<br>{tags_html}", unsafe_allow_html=True)

                    # Expandable full explanation
                    with st.expander("📖 Full explanation"):
                        st.markdown(f"**Description:** {match.description[:400]}{'...' if len(match.description) > 400 else ''}")
                        st.code(match.explanation, language=None)

                        if match.unreported_skills:
                            gap_html = "".join(
                                f'<span class="gap-tag">⚠ {s.title()}</span>'
                                for s in match.unreported_skills[:10]
                            )
                            st.markdown(
                                f"**Skills not yet reported** *(for awareness only)*:<br>{gap_html}",
                                unsafe_allow_html=True,
                            )

                        if match.suggested_skills:
                            sug_html = "".join(
                                f'<span class="suggest-tag">📚 {s.title()}</span>'
                                for s in match.suggested_skills
                            )
                            st.markdown(
                                f"**Suggested skills to learn next:**<br>{sug_html}",
                                unsafe_allow_html=True,
                            )

            # ------------------------------------------------------------------
            # Section 2: Comparison chart
            # ------------------------------------------------------------------
            st.markdown('<div class="section-header">📊 Career Score Comparison</div>', unsafe_allow_html=True)

            chart_data = results[:min(10, len(results))]
            titles     = [m.occupation_title for m in chart_data]
            s_scores   = [m.skill_score    for m in chart_data]
            i_scores   = [m.interest_score for m in chart_data]
            ws_scores  = [m.style_score    for m in chart_data]
            overall    = [m.overall_score  for m in chart_data]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Skills Match",
                x=titles, y=s_scores,
                marker_color="#6366f1",
                opacity=0.85,
            ))
            fig.add_trace(go.Bar(
                name="Interests Match",
                x=titles, y=i_scores,
                marker_color="#8b5cf6",
                opacity=0.85,
            ))
            fig.add_trace(go.Bar(
                name="Work Style Match",
                x=titles, y=ws_scores,
                marker_color="#a78bfa",
                opacity=0.85,
            ))
            fig.add_trace(go.Scatter(
                name="Overall Score",
                x=titles, y=overall,
                mode="markers+lines",
                marker=dict(color="#fbbf24", size=10, symbol="diamond"),
                line=dict(color="#fbbf24", width=2, dash="dot"),
            ))
            fig.update_layout(
                barmode="group",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font=dict(color="#94a3b8", family="Inter"),
                xaxis=dict(tickangle=-35, gridcolor="#1e293b"),
                yaxis=dict(title="Score (0–100)", gridcolor="#1e293b", range=[0, 105]),
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)",
                ),
                margin=dict(l=40, r=20, t=60, b=120),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            # ------------------------------------------------------------------
            # Section 3: Full ranked table
            # ------------------------------------------------------------------
            st.markdown('<div class="section-header">📋 Complete Ranked Career Table</div>', unsafe_allow_html=True)

            table_rows = []
            for m in results:
                table_rows.append({
                    "Rank": m.rank,
                    "Career Title": m.occupation_title,
                    "SOC Code": m.occupation_code,
                    "Overall %": f"{m.overall_score:.1f}",
                    "Skills %": f"{m.skill_score:.1f}",
                    "Interests %": f"{m.interest_score:.1f}",
                    "Styles %": f"{m.style_score:.1f}",
                    "Matched Skills": len(m.matched_skills),
                    "Skill Gaps": len(m.unreported_skills),
                })

            table_df = pd.DataFrame(table_rows)
            st.dataframe(table_df, use_container_width=True, hide_index=True)

            # ------------------------------------------------------------------
            # Section 4: Skill-gap analysis
            # ------------------------------------------------------------------
            st.markdown('<div class="section-header">🔍 Skill-Gap Analysis</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="info-box">
                The following skills appear in the O*NET data for each career but were not included
                in your profile.  They are labelled <em>not yet reported</em> — this does not
                confirm that you lack these skills.
                </div>
                """,
                unsafe_allow_html=True,
            )

            for m in results[:5]:
                if m.unreported_skills:
                    with st.expander(f"📌 {m.occupation_title} — {len(m.unreported_skills)} skills not yet reported"):
                        gap_html = "".join(
                            f'<span class="gap-tag">⚠ {s.title()}</span>'
                            for s in m.unreported_skills
                        )
                        st.markdown(gap_html, unsafe_allow_html=True)
                        if m.suggested_skills:
                            sug_html = "".join(
                                f'<span class="suggest-tag">📚 {s.title()}</span>'
                                for s in m.suggested_skills
                            )
                            st.markdown(
                                f"<br><b>Suggested skills to learn next:</b><br>{sug_html}",
                                unsafe_allow_html=True,
                            )

            # ------------------------------------------------------------------
            # Section 5: Download
            # ------------------------------------------------------------------
            st.markdown('<div class="section-header">⬇️ Download Results</div>', unsafe_allow_html=True)
            download_df = pd.DataFrame([{
                "Rank": m.rank,
                "SOC Code": m.occupation_code,
                "Career Title": m.occupation_title,
                "Overall Score": m.overall_score,
                "Skill Score": m.skill_score,
                "Interest Score": m.interest_score,
                "Style Score": m.style_score,
                "Matched Skills": "; ".join(m.matched_skills),
                "Skills Not Yet Reported": "; ".join(m.unreported_skills),
                "Suggested Skills": "; ".join(m.suggested_skills),
                "Description": m.description,
            } for m in results])

            csv_bytes = download_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download recommendations as CSV",
                data=csv_bytes,
                file_name="career_recommendations.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Section 6: Dataset information
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header">📚 Dataset Information</div>', unsafe_allow_html=True)
with st.expander("View O*NET dataset details"):
    st.markdown(
        """
        **Source**: O*NET 31.0 Database, National Center for O*NET Development
        ([onetcenter.org](https://www.onetcenter.org/database.html))

        This system uses five O*NET data files:

        | File | Description |
        |------|-------------|
        | occupation_data.csv | Occupation titles and descriptions |
        | essential_skills.csv | Skills importance ratings (IM scale) |
        | career_interest_types.csv | Holland occupational interest scores |
        | work_styles.csv | Work-style importance ratings |
        | software_skills.csv | Tools and Technology listings |

        **Important notes**:
        - O*NET data is primarily based on US occupational information and
          may require adaptation for students in other countries (e.g., India).
        - Learning suggestions are derived from the O*NET skill lists for each
          occupation, not from external curricula.
        - This system does **not** use machine learning or generative AI.
        """
    )
    if df is not None:
        st.dataframe(cd.careers_summary(df), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Section 7: Cognitive Computing concepts
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header">🧠 Cognitive Computing Concepts</div>', unsafe_allow_html=True)
with st.expander("How this system demonstrates Cognitive Computing"):
    st.markdown(
        """
        | Concept | Implementation |
        |---------|---------------|
        | **Knowledge Representation** | Structured O*NET occupational dataset with skill, interest, work-style and software fields per occupation |
        | **Rule-Based Reasoning** | Deterministic weighted scoring formula: Skills 60% + Interests 30% + Work Styles 10% |
        | **Explainability** | Every score comes with a written breakdown — matched items, unmatched items, and score per category |
        | **Personalization** | Rankings change when the student profile changes; weights re-normalize for missing categories |
        | **Decision Support** | Side-by-side career comparisons, skill-gap identification and learning suggestions |

        **Scoring formula:**

        ```
        Overall = (W_skills × Skill_Score) + (W_interests × Interest_Score) + (W_styles × Style_Score)

        where W_skills + W_interests + W_styles = 1.0
        Default: W_skills=0.60, W_interests=0.30, W_styles=0.10
        ```

        If a category is omitted from the profile, its weight is redistributed proportionally.

        > This system is **not** a trained machine-learning model and does **not** predict
        > actual employment outcomes or career success probabilities.
        """
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <p style="text-align:center; color:#475569; font-size:0.82rem;">
    Explainable Career Recommendation System &nbsp;|&nbsp;
    Built with Python, Streamlit & O*NET Data &nbsp;|&nbsp;
    O*NET is sponsored by the U.S. Department of Labor &nbsp;|&nbsp;
    <a href="https://www.onetcenter.org" target="_blank" style="color:#6366f1;">onetcenter.org</a>
    </p>
    """,
    unsafe_allow_html=True,
)
