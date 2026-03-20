import io
import time
import traceback

import streamlit as st

from utils.pdf_processor import (
    extract_text_from_pdf_bytes,
    extract_text_from_url,
)
from graph.workflow import run_pipeline

st.set_page_config(
    page_title="Research Paper Analyzer",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
.agent-box {
    border-radius: 8px;
    padding: 10px 16px;
    margin: 4px 0;
    font-size: 0.9rem;
}
.agent-running { background: #fff3cd; border-left: 4px solid #ffc107; }
.agent-done    { background: #d4edda; border-left: 4px solid #28a745; }
.agent-pending { background: #f8f9fa; border-left: 4px solid #dee2e6; color: #6c757d; }
.score-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: bold;
    font-size: 0.8rem;
}
.score-high { background: #d4edda; color: #155724; }
.score-mid  { background: #fff3cd; color: #856404; }
.score-low  { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

AGENTS = [
    ("🤖", "Boss Agent",              "Orchestrates the full pipeline"),
    ("🔬", "Paper Analyzer Agent",    "Extracts methodology & findings"),
    ("🔎", "Review Agent",            "Quality control & scoring"),
    ("📝", "Summary Generator Agent", "Creates 150-200 word summary"),
    ("📚", "Citation Extractor Agent","Organizes all references"),
    ("💡", "Key Insights Agent",      "Generates practical takeaways"),
]


def render_agent_pipeline(current_phase: str) -> None:
    phase_map = {
        "init":      -1,
        "analysis":   1,
        "summary":    3,
        "citations":  4,
        "insights":   5,
        "done":       6,
    }
    active = phase_map.get(current_phase, -1)

    st.markdown("#### 🔄 Agent Pipeline")
    for i, (icon, name, desc) in enumerate(AGENTS):
        if i < active:
            cls = "agent-done"
            status = "✅ Done"
        elif i == active:
            cls = "agent-running"
            status = "⚡ Running…"
        else:
            cls = "agent-pending"
            status = "⏳ Pending"
        st.markdown(
            f'<div class="agent-box {cls}">'
            f'{icon} <b>{name}</b> — {desc} &nbsp;&nbsp; <span style="float:right">{status}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_score_badge(score: int) -> str:
    if score >= 8:
        cls = "score-high"
    elif score >= 6:
        cls = "score-mid"
    else:
        cls = "score-low"
    return f'<span class="score-badge {cls}">{score}/10</span>'


def render_review_table(scores: dict, feedback: dict, retries: dict) -> None:
    if not scores:
        return
    st.markdown("#### 📊 Quality Review Scores")
    cols = st.columns(len(scores))
    labels = {
        "analysis":  "🔬 Analysis",
        "summary":   "📝 Summary",
        "citations": "📚 Citations",
        "insights":  "💡 Insights",
    }
    for col, (key, score) in zip(cols, scores.items()):
        with col:
            st.markdown(
                f"**{labels.get(key, key)}**<br>"
                f"{render_score_badge(score)}<br>"
                f"<small>Retries: {retries.get(key, 0)}</small>",
                unsafe_allow_html=True,
            )
            if feedback.get(key):
                st.caption(feedback[key])



def main():
    # ── Header ──────────────────────────────────────────────────────────────
    st.title("🔬 AI-Powered Research Paper Analyzer")
    st.markdown(
        "Multi-agent system that reads academic papers and generates a complete research brief. "
        "Built with **LangGraph** + **Google Gemini 2.5 Flash**."
    )
    st.divider()


    with st.sidebar:
        st.header("⚙️ Settings")
        st.info(
            "Uses **Gemini 2.5 Flash** (free tier).\n\n"
            "Token-efficient RAG: each agent receives only relevant paper sections."
        )
        st.markdown("---")
        st.markdown("**Architecture**")
        st.markdown(
            "- Boss Agent (Orchestrator)\n"
            "- Paper Analyzer Agent\n"
            "- Summary Generator Agent\n"
            "- Citation Extractor Agent\n"
            "- Key Insights Agent\n"
            "- Review Agent (QC loop, max 2 retries)"
        )
        st.markdown("---")
        st.caption("Set `GEMINI_API_KEY` in your `.env` file.")


    st.subheader("📄 Input Paper")
    input_method = st.radio(
        "Input method:",
        ["Upload PDF", "Paste PDF URL", "Paste Text"],
        horizontal=True,
    )

    paper_text = ""

    if input_method == "Upload PDF":
        uploaded = st.file_uploader("Upload a research paper PDF", type=["pdf"])
        if uploaded:
            with st.spinner("Extracting text from PDF…"):
                try:
                    paper_text = extract_text_from_pdf_bytes(uploaded.read())
                    st.success(f"✅ Extracted {len(paper_text):,} characters from PDF.")
                except Exception as e:
                    st.error(f"PDF extraction failed: {e}")

    elif input_method == "Paste PDF URL":
        url = st.text_input("PDF URL (e.g. arXiv PDF link):",
                            placeholder="https://arxiv.org/pdf/2301.00234")
        if url and st.button("Fetch Paper"):
            with st.spinner("Downloading and extracting paper…"):
                try:
                    paper_text = extract_text_from_url(url)
                    st.success(f"✅ Fetched {len(paper_text):,} characters.")
                except Exception as e:
                    st.error(f"Failed to fetch paper: {e}")

    elif input_method == "Paste Text":
        paper_text = st.text_area(
            "Paste paper text here:",
            height=300,
            placeholder="Paste the full text of the research paper…",
        )
        if paper_text:
            st.info(f"Text length: {len(paper_text):,} characters")

    # ── Preview ──────────────────────────────────────────────────────────────
    if paper_text:
        with st.expander("👀 Paper text preview (first 1000 chars)"):
            st.text(paper_text[:1000])

    st.divider()

    if not paper_text:
        st.info("👆 Provide a paper above to start analysis.")
        return

    if st.button("🚀 Analyze Paper", type="primary", use_container_width=True):

    
        progress_bar  = st.progress(0, text="Starting pipeline…")
        phase_display = st.empty()
        log_display   = st.empty()

        # Run in a try-except so UI never crashes
        try:
            with st.spinner("Multi-agent pipeline running…"):
                start = time.time()
                result = run_pipeline(paper_text)
                elapsed = time.time() - start

            progress_bar.progress(100, text="✅ Pipeline complete!")

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.code(traceback.format_exc())
            return

        # Results 
        st.success(f"✅ Analysis complete in {elapsed:.1f}s")
        st.divider()

        # Agent pipeline status
        render_agent_pipeline("done")
        st.divider()

        # Review scores
        render_review_table(
            result.get("scores", {}),
            result.get("feedback", {}),
            result.get("retries", {}),
        )
        st.divider()

        # Progress log
        logs = result.get("logs", [])
        if logs:
            with st.expander("📋 Workflow Log", expanded=False):
                for log in logs:
                    st.markdown(f"- {log}")

        # Errors
        errors = result.get("errors", [])
        if errors:
            with st.expander("⚠️ Errors / Warnings", expanded=False):
                for err in errors:
                    st.warning(err)

        # Final Research Brief
        st.subheader("📑 Research Brief")

        tabs = st.tabs([
            "📋 Full Brief",
            "🔬 Analysis",
            "📝 Summary",
            "📚 Citations",
            "💡 Insights",
        ])

        with tabs[0]:
            brief = result.get("final_brief", "")
            if brief:
                st.text_area("Complete Research Brief", brief, height=600)
                st.download_button(
                    label="⬇️ Download Research Brief (.txt)",
                    data=brief,
                    file_name="research_brief.txt",
                    mime="text/plain",
                )
            else:
                st.warning("No brief was generated.")

        with tabs[1]:
            st.markdown(result.get("analysis", "N/A"))

        with tabs[2]:
            st.markdown(result.get("summary", "N/A"))

        with tabs[3]:
            st.markdown(result.get("citations", "N/A"))

        with tabs[4]:
            st.markdown(result.get("insights", "N/A"))

        # Metadata card
        meta = result.get("metadata", {})
        if meta:
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Title", meta.get("title", "N/A")[:50])
            c2.metric("Authors", meta.get("authors", "N/A")[:40])
            c3.metric("Year", meta.get("year", "N/A"))


if __name__ == "__main__":
    main()
