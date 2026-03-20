"""
workflow.py — LangGraph Multi-Agent Workflow

Architecture:
  preprocess → analyze → review_analysis ─┬→ (retry) → analyze
                                           └→ summarize → review_summary ─┬→ (retry)
                                                                           └→ extract_citations → ...
                                                                              → generate_insights → ...
                                                                              → combine → END
"""

import logging
import time

from langgraph.graph import StateGraph, END

from graph.state import ResearchState
from utils.pdf_processor import extract_sections, build_agent_context, extract_metadata
from agents.analyzer import run_analyzer
from agents.summarizer import run_summarizer
from agents.citation_extractor import run_citation_extractor
from agents.insights import run_insights
from agents.reviewer import run_reviewer

logger = logging.getLogger(__name__)

REVIEW_THRESHOLD = 7   # Score >= 7 → approved
MAX_RETRIES      = 2   # Max re-generations per agent
INTER_CALL_DELAY = 2   # Seconds between Gemini calls (free-tier courtesy)



def _log(state: ResearchState, msg: str) -> None:
    state["logs"].append(msg)
    logger.info(msg)



# NODE DEFINITIONS


def preprocess_node(state: ResearchState) -> ResearchState:
    """Boss Agent: parse paper, extract sections, build agent contexts."""
    _log(state, "🤖 Boss Agent: Preprocessing paper…")
    try:
        sections  = extract_sections(state["paper_text"])
        metadata  = extract_metadata(state["paper_text"])

        state["sections"]           = sections
        state["metadata"]           = metadata
        state["analyzer_context"]   = build_agent_context(sections, "analyzer")
        state["summarizer_context"] = build_agent_context(sections, "summarizer")
        state["citations_context"]  = build_agent_context(sections, "citations")
        state["insights_context"]   = build_agent_context(sections, "insights")

        state["scores"]   = {}
        state["feedback"] = {}
        state["retries"]  = {"analysis": 0, "summary": 0, "citations": 0, "insights": 0}

        _log(state, f"✅ Preprocessing done. Sections found: {list(sections.keys())}")
    except Exception as e:
        state["errors"].append(f"Preprocess error: {e}")
        _log(state, f"❌ Preprocessing failed: {e}")
    return state


#Analyzer

def analyze_node(state: ResearchState) -> ResearchState:
    """Agent 1: Paper Analyzer — extracts methodology, findings."""
    retry = state["retries"].get("analysis", 0)
    label = "Retrying" if retry > 0 else "Running"
    _log(state, f"🔬 Paper Analyzer Agent ({label}, attempt {retry + 1})…")
    try:
        state["current_phase"] = "analysis"
        state["analysis"] = run_analyzer(state["analyzer_context"])
        time.sleep(INTER_CALL_DELAY)
    except Exception as e:
        state["errors"].append(f"Analyzer error: {e}")
        state["analysis"] = f"[Analyzer failed: {e}]"
    return state


def review_analysis_node(state: ResearchState) -> ResearchState:
    """Review Agent: checks Paper Analyzer output."""
    _log(state, "🔎 Review Agent: Checking analysis quality…")
    try:
        score, feedback = run_reviewer("analysis", state["analysis"])
        state["scores"]["analysis"]   = score
        state["feedback"]["analysis"] = feedback
        _log(state, f"   Analysis score: {score}/10 — {feedback}")
        time.sleep(INTER_CALL_DELAY)
    except Exception as e:
        state["errors"].append(f"Review analysis error: {e}")
        state["scores"]["analysis"] = REVIEW_THRESHOLD  # pass on error
    return state


def _route_analysis(state: ResearchState) -> str:
    """Conditional edge: approve analysis or retry."""
    score   = state["scores"].get("analysis", 0)
    retries = state["retries"].get("analysis", 0)
    if score >= REVIEW_THRESHOLD or retries >= MAX_RETRIES:
        return "approved"
    state["retries"]["analysis"] = retries + 1
    return "retry"


#Summarizer

def summarize_node(state: ResearchState) -> ResearchState:
    """Agent 2: Summary Generator."""
    retry = state["retries"].get("summary", 0)
    label = "Retrying" if retry > 0 else "Running"
    _log(state, f"📝 Summary Generator Agent ({label}, attempt {retry + 1})…")
    try:
        state["current_phase"] = "summary"
        state["summary"] = run_summarizer(
            state["summarizer_context"],
            state["analysis"],
        )
        time.sleep(INTER_CALL_DELAY)
    except Exception as e:
        state["errors"].append(f"Summarizer error: {e}")
        state["summary"] = f"[Summary failed: {e}]"
    return state


def review_summary_node(state: ResearchState) -> ResearchState:
    _log(state, "🔎 Review Agent: Checking summary quality…")
    try:
        score, feedback = run_reviewer("summary", state["summary"])
        state["scores"]["summary"]   = score
        state["feedback"]["summary"] = feedback
        _log(state, f"   Summary score: {score}/10 — {feedback}")
        time.sleep(INTER_CALL_DELAY)
    except Exception as e:
        state["errors"].append(f"Review summary error: {e}")
        state["scores"]["summary"] = REVIEW_THRESHOLD
    return state


def _route_summary(state: ResearchState) -> str:
    score   = state["scores"].get("summary", 0)
    retries = state["retries"].get("summary", 0)
    if score >= REVIEW_THRESHOLD or retries >= MAX_RETRIES:
        return "approved"
    state["retries"]["summary"] = retries + 1
    return "retry"


#Citation Extractor 

def citations_node(state: ResearchState) -> ResearchState:
    """Agent 3: Citation Extractor."""
    retry = state["retries"].get("citations", 0)
    label = "Retrying" if retry > 0 else "Running"
    _log(state, f"📚 Citation Extractor Agent ({label}, attempt {retry + 1})…")
    try:
        state["current_phase"] = "citations"
        state["citations"] = run_citation_extractor(state["citations_context"])
        time.sleep(INTER_CALL_DELAY)
    except Exception as e:
        state["errors"].append(f"Citation extractor error: {e}")
        state["citations"] = f"[Citations failed: {e}]"
    return state


def review_citations_node(state: ResearchState) -> ResearchState:
    _log(state, "🔎 Review Agent: Checking citations quality…")
    try:
        score, feedback = run_reviewer("citations", state["citations"])
        state["scores"]["citations"]   = score
        state["feedback"]["citations"] = feedback
        _log(state, f"   Citations score: {score}/10 — {feedback}")
        time.sleep(INTER_CALL_DELAY)
    except Exception as e:
        state["errors"].append(f"Review citations error: {e}")
        state["scores"]["citations"] = REVIEW_THRESHOLD
    return state


def _route_citations(state: ResearchState) -> str:
    score   = state["scores"].get("citations", 0)
    retries = state["retries"].get("citations", 0)
    if score >= REVIEW_THRESHOLD or retries >= MAX_RETRIES:
        return "approved"
    state["retries"]["citations"] = retries + 1
    return "retry"


#Key Insights

def insights_node(state: ResearchState) -> ResearchState:
    """Agent 4: Key Insights Generator."""
    retry = state["retries"].get("insights", 0)
    label = "Retrying" if retry > 0 else "Running"
    _log(state, f"💡 Key Insights Agent ({label}, attempt {retry + 1})…")
    try:
        state["current_phase"] = "insights"
        state["insights"] = run_insights(
            state["insights_context"],
            state["analysis"],
        )
        time.sleep(INTER_CALL_DELAY)
    except Exception as e:
        state["errors"].append(f"Insights error: {e}")
        state["insights"] = f"[Insights failed: {e}]"
    return state


def review_insights_node(state: ResearchState) -> ResearchState:
    _log(state, "🔎 Review Agent: Checking insights quality…")
    try:
        score, feedback = run_reviewer("insights", state["insights"])
        state["scores"]["insights"]   = score
        state["feedback"]["insights"] = feedback
        _log(state, f"   Insights score: {score}/10 — {feedback}")
        time.sleep(INTER_CALL_DELAY)
    except Exception as e:
        state["errors"].append(f"Review insights error: {e}")
        state["scores"]["insights"] = REVIEW_THRESHOLD
    return state


def _route_insights(state: ResearchState) -> str:
    score   = state["scores"].get("insights", 0)
    retries = state["retries"].get("insights", 0)
    if score >= REVIEW_THRESHOLD or retries >= MAX_RETRIES:
        return "approved"
    state["retries"]["insights"] = retries + 1
    return "retry"


# (Boss Agent) 

def combine_node(state: ResearchState) -> ResearchState:
    """Boss Agent: assembles all approved outputs into the final research brief."""
    _log(state, "🤖 Boss Agent: Compiling final research brief…")

    meta = state.get("metadata", {})
    scores = state.get("scores", {})

    brief = f"""
            RESEARCH BRIEF — AI ANALYSIS REPORT           

 1. PAPER METADATA
Title   : {meta.get('title', 'N/A')}
Authors : {meta.get('authors', 'N/A')}
Year    : {meta.get('year', 'N/A')}

 2. RESEARCH ANALYSIS  [Quality Score: {scores.get('analysis', 'N/A')}/10]
{state.get('analysis', 'N/A')}

 3. EXECUTIVE SUMMARY  [Quality Score: {scores.get('summary', 'N/A')}/10]
{state.get('summary', 'N/A')}


 4. CITATIONS & REFERENCES  [Quality Score: {scores.get('citations', 'N/A')}/10]
{state.get('citations', 'N/A')}

 5. KEY INSIGHTS & TAKEAWAYS  [Quality Score: {scores.get('insights', 'N/A')}/10]
{state.get('insights', 'N/A')}

 ANALYSIS COMPLETE — Generated by Multi-Agent Research Analyzer
""".strip()

    state["final_brief"] = brief
    state["current_phase"] = "done"
    _log(state, "✅ Research brief compiled successfully!")
    return state



# GRAPH CONSTRUCTION


def build_workflow() -> StateGraph:
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("preprocess",        preprocess_node)
    graph.add_node("analyze",           analyze_node)
    graph.add_node("review_analysis",   review_analysis_node)
    graph.add_node("summarize",         summarize_node)
    graph.add_node("review_summary",    review_summary_node)
    graph.add_node("extract_citations", citations_node)
    graph.add_node("review_citations",  review_citations_node)
    graph.add_node("generate_insights", insights_node)
    graph.add_node("review_insights",   review_insights_node)
    graph.add_node("combine",           combine_node)

    # Sequential edges
    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess",        "analyze")
    graph.add_edge("analyze",           "review_analysis")

    # Conditional: analysis approved → summarize, else retry analyze
    graph.add_conditional_edges(
        "review_analysis",
        _route_analysis,
        {"approved": "summarize", "retry": "analyze"},
    )

    graph.add_edge("summarize",         "review_summary")
    graph.add_conditional_edges(
        "review_summary",
        _route_summary,
        {"approved": "extract_citations", "retry": "summarize"},
    )

    graph.add_edge("extract_citations", "review_citations")
    graph.add_conditional_edges(
        "review_citations",
        _route_citations,
        {"approved": "generate_insights", "retry": "extract_citations"},
    )

    graph.add_edge("generate_insights", "review_insights")
    graph.add_conditional_edges(
        "review_insights",
        _route_insights,
        {"approved": "combine", "retry": "generate_insights"},
    )

    graph.add_edge("combine", END)

    return graph.compile()


#Public entry point

def run_pipeline(paper_text: str) -> ResearchState:
    """
    Run the full multi-agent pipeline on extracted paper text.
    Returns the final ResearchState.
    """
    initial_state: ResearchState = {
        "paper_text":          paper_text,
        "sections":            {},
        "metadata":            {},
        "analyzer_context":    "",
        "summarizer_context":  "",
        "citations_context":   "",
        "insights_context":    "",
        "analysis":            "",
        "summary":             "",
        "citations":           "",
        "insights":            "",
        "scores":              {},
        "feedback":            {},
        "retries":             {},
        "current_phase":       "init",
        "logs":                [],
        "errors":              [],
        "final_brief":         "",
    }
    workflow = build_workflow()
    result   = workflow.invoke(initial_state)
    return result
