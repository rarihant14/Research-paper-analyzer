
from typing import TypedDict, Optional


class ResearchState(TypedDict):
    # Input
    paper_text: str                # Full extracted text
    sections: dict                 # Parsed sections dict
    metadata: dict                 # Title, authors, year

    # (token-efficient slices) 
    analyzer_context: str
    summarizer_context: str
    citations_context: str
    insights_context: str

    # Agent outputs
    analysis: str
    summary: str
    citations: str
    insights: str

    #Review tracking
    scores: dict           # analysis: 8, "summary": 7}
    feedback: dict         # analysis: "Good.", ...}
    retries: dict          # analysis: 0, "summary": 1,}

    # Control flow
    current_phase: str     # Which agent is currently running
    logs: list             # Progress log shown in UI
    errors: list           # Error messages

    #Final output
    final_brief: str
