"""
reviewer.py — Review Agent (Quality Control)
Scores each agent's output 1-10. Uses local heuristics first to save quota;
only calls Gemini API for borderline cases.
"""

import re
from utils.gemini_client import call_gemini

PROMPT_TEMPLATE = """\
Score this {agent_type} output for a research paper (1-10 scale).
Criteria: accuracy, completeness, clarity, relevance.

OUTPUT TO REVIEW:
{output}

Reply in EXACTLY this format (nothing else):
SCORE: <integer 1-10>
FEEDBACK: <one sentence, or "Approved." if score >= 7>
"""

_MIN_LENGTHS = {"analysis": 200, "summary": 80, "citations": 50, "insights": 100}
_FAILURE_PHRASES = ["[failed", "[error", "i cannot", "i'm unable", "as an ai"]

def _local_score(agent_type: str, output: str):
    """Fast local check — returns (score, feedback) or None for borderline."""
    text = output.strip().lower()
    if any(p in text for p in _FAILURE_PHRASES):
        return 2, "Output contains error or refusal text."
    min_len = _MIN_LENGTHS.get(agent_type, 100)
    if len(output) < min_len // 2:
        return 3, "Output is too short; needs more detail."
    has_structure = bool(re.search(r"[-•\d]\s+\w", output))
    if len(output) > min_len * 2 and has_structure:
        return 8, "Approved."
    return None   # borderline — use Gemini

def run_reviewer(agent_type: str, output: str) -> tuple:
    local = _local_score(agent_type, output)
    if local is not None:
        return local
    prompt = PROMPT_TEMPLATE.format(agent_type=agent_type, output=output[:900])
    raw = call_gemini(prompt, task_type="review", temperature=0.1)
    return _parse_score(raw), _parse_feedback(raw)

def _parse_score(text: str) -> int:
    m = re.search(r"SCORE:\s*(\d+)", text, re.IGNORECASE)
    if m:
        return max(1, min(10, int(m.group(1))))
    m = re.search(r"\b([1-9]|10)\b", text)
    return int(m.group(1)) if m else 5

def _parse_feedback(text: str) -> str:
    m = re.search(r"FEEDBACK:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()[:200]
    return text.strip()[:200]
