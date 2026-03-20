"""
summarizer.py — Summary Generator Agent
Creates a 150-200 word executive summary.
"""

from utils.gemini_client import call_gemini

PROMPT_TEMPLATE = """\
Write a clear executive summary of this research paper.
Requirements:
- Exactly 150-200 words
- Cover: problem, approach, key results, significance
- Plain prose (no bullet points, no headers)
- Accessible to non-experts

ABSTRACT & CONCLUSION EXCERPT:
{context}

PRIOR ANALYSIS (use for accuracy):
{analysis}

Write ONLY the executive summary paragraph(s):
"""

def run_summarizer(context: str, analysis: str) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context[:1500], analysis=analysis[:1000])
    return call_gemini(prompt, task_type="summary", temperature=0.4)
