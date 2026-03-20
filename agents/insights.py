"""
insights.py — Key Insights Agent
Generates practical takeaways, field implications, and potential applications.
"""

from utils.gemini_client import call_gemini

PROMPT_TEMPLATE = """\
Based on this research analysis, provide:

1. PRACTICAL TAKEAWAYS (3 points): What can practitioners do with these findings?
2. FIELD IMPLICATIONS (2 points): How does this advance the research field?
3. POTENTIAL APPLICATIONS (2-3 points): Real-world use cases enabled by this work.
4. LIMITATIONS & FUTURE WORK (2 points): Key gaps or next research directions.

Be specific and actionable. Use plain text bullet points only.

RESEARCH CONTEXT:
{context}

PRIOR ANALYSIS:
{analysis}
"""

def run_insights(context: str, analysis: str) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context[:2000], analysis=analysis[:1000])
    return call_gemini(prompt, task_type="insights", temperature=0.4)
