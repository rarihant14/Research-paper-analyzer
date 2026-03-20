"""
analyzer.py — Paper Analyzer Agent
Extracts: research problem, hypothesis, methodology, experiments, key findings.
"""

from utils.gemini_client import call_gemini

PROMPT_TEMPLATE = """\
You are a research paper analyst. Analyze the paper excerpt below and extract:
1. Research Problem / Hypothesis (1-2 sentences)
2. Methodology & Approach (2-3 sentences)
3. Key Experiments / Studies (2-3 bullet points)
4. Main Findings & Results (3-4 bullet points)

Be concise. Do NOT repeat the input. Use plain text only (no markdown headers).

PAPER EXCERPT:
{context}

OUTPUT FORMAT (use exactly these labels):
PROBLEM: <text>
METHODOLOGY: <text>
EXPERIMENTS:
- <item>
FINDINGS:
- <item>
"""

def run_analyzer(context: str) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context[:3500])
    return call_gemini(prompt, task_type="analysis", temperature=0.2)
