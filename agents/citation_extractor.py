"""
citation_extractor.py — Citation Extractor Agent
Extracts and organizes all references from the paper.
"""

from utils.gemini_client import call_gemini

PROMPT_TEMPLATE = """\
You are a research librarian. From the references section below, extract and organize citations.

For each reference list: Author(s), Year, Title, Venue (journal/conference).
List up to 20 most important references. Mark 2-3 as KEY RELATED WORKS.

REFERENCES SECTION:
{context}

OUTPUT FORMAT:
KEY RELATED WORKS:
1. <citation>

ALL REFERENCES:
1. <Author(s)> (<Year>). <Title>. <Venue>
"""

def run_citation_extractor(context: str) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context[:3500])
    return call_gemini(prompt, task_type="citations", temperature=0.1)
