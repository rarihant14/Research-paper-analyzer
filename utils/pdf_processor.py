"""
Handles PDF text extraction and smart section parsing.
Token-efficient: each agent only receives the relevant section, not the full paper.
"""

import re
import io
import logging
import requests
import PyPDF2

logger = logging.getLogger(__name__)


# Section detection patterns (case-insensitive, with common header keywords)

SECTION_PATTERNS = {
    "abstract":     [r"\babstract\b"],
    "introduction": [r"\bintroduction\b", r"\b1[\.\s]+intro"],
    "related_work": [r"\brelated\s+work\b", r"\bliterature\s+review\b", r"\bbackground\b"],
    "methodology":  [r"\bmethodolog\w*\b", r"\bapproach\b", r"\bproposed\s+method\b",
                     r"\bframework\b", r"\barchitecture\b", r"\b\d[\.\s]+method"],
    "experiments":  [r"\bexperiment\w*\b", r"\bevaluation\b", r"\bimplementation\b",
                     r"\bsetup\b"],
    "results":      [r"\bresult\w*\b", r"\bfinding\w*\b", r"\bperformance\b"],
    "discussion":   [r"\bdiscussion\b", r"\banalysis\b", r"\blimitation\w*\b"],
    "conclusion":   [r"\bconclusion\w*\b", r"\bsummary\b", r"\bfuture\s+work\b"],
    "references":   [r"\breference\w*\b", r"\bbibliograph\w*\b"],
}

MAX_SECTION_CHARS = 3000   # ~750 tokens — enough context per agent call


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    text_parts = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())
    except Exception as e:
        logger.error("PDF read error: %s", e)
        raise ValueError(f"Could not read PDF: {e}")

    full_text = "\n".join(text_parts)
    if len(full_text) < 200:
        raise ValueError("Extracted text is too short. The PDF may be image-based or encrypted.")
    return full_text


def extract_text_from_url(url: str) -> str:
    """Download a PDF from a URL and extract its text."""
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            # Might be an HTML page — return raw text as fallback
            logger.warning("URL does not appear to be a PDF; using raw text.")
            return resp.text[:15000]
        return extract_text_from_pdf_bytes(resp.content)
    except requests.RequestException as e:
        raise ValueError(f"Could not download from URL: {e}")




def _find_section_start(text_lower: str, patterns: list[str]) -> int:
    """Return the earliest character index where any pattern matches."""
    best = -1
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m and (best == -1 or m.start() < best):
            best = m.start()
    return best


def extract_sections(text: str) -> dict:
    """
    Parse the paper text into named sections.
    Returns a dict: {section_name: text_snippet (up to MAX_SECTION_CHARS)}.
    Falls back to positional heuristics if headers are not found.
    """
    text_lower = text.lower()
    total = len(text)

    # Find start positions of each section
    positions = {}
    for sec, pats in SECTION_PATTERNS.items():
        idx = _find_section_start(text_lower, pats)
        if idx != -1:
            positions[sec] = idx

    # Sort sections by position
    ordered = sorted(positions.items(), key=lambda x: x[1])

    sections = {}
    for i, (sec, start) in enumerate(ordered):
        # End = start of next section or start+MAX_SECTION_CHARS
        end = ordered[i + 1][1] if i + 1 < len(ordered) else min(start + MAX_SECTION_CHARS, total)
        snippet = text[start:end].strip()
        sections[sec] = snippet[:MAX_SECTION_CHARS]

    # --- Positional fallbacks for missing sections ---
    if "abstract" not in sections:
        # First 1500 chars usually contain abstract
        sections["abstract"] = text[:1500].strip()

    if "methodology" not in sections and "experiments" not in sections:
        # Middle third of paper
        mid = total // 3
        sections["methodology"] = text[mid: mid + MAX_SECTION_CHARS].strip()

    if "results" not in sections:
        # 55–70 % of paper
        r_start = int(total * 0.55)
        sections["results"] = text[r_start: r_start + MAX_SECTION_CHARS].strip()

    if "conclusion" not in sections:
        # Last 10 % of paper (before references)
        c_start = max(0, total - 3000)
        sections["conclusion"] = text[c_start: c_start + MAX_SECTION_CHARS].strip()

    if "references" not in sections:
        # Very last part of paper
        sections["references"] = text[max(0, total - 4000):].strip()

    return sections

# Context builder (token-efficient)


def build_agent_context(sections: dict, agent_type: str) -> str:
    """
    Returns only the sections relevant to a particular agent.
    This is the key token-saving mechanism — each agent gets <2000 tokens of context.
    """
    agent_sections = {
        "analyzer":   ["abstract", "introduction", "methodology", "experiments", "results"],
        "summarizer": ["abstract", "introduction", "conclusion"],
        "citations":  ["references", "related_work"],
        "insights":   ["abstract", "results", "discussion", "conclusion"],
    }
    wanted = agent_sections.get(agent_type, ["abstract"])
    parts = []
    for sec in wanted:
        if sec in sections:
            parts.append(f"=== {sec.upper()} ===\n{sections[sec]}")

    context = "\n\n".join(parts)
    # Hard cap: 4000 chars ≈ 1000 tokens
    return context[:4000]


def extract_metadata(text: str) -> dict:
    """
    Quick heuristic metadata extraction from the first ~500 chars of the paper.
    No LLM call — saves tokens.
    """
    header = text[:800]
    lines = [l.strip() for l in header.split("\n") if l.strip()]

    title = lines[0] if lines else "Unknown Title"
    authors = lines[1] if len(lines) > 1 else "Unknown Authors"
    year_match = re.search(r"\b(19|20)\d{2}\b", header)
    year = year_match.group(0) if year_match else "Unknown Year"

    return {"title": title, "authors": authors, "year": year}
