"""
Free-tier RPM budget :
  gemini-2.5-flash          -> 4 RPM  (free limit ~5, we use 4)
  gemini-2.5-flash-lite     -> 8 RPM  (free limit ~10, we use 8)

Why lower than Google's stated limit?
  The pipeline fires 4-8 calls in a burst.  Staying well below the
  hard limit avoids 429s entirely instead of hitting-then-waiting.

Model priority order (free-tier friendly):
  1. gemini-2.5-flash-lite   — faster, 8 RPM budget  <- default
  2. gemini-2.5-flash        — smarter, 4 RPM budget  <- fallback / reasoning
"""

# GROQ
# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage
#
# PRIMARY_MODEL   = "llama-3.3-70b-versatile"
# FAST_MODEL      = "llama-3.1-8b-instant"
# REASONING_MODEL = "qwen/qwen-3-32b"
#
# def _get_groq_client(model_name, temperature=0.3, max_tokens=800):
#     return ChatGroq(
#         model=model_name,
#         api_key=os.getenv("GROQ_API_KEY", ""),
#         temperature=temperature,
#         max_tokens=max_tokens,
#     )
#
# def call_llm(prompt, task_type="default", use_reasoning=False,
#              temperature=0.3, max_retries=3):
#     model_name = REASONING_MODEL if use_reasoning else PRIMARY_MODEL
#     if use_reasoning:
#         temperature = max(temperature, 0.5)
#     max_tokens = {"analysis":900,"summary":400,"citations":800,
#                   "insights":700,"review":120}.get(task_type, 800)
#     for attempt in range(1, max_retries + 1):
#         try:
#             client = _get_groq_client(model_name, temperature, max_tokens)
#             response = client.invoke([HumanMessage(content=prompt)])
#             text = response.content.strip()
#             if use_reasoning:
#                 text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
#             return text
#         except Exception as exc:
#             err = str(exc)
#             if "429" in err:
#                 wait = int(re.search(r"([\d.]+)s", err).group(1)) + 2 if re.search(r"([\d.]+)s", err) else 15
#                 time.sleep(wait)
#             elif "model_not_found" in err.lower():
#                 model_name = PRIMARY_MODEL
#             else:
#                 break
#     raise RuntimeError("Groq call failed. Last error: " + str(exc))

import os
import re
import time
import logging
import threading

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


_MODELS = {
    # name                       rpm_budget  output_tokens
    "gemini-2.5-flash-lite":  {"rpm": 8,  "max_out": 1024},
    "gemini-2.5-flash":       {"rpm": 4,  "max_out": 1024},
}

# Priority order: lite first 
_MODEL_PRIORITY = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

# ── Per-model rate-limit state (thread-safe) ─────────────────────────────────
# Tracks timestamps of recent calls so we never exceed self-imposed RPM.
_rate_state: dict = {
    name: {"lock": threading.Lock(), "call_times": []}
    for name in _MODEL_PRIORITY
}

_CONFIGURED = False



def _enforce_rpm(model_name: str) -> None:
    """
    Block until it is safe to fire a request for model_name,
    respecting its self-imposed RPM cap.

    Algorithm: sliding 60-second window.
      - Keep a list of recent call timestamps.
      - If len(recent) >= rpm_budget, sleep until the oldest
        call is >60 s old, then proceed.
    """
    cfg   = _MODELS[model_name]
    rpm   = cfg["rpm"]
    state = _rate_state[model_name]

    with state["lock"]:
        now   = time.monotonic()
        # Evict calls older than 60 s
        state["call_times"] = [t for t in state["call_times"] if now - t < 60.0]

        if len(state["call_times"]) >= rpm:
            # Must wait until the oldest call ages out
            oldest   = state["call_times"][0]
            wait_sec = 60.0 - (now - oldest) + 0.2   # +0.2 s buffer
            logger.info(
                "[RateLimiter] %s at %d/%d RPM — sleeping %.1fs",
                model_name, len(state["call_times"]), rpm, wait_sec,
            )
            time.sleep(max(wait_sec, 0))
            # Re-evict after sleep
            now = time.monotonic()
            state["call_times"] = [t for t in state["call_times"] if now - t < 60.0]

        state["call_times"].append(time.monotonic())


def _ensure_configured() -> None:
    global _CONFIGURED
    if not _CONFIGURED:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set.\n"
                "Add it to your .env file:  GEMINI_API_KEY=your_key_here\n"
                "Free key: https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=api_key)
        _CONFIGURED = True


def _parse_retry_secs(err_str: str) -> float:
    """Pull the suggested wait time out of a 429 error string."""
    # "Please retry in 19.829s"  or  "retry_delay { seconds: 19 }"
    m = re.search(r"retry in\s+([\d.]+)s", err_str, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 1.0
    m = re.search(r"seconds:\s*(\d+)", err_str)
    if m:
        return float(m.group(1)) + 1.0
    return 20.0   # safe default


def _is_daily_quota(err_str: str) -> bool:
    low = err_str.lower()
    return (
        "perday" in low
        or "per_day" in low
        or "GenerateRequestsPerDayPerProject" in err_str
        or ("day" in low and "requests" in low)
    )



def call_gemini(
    prompt: str,
    task_type: str = "default",
    temperature: float = 0.3,
    max_retries: int = 3,
) -> str:
    """
    Call Gemini and return the response text.

    task_type  — hint that selects max_output_tokens budget:
                 'analysis' | 'summary' | 'citations' | 'insights'
                 | 'review' | 'default'

    Model selection:
      • Uses gemini-2.5-flash-lite by default (8 RPM budget).
      • Falls back to gemini-2.5-flash (4 RPM budget) on 404 or
        daily-quota exhaustion of the lite model.

    Rate limiting:
      • Never exceeds self-imposed RPM cap per model.
      • On a 429, reads the suggested retry delay and obeys it.
    """
    _ensure_configured()

    # Task -> max output tokens (keeps responses tight, saves quota)
    token_budget = {
        "analysis":  900,
        "summary":   400,
        "citations": 800,
        "insights":  700,
        "review":    120,
        "default":   800,
    }.get(task_type, 800)

    # Try each model in priority order
    for model_name in _MODEL_PRIORITY:
        max_out = min(token_budget, _MODELS[model_name]["max_out"])
        gen_cfg = genai.types.GenerationConfig(
            max_output_tokens=max_out,
            temperature=temperature,
        )
        model = genai.GenerativeModel(model_name)
        last_err = None

        for attempt in range(1, max_retries + 1):
            # Respect self-imposed RPM before every call 
            _enforce_rpm(model_name)

            logger.info(
                "Gemini call — model=%s  task=%s  attempt=%d/%d",
                model_name, task_type, attempt, max_retries,
            )
            try:
                response = model.generate_content(prompt, generation_config=gen_cfg)
                text = response.text
                if not text or not text.strip():
                    logger.warning("Empty response (attempt %d), retrying…", attempt)
                    time.sleep(2)
                    continue
                logger.info("Gemini OK — model=%s  chars=%d", model_name, len(text))
                return text.strip()

            except Exception as exc:
                last_err = exc
                err_str   = str(exc)
                logger.warning(
                    "Gemini error — model=%s attempt=%d: %s",
                    model_name, attempt, exc,
                )

                # 404: this model is unavailable → try next in priority list
                if "404" in err_str or "not found" in err_str.lower():
                    logger.warning("Model %s not found. Trying next model.", model_name)
                    break   # break inner loop → outer loop picks next model

                # 429: quota / rate-limit
                if "429" in err_str or "quota" in err_str.lower():
                    if _is_daily_quota(err_str):
                        logger.warning(
                            "Daily quota hit on %s. Trying next model.", model_name
                        )
                        break  # move to next model
                    else:
                        # Per-minute throttle: wait exactly what Google says
                        wait = _parse_retry_secs(err_str)
                        logger.info(
                            "Per-minute rate limit on %s. "
                            "Google says wait %.1fs — obeying.",
                            model_name, wait,
                        )
                        time.sleep(wait)
                        # Don't count this as a real attempt
                        continue

                # 5xx transient server errors
                if any(code in err_str for code in ("500", "503", "502")):
                    backoff = 5 * attempt
                    logger.info("Server error — retrying in %ds", backoff)
                    time.sleep(backoff)
                    continue

                # Any other error (auth, bad request…) — stop immediately
                logger.error("Non-transient Gemini error: %s", exc)
                raise

        # If inner loop exhausted without returning, last_err is set
        if last_err and "404" not in str(last_err) and not _is_daily_quota(str(last_err)):
            # Not a "try-next-model" exit — propagate
            raise RuntimeError(
                f"Gemini call failed on {model_name} after {max_retries} attempts.\n"
                f"Last error: {last_err}"
            )

    # All models exhausted
    raise RuntimeError(
        "Gemini call failed on ALL models: {}\n\n"
        "Possible causes:\n"
        "  • Daily free-tier quota exhausted — wait until midnight Pacific Time\n"
        "  • API key invalid — check at https://aistudio.google.com/\n"
        "  • All models temporarily unavailable\n"
        "Last error: {}".format(_MODEL_PRIORITY, last_err)
    )
