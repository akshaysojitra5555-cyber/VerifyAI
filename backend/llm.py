"""
llm.py - VerifyAI LLM explanation layer

The deterministic comparison.py and risk_engine.py already detect document
inconsistencies and calculate the verification-risk score.

This module is ONLY an explanation layer. The LLM:
    - explains the existing structured results in simple language
    - does NOT decide whether a document is fake
    - does NOT claim fraud
    - does NOT calculate or change the risk score
    - does NOT create new mismatches or invent information

Setup:
    pip install openai python-dotenv
    Put OPENAI_API_KEY=your_key_here in a .env file (see notes in the chat).

Test:
    python llm.py
"""

import json
import os
import re

# Load variables from a .env file (python-dotenv). If the package is missing we
# carry on, so the FastAPI server never crashes just because of this import.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Same idea for the OpenAI SDK: report a clean error at call time instead.
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-4o-mini"      # override with OPENAI_MODEL in .env
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 2                    # SDK-level retries (network / rate limit)
MAX_ATTEMPTS = 2                   # re-ask if the answer is unusable / unsafe
TEMPERATURE = 0.2                  # low = more faithful to the input

DISCLAIMER = ("This explanation is based on detected inconsistencies and "
              "supports human verification.")

# Words the explanation must never contain (safety net for the prompt rules).
FORBIDDEN_WORDS = re.compile(
    r"\b(fake|forged|forgery|fraud|fraudulent|fabricated|counterfeit)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are an explanation assistant for VerifyAI, a document verification support tool.

A deterministic comparison engine and a rule-based risk engine have ALREADY analysed the documents. Your only job is to explain their structured results in simple, clear language for a non-technical human reviewer.

STRICT RULES
1. Use ONLY facts present in the supplied JSON. Never invent documents, fields, values, mismatches, dates, names or reasons.
2. Never create a mismatch that is not in the input. Only describe a field as mismatched if its status is MISMATCH, as partially similar if PARTIAL_MATCH, as matched if MATCH, and as unavailable if NOT_AVAILABLE.
3. Never change, recalculate or question the risk score or risk level. Quote them exactly as supplied.
4. The risk score is a verification-priority score, NOT a probability that anything is wrong.
5. Never decide whether a document is genuine or not. Never call any document or person fake, forged, fabricated or fraudulent, and never state or imply that fraud occurred. Do not use the words "fake", "forged", "fraud" or "fraudulent" at all, not even in negated sentences. Use wording such as "potential inconsistency", "mismatch" and "needs manual verification".
6. NOT_AVAILABLE means the information could not be compared. It is not a problem by itself; mention it only when it is present in the input.
7. If the input shows no comparable fields (for example only one document, or everything unavailable), say there was not enough data to assess consistency.
8. Plain English, concise, no markdown, no bullet characters inside strings.

OUTPUT
Return ONE JSON object with exactly these keys and nothing else:
{
  "summary": "2-3 sentences summarising the verification result (how many documents were compared and the overall outcome).",
  "key_findings": ["3 to 6 short strings. Cover matched fields, mismatched fields (name the documents involved and briefly why that matters, e.g. name and date of birth identify the person so they should agree), partially similar fields, and unavailable fields when present."],
  "risk_explanation": "2-3 sentences stating the exact risk score (out of 100) and risk level supplied, and which detected inconsistencies the rule-based engine counted.",
  "verification_recommendation": "1-2 sentences recommending human verification, naming the fields and documents worth checking against originals."
}"""


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _build_messages(comparison_result, risk_result):
    """Create the chat messages sent to the model."""
    comparison_json = json.dumps(comparison_result, indent=2,
                                 ensure_ascii=False, default=str)
    risk_json = json.dumps(risk_result, indent=2,
                           ensure_ascii=False, default=str)
    user_prompt = (
        "Explain the following verification results.\n\n"
        f"COMPARISON_RESULT:\n{comparison_json}\n\n"
        f"RISK_RESULT:\n{risk_json}\n\n"
        "Respond with the JSON object only."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _parse_and_validate(content):
    """
    Turn the model's text into the final explanation dict.

    Returns (explanation, None) on success or (None, problem_text) on failure.
    """
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None, "the response was not valid JSON"

    if not isinstance(data, dict):
        return None, "the response was not a JSON object"

    summary = data.get("summary")
    risk_explanation = data.get("risk_explanation")
    recommendation = data.get("verification_recommendation")
    findings = data.get("key_findings")

    for name, value in (("summary", summary),
                        ("risk_explanation", risk_explanation),
                        ("verification_recommendation", recommendation)):
        if not isinstance(value, str) or not value.strip():
            return None, f"'{name}' was missing or empty"

    if not isinstance(findings, list):
        return None, "'key_findings' was not a list"
    findings = [str(item).strip() for item in findings if str(item).strip()]
    if not findings:
        return None, "'key_findings' was empty"

    # Safety net: reject definitive fake / fraud wording.
    all_text = " ".join([summary, risk_explanation, recommendation] + findings)
    if FORBIDDEN_WORDS.search(all_text):
        return None, "the response used wording that implies a fake/fraud verdict"

    return {
        "success": True,
        "summary": summary.strip(),
        "key_findings": findings,
        "risk_explanation": risk_explanation.strip(),
        "verification_recommendation": recommendation.strip(),
        "disclaimer": DISCLAIMER,  # fixed text, never taken from the LLM
    }, None


def _friendly_error(exc):
    """Map an exception to a safe message (never echoes the raw error text,
    because it can contain parts of the API key)."""
    messages = {
        "AuthenticationError": "OpenAI authentication failed. Check OPENAI_API_KEY.",
        "PermissionDeniedError": "OpenAI denied access for this API key or model.",
        "RateLimitError": "OpenAI rate limit or quota exceeded. Please try again later.",
        "APITimeoutError": "The OpenAI request timed out. Please try again.",
        "APIConnectionError": "Could not connect to OpenAI. Check the internet connection.",
        "NotFoundError": "The configured OpenAI model was not found. Check OPENAI_MODEL.",
        "InternalServerError": "OpenAI had a temporary server problem. Please try again.",
    }
    return messages.get(type(exc).__name__,
                        f"OpenAI request failed ({type(exc).__name__}).")


# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------

def generate_explanation(comparison_result, risk_result):
    """
    Ask the LLM to explain existing comparison + risk results.

    Success -> {"success": True, "summary", "key_findings", "risk_explanation",
                "verification_recommendation", "disclaimer"}
    Failure -> {"success": False, "error": "..."}   (never raises)
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return {"success": False, "error": "OPENAI_API_KEY is not configured"}

    if OpenAI is None:
        return {"success": False,
                "error": "The 'openai' package is not installed. Run: pip install openai"}

    if not isinstance(comparison_result, dict) or not isinstance(risk_result, dict):
        return {"success": False,
                "error": "comparison_result and risk_result must both be dictionaries"}

    try:
        client = OpenAI(api_key=api_key,
                        timeout=REQUEST_TIMEOUT_SECONDS,
                        max_retries=MAX_RETRIES)
        model = os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        messages = _build_messages(comparison_result, risk_result)

        last_problem = "no response received"
        for _ in range(MAX_ATTEMPTS):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            explanation, problem = _parse_and_validate(content)
            if explanation is not None:
                return explanation
            last_problem = problem

        return {"success": False,
                "error": f"The AI explanation could not be used: {last_problem}."}

    except Exception as exc:  # never crash the FastAPI server
        return {"success": False, "error": _friendly_error(exc)}


# ---------------------------------------------------------------------------
# TEST  (run with: python llm.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_comparison = {
        "documents_compared": 2,
        "pair_comparisons": [
            {
                "document_1": "Aadhaar",
                "document_2": "Marksheet",
                "field_comparisons": [
                    {"field": "name", "status": "MISMATCH", "similarity": 42},
                    {"field": "dob", "status": "MATCH", "similarity": 100},
                    {"field": "address", "status": "PARTIAL_MATCH", "similarity": 74},
                    {"field": "income", "status": "NOT_AVAILABLE"},
                ],
                "consistency_score": 54.0,
                "mismatch_count": 1,
                "partial_match_count": 1,
                "matched_field_count": 1,
                "not_available_count": 1,
            }
        ],
        "overall_consistency_score": 54.0,
        "total_matches": 1,
        "total_partial_matches": 1,
        "total_mismatches": 1,
        "total_not_available": 1,
    }

    sample_risk = {
        "risk_score": 37,
        "risk_level": "MEDIUM",
        "risk_indicators": [
            {
                "field": "name",
                "status": "MISMATCH",
                "severity": "HIGH",
                "points": 30,
                "document_1": "Aadhaar",
                "document_2": "Marksheet",
                "message": "Name mismatch detected between Aadhaar and Marksheet.",
            },
            {
                "field": "address",
                "status": "PARTIAL_MATCH",
                "severity": "LOW",
                "points": 7,
                "document_1": "Aadhaar",
                "document_2": "Marksheet",
                "message": ("Address shows partial similarity between Aadhaar and "
                            "Marksheet and may require verification."),
            },
        ],
        "statistics": {"matches": 1, "partial_matches": 1,
                       "mismatches": 1, "not_available": 1},
        "reasons": [
            "Name mismatch detected between Aadhaar and Marksheet.",
            "Address shows partial similarity between Aadhaar and Marksheet "
            "and may require verification.",
            "One field was unavailable for comparison.",
        ],
        "recommendation": ("Some inconsistencies were detected. "
                           "Manual verification is recommended."),
    }

    result = generate_explanation(sample_comparison, sample_risk)
    print(json.dumps(result, indent=2, ensure_ascii=False))
