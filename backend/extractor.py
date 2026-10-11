"""
extractor.py - Field Normalization, Validation, and Emergency Fallback Module.

Role:
  1. Validates and normalizes structured fields output by Gemini.
  2. Guarantees complete field compatibility with comparison.py and risk_engine.py.
  3. Provides an explicitly tagged fallback (fallback_rule_extraction) ONLY when Gemini is unavailable.
"""

import re
import unicodedata
from datetime import date
from typing import Any, Dict, List, Optional

# =====================================================================
# 1. NORMALIZATION & VALIDATION (FOR LLM OUTPUT)
# =====================================================================

def normalize_date_string(raw: Optional[str]) -> Optional[str]:
    """Convert valid date strings to canonical DD/MM/YYYY format."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()

    # Matches DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, etc.
    m_iso = re.match(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$", s)
    if m_iso:
        y, mo, d = map(int, m_iso.groups())
        try:
            return f"{d:02d}/{mo:02d}/{y}"
        except ValueError:
            return None

    m_dmy = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$", s)
    if m_dmy:
        d, mo, y = map(int, m_dmy.groups())
        if y < 100:
            y += 2000 if y <= 35 else 1900
        try:
            date(y, mo, d)
            return f"{d:02d}/{mo:02d}/{y}"
        except ValueError:
            return None

    return s


def normalize_clean_name(raw: Optional[str]) -> Optional[str]:
    """Normalize name casing while preserving multi-word tokens."""
    if not raw or not isinstance(raw, str):
        return None
    # Strip stray alphanumeric tags accidentally attached
    cleaned = re.sub(r"\b\d+[A-Za-z0-9\-]+\b", " ", raw)
    cleaned = re.sub(r"[^A-Za-z\u0900-\u0DFF .'\-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-'")
    if not cleaned or len(cleaned) < 2:
        return None
    return cleaned.title() if cleaned.isupper() else cleaned


def validate_and_normalize_extracted_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-processes and validates fields extracted by Gemini.
    Synchronizes aliases and sanitizes data for comparison.py.
    """
    normalized = dict(fields)

    # 1. Normalize Person Name
    if normalized.get("name"):
        normalized["name"] = normalize_clean_name(normalized["name"])

    # 2. Normalize Dates
    if normalized.get("dob"):
        normalized["dob"] = normalize_date_string(normalized["dob"])

    if normalized.get("all_dates") and isinstance(normalized["all_dates"], list):
        norm_dates = [normalize_date_string(d) for d in normalized["all_dates"]]
        normalized["all_dates"] = list(dict.fromkeys([d for d in norm_dates if d]))

    # 3. Synchronize Marks Aliases
    obtained = normalized.get("obtained_marks") or normalized.get("marks_obtained")
    total = normalized.get("total_marks") or normalized.get("max_marks")

    normalized["obtained_marks"] = obtained
    normalized["marks_obtained"] = obtained
    normalized["total_marks"] = total
    normalized["max_marks"] = total

    # 4. Marksheet Pincode Suppression
    # Marksheet seat/roll numbers should never be misclassified as postal pincodes
    if normalized.get("document_type") == "Marksheet":
        normalized["pincode"] = None

    # 5. Clean Pincode
    if normalized.get("pincode"):
        pin_match = re.search(r"\b[1-9]\d{5}\b", re.sub(r"\s+", "", str(normalized["pincode"])))
        normalized["pincode"] = pin_match.group(0) if pin_match else None

    # 6. Ensure Base Required Keys Exist
    standard_keys = [
        "document_type", "name", "father_name", "mother_name", "spouse_name", "relation",
        "dob", "age", "gender", "nationality", "address", "village", "taluka", "district",
        "state", "pincode", "phone", "email", "aadhaar_number", "aadhaar_masked", "pan_number",
        "voter_id", "driving_licence_number", "passport_number", "certificate_number",
        "application_number", "reference_number", "income", "income_in_words", "roll_number",
        "seat_number", "sid_number", "statement_number", "enrollment_number", "board",
        "institute", "exam_name", "obtained_marks", "total_marks", "percentage", "cgpa",
        "grade", "result", "exam_year", "invoice_number", "invoice_amount", "all_dates"
    ]
    for key in standard_keys:
        if key not in normalized:
            normalized[key] = None

    return normalized


# =====================================================================
# 2. EMERGENCY REGEX FALLBACK (ONLY IF GEMINI IS UNAVAILABLE)
# =====================================================================

def _build_digit_map():
    m = {}
    for cp in list(range(0x0660, 0x066A)) + list(range(0x0900, 0x0E00)):
        if unicodedata.category(chr(cp)) == "Nd":
            m[cp] = str(unicodedata.digit(chr(cp)))
    return m

_DIGIT_MAP = _build_digit_map()
_PUNCT_MAP = str.maketrans({
    "–": "-", "—": "-", "‐": "-", "‑": "-", "’": "'", "‘": "'",
    "“": '"', "”": '"', "\u200b": "", "\u00a0": " ", "\ufeff": ""
})

def _clean_text(text: str) -> str:
    text = (text or "").replace("\x00", " ").translate(_DIGIT_MAP).translate(_PUNCT_MAP)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

_AADHAAR_RE = re.compile(r"(?<!\d)(?<!\d[ \t])([2-9]\d{3})[ \t]?(\d{4})[ \t]?(\d{4})(?![ \t]?\d)")

def _extract_aadhaar_fallback(text: str) -> Dict[str, Any]:
    cands = ["".join(m.groups()) for m in _AADHAAR_RE.finditer(text)]
    pick = cands[0] if cands else None
    masked = re.search(r"(?:[Xx*]{4}[ \t]?){2}(\d{4})(?!\d)", text)
    return {
        "aadhaar_number": f"{pick[:4]} {pick[4:8]} {pick[8:]}" if pick else None,
        "aadhaar_masked": f"XXXX XXXX {masked.group(1)}" if masked else None,
    }

def _extract_ids_fallback(text: str) -> Dict[str, Optional[str]]:
    up = text.upper()
    pan = re.search(r"(?<![A-Z0-9])[A-Z]{3}[ABCFGHLJPTK][A-Z]\d{4}[A-Z](?![A-Z0-9])", up)
    voter = re.search(r"(?<![A-Z0-9])[A-Z]{3}\d{7}(?!\d)", up) if re.search(r"ELECTION|VOTER", up) else None
    return {
        "pan_number": pan.group(0) if pan else None,
        "voter_id": voter.group(0) if voter else None,
    }

def fallback_rule_extraction(ocr_text: str) -> Dict[str, Any]:
    """
    Emergency rule-based extraction used strictly when the LLM service is unavailable.
    Explicitly tags output with _extraction_source = 'fallback'.
    """
    raw = _clean_text(ocr_text)
    doc_type = "Unknown"
    low = raw.lower()

    if "aadhaar" in low or "uidai" in low:
        doc_type = "Aadhaar"
    elif "income tax department" in low or "permanent account" in low:
        doc_type = "PAN Card"
    elif "marksheet" in low or "statement of marks" in low or "higher secondary" in low:
        doc_type = "Marksheet"
    elif "income certificate" in low or "annual income" in low:
        doc_type = "Income Certificate"

    name = None
    m_name = re.search(r"(?:Candidate'?s?\s*Name|Name)\s*[,;:\-]\s*([A-Za-z\s]+)", raw, re.I)
    if m_name:
        name = normalize_clean_name(m_name.group(1))

    dob = None
    m_dob = re.search(r"(?:DOB|Date\s*of\s*Birth)\s*[:\-]?\s*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})", raw, re.I)
    if m_dob:
        dob = normalize_date_string(m_dob.group(1))

    income = None
    m_inc = re.search(r"(?:income|वार्षिक आय)[^\d\n]{0,30}(\d[\d,]+)", raw, re.I)
    if m_inc:
        income = m_inc.group(1).replace(",", "")

    result: Dict[str, Any] = {
        "document_type": doc_type,
        "name": name,
        "dob": dob,
        "gender": "Male" if re.search(r"\bmale\b", low) else ("Female" if re.search(r"\bfemale\b", low) else None),
        "income": income,
        **_extract_aadhaar_fallback(raw),
        **_extract_ids_fallback(raw),
        "_extraction_source": "fallback",
        "confidence": {"name": 0.5 if name else 0.0},
        "extraction_notes": ["Emergency rule-based fallback used because LLM was unavailable."]
    }

    return validate_and_normalize_extracted_fields(result)


# Backwards compatibility alias
extract_fields = fallback_rule_extraction