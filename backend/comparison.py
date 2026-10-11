"""
comparison.py  --  VerifyAI multi-document comparison engine
=============================================================

Compares structured fields extracted from different identity / certificate
documents (Aadhaar, PAN, Marksheet, Income Certificate, ...) and produces a
clean, LLM-ready description of how consistent they are.

Design rules
------------
* No plain `==` on raw strings: every value is normalized first.
* Fuzzy matching (RapidFuzz) is used ONLY for name and address.
* DOB, gender, pincode and income use exact comparison on normalized values.
* Only fields that make sense for BOTH document types are compared
  (see DOCUMENT_COMPARISON_RULES).
* This module never claims a document is fake. It only reports
  MATCH / PARTIAL_MATCH / MISMATCH / NOT_AVAILABLE and the phrase
  "Potential inconsistency detected" for anything that is not a clean match.
* No LLM is used here. build_llm_comparison_context() prepares the data
  for a future LLM step.

Install dependency:  pip install rapidfuzz
Run self-test:       python comparison.py
"""

import json
import re
from datetime import date, datetime
from itertools import combinations
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Status values
MATCH = "MATCH"
PARTIAL_MATCH = "PARTIAL_MATCH"
MISMATCH = "MISMATCH"
NOT_AVAILABLE = "NOT_AVAILABLE"

# Which fields matter for each document type.
DOCUMENT_COMPARISON_RULES: Dict[str, List[str]] = {
    "Aadhaar": ["name", "dob", "gender", "address", "pincode"],
    "PAN Card": ["name", "dob"],
    "Voter ID": ["name", "dob", "address", "pincode"],
    "Driving Licence": ["name", "dob", "address"],
    "Passport": ["name", "dob", "gender"],
    "Income Certificate": ["name", "address", "pincode", "income"],
    "Caste Certificate": ["name", "dob", "address", "pincode"],
    "Domicile Certificate": ["name", "dob", "address", "pincode"],
    "Birth Certificate": ["name", "dob"],
    "Marksheet": ["name", "dob"],
    "Bank Statement": ["name", "address"],
    "Ration Card": ["name", "address", "pincode"],
    "Scholarship Application": ["name", "dob", "address", "pincode", "income"],
    "Invoice": ["name", "address"],
}

# Weights used ONLY for the consistency score (NOT a fraud probability).
# Configurable: edit here, or pass `weights=` to the compare functions.
FIELD_WEIGHTS: Dict[str, float] = {
    "name": 0.35,
    "dob": 0.25,
    "address": 0.15,
    "pincode": 0.10,
    "gender": 0.05,
    "income": 0.10,
}

# Fuzzy thresholds (inclusive lower bounds for MATCH / PARTIAL_MATCH)
NAME_MATCH_THRESHOLD = 90.0
NAME_PARTIAL_THRESHOLD = 70.0
ADDRESS_MATCH_THRESHOLD = 85.0
ADDRESS_PARTIAL_THRESHOLD = 65.0

# When one address is a token-subset of the other, RapidFuzz's
# token_set_ratio says 100. That is too generous, so it is scaled down.
ADDRESS_SUBSET_FACTOR = 0.90

# Income difference (percent) tolerated before status becomes MISMATCH.
# 0.0 = must be identical, as requested. Raise it if you want some slack.
INCOME_TOLERANCE_PERCENT = 0.0

INCONSISTENCY_MESSAGE = "Potential inconsistency detected"

# ---------------------------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------------------------


def _is_blank(value: Any) -> bool:
    """True for None, empty / whitespace-only strings and common OCR 'null' words."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null", "n/a", "na", "nan", "-"}
    return False


def _result(
    field: str,
    value_1: Any,
    value_2: Any,
    similarity: Optional[float],
    status: str,
    **extra: Any,
) -> Dict[str, Any]:
    """Build the standard field-comparison dictionary."""
    out: Dict[str, Any] = {
        "field": field,
        "value_1": value_1,
        "value_2": value_2,
        "similarity": None if similarity is None else round(float(similarity), 1),
        "status": status,
    }
    out.update(extra)
    return out


def _status_from_score(score: float, match_t: float, partial_t: float) -> str:
    score = round(score, 1)  # classify on the same value that is displayed
    if score >= match_t:
        return MATCH
    if score >= partial_t:
        return PARTIAL_MATCH
    return MISMATCH


# ---------------------------------------------------------------------------
# 1. NORMALIZATION
# ---------------------------------------------------------------------------

# Honorifics carry no identity information. ("kumar"/"kumari" are deliberately
# NOT here: they are very common real name parts in India.)
_HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "shri", "shree", "sri", "smt", "shrimati"}


def normalize_name(value: Any) -> Optional[str]:
    """
    Lowercase, strip punctuation / digits / honorifics, collapse whitespace.
    Word order is preserved here; order-independence is handled by RapidFuzz's
    token_sort_ratio / token_set_ratio in compare_name().
    Returns None if nothing meaningful is left.
    """
    if _is_blank(value):
        return None
    text = str(value).casefold()
    text = re.sub(r"[\'’`]", "", text)          # o'brien -> obrien
    text = re.sub(r"[^\w\s]|_", " ", text)      # . , - / | etc. -> space
    text = re.sub(r"\d+", " ", text)            # digits in a name = OCR noise
    tokens = [t for t in text.split() if t not in _HONORIFICS]
    return " ".join(tokens) or None


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _safe_date(year: int, month: int, day: int) -> Optional[str]:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def normalize_dob(value: Any) -> Optional[str]:
    """
    Convert many date formats into ISO 'YYYY-MM-DD'.
    Handles 02/12/2006, 02-12-2006, 02.12.2006, 2006-12-02, 02 Dec 2006,
    02122006 and 2-digit years. Indian day-first order is assumed.
    Returns None if the value cannot be parsed into a valid date.
    """
    if _is_blank(value):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    text = str(value).strip()

    # Numeric with separators:  dd/mm/yyyy  dd-mm-yyyy  dd.mm.yyyy  yyyy-mm-dd
    m = re.search(r"(\d{1,4})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{1,4})", text)
    if m:
        a, b, c = m.group(1), m.group(2), m.group(3)
        if len(a) == 4:                              # yyyy-mm-dd
            return _safe_date(int(a), int(b), int(c))
        day, month = int(a), int(b)
        if len(c) == 4:
            year = int(c)
        elif len(c) == 2:                            # 2-digit year
            yy = int(c)
            year = 2000 + yy if yy <= date.today().year % 100 else 1900 + yy
        else:
            return None
        iso = _safe_date(year, month, day)
        if iso is None and month > 12 and day <= 12:  # looks like mm/dd/yyyy
            iso = _safe_date(year, day, month)
        return iso

    # Month name:  02 Dec 2006 / 2nd December, 2006
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?[\s\-/.,]*([A-Za-z]{3,9})[\s\-/.,]*(\d{4})", text)
    if m:
        word = m.group(2).lower()
        mon = _MONTHS.get(word) or _MONTHS.get(word[:3])
        if mon:
            return _safe_date(int(m.group(3)), mon, int(m.group(1)))

    # Compact ddmmyyyy
    m = re.fullmatch(r"(\d{2})(\d{2})(\d{4})", text)
    if m:
        return _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))

    return None


# Only unambiguous abbreviations are expanded.
_ADDRESS_ABBREVIATIONS = {
    "rd": "road", "st": "street", "nr": "near", "opp": "opposite",
    "apt": "apartment", "bldg": "building", "soc": "society",
    "dist": "district", "distt": "district", "tal": "taluka",
    "ave": "avenue", "blk": "block", "flr": "floor", "po": "post",
}


def normalize_address(value: Any) -> Optional[str]:
    """
    Lowercase, remove punctuation, collapse whitespace, expand safe
    abbreviations, and drop 6-digit pincodes (pincode is compared separately,
    so it should not be double counted in the address score).
    """
    if _is_blank(value):
        return None
    text = str(value).casefold()
    text = re.sub(r"[^\w\s]|_", " ", text)
    text = re.sub(r"\b[1-9]\d{5}\b", " ", text)  # remove pincode
    tokens = [_ADDRESS_ABBREVIATIONS.get(t, t) for t in text.split()]
    return " ".join(tokens) or None


def normalize_pincode(value: Any) -> Optional[str]:
    """Return a clean 6-digit Indian pincode string, or None."""
    if _is_blank(value):
        return None
    text = str(value).strip()
    compact = re.sub(r"[\s\-]", "", text)            # "365 220" -> "365220"
    if re.fullmatch(r"[1-9]\d{5}", compact):
        return compact
    m = re.search(r"(?<!\d)[1-9]\d{5}(?!\d)", text)  # pincode inside longer text
    return m.group(0) if m else None


_INCOME_MULTIPLIERS = [
    (r"\b(crore|crores|cr)\b", 1e7),
    (r"\b(lakh|lakhs|lac|lacs|lpa)\b", 1e5),
    (r"\b(thousand|k)\b", 1e3),
]


def normalize_income(value: Any) -> Optional[float]:
    """
    Convert income to a float (rupees).
    Handles 150000, "Rs. 1,50,000/-", "₹1.5 lakh", "2 lakhs", "1 crore".
    Returns None if no number can be found.
    """
    if _is_blank(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None

    text = str(value).casefold()
    m = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    if not m:
        return None
    try:
        number = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    for pattern, factor in _INCOME_MULTIPLIERS:
        if re.search(pattern, text[m.end():]) or re.search(pattern, text[:m.start()]):
            number *= factor
            break
    return number


_GENDER_MAP = {
    "male": "male", "m": "male", "man": "male", "boy": "male",
    "purush": "male", "पुरुष": "male",
    "female": "female", "f": "female", "woman": "female", "girl": "female",
    "mahila": "female", "महिला": "female", "स्त्री": "female",
    "transgender": "other", "other": "other", "others": "other",
    "tg": "other", "t": "other",
}


def normalize_gender(value: Any) -> Optional[str]:
    """Map OCR variants (M, MALE, Male / पुरुष, F, ...) to male / female / other."""
    if _is_blank(value):
        return None
    for token in re.findall(r"\w+", str(value).casefold()):
        if token in _GENDER_MAP:
            return _GENDER_MAP[token]
    return None


# ---------------------------------------------------------------------------
# 2. FIELD COMPARISON
# ---------------------------------------------------------------------------


def _unparseable_note(raw_1: Any, raw_2: Any, n1: Any, n2: Any) -> str:
    bad = []
    if _is_blank(raw_1):
        bad.append("value_1 missing")
    elif n1 is None:
        bad.append("value_1 could not be normalized")
    if _is_blank(raw_2):
        bad.append("value_2 missing")
    elif n2 is None:
        bad.append("value_2 could not be normalized")
    return "; ".join(bad)



def _token_coverage_score(n1: str, n2: str) -> float:
    """
    Token-level name score for names of different length (missing middle /
    surname, initials, shortened forms like 'Suresh' vs 'Sureshbhai').
    Each token of the shorter name is matched to its best token in the longer
    name; the average is then scaled down by how much of the longer name is
    actually covered, so partial names can reach PARTIAL_MATCH but not MATCH.
    """
    t1, t2 = n1.split(), n2.split()
    short, long_ = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
    scores = []
    for s in short:
        best = 0.0
        for l in long_:
            if s == l:
                sc = 100.0
            elif len(s) == 1 or len(l) == 1:            # initial: "s" vs "suresh"
                sc = 85.0 if s[0] == l[0] else 0.0
            elif l.startswith(s) or s.startswith(l):    # "suresh" vs "sureshbhai"
                sc = 90.0
            else:
                sc = fuzz.ratio(s, l)
            best = max(best, sc)
        scores.append(best)
    mean = sum(scores) / len(scores)
    coverage = len(short) / len(long_)
    return mean * (0.7 + 0.3 * coverage)


def compare_name(value_1: Any, value_2: Any) -> Dict[str, Any]:
    """Fuzzy, word-order-independent name comparison."""
    n1, n2 = normalize_name(value_1), normalize_name(value_2)
    if n1 is None or n2 is None:
        return _result("name", value_1, value_2, None, NOT_AVAILABLE,
                       note=_unparseable_note(value_1, value_2, n1, n2))
    # token_sort_ratio handles reordered names and OCR typos; the token
    # coverage score handles missing / shortened / initial-only name parts.
    score = max(fuzz.token_sort_ratio(n1, n2), _token_coverage_score(n1, n2))
    status = _status_from_score(score, NAME_MATCH_THRESHOLD, NAME_PARTIAL_THRESHOLD)
    return _result("name", value_1, value_2, score, status,
                   normalized_1=n1, normalized_2=n2)


def compare_dob(value_1: Any, value_2: Any) -> Dict[str, Any]:
    """Exact comparison of normalized dates (no fuzzy matching)."""
    d1, d2 = normalize_dob(value_1), normalize_dob(value_2)
    if d1 is None or d2 is None:
        return _result("dob", value_1, value_2, None, NOT_AVAILABLE,
                       note=_unparseable_note(value_1, value_2, d1, d2))
    same = d1 == d2
    return _result("dob", value_1, value_2, 100.0 if same else 0.0,
                   MATCH if same else MISMATCH,
                   normalized_1=d1, normalized_2=d2)


def compare_gender(value_1: Any, value_2: Any) -> Dict[str, Any]:
    g1, g2 = normalize_gender(value_1), normalize_gender(value_2)
    if g1 is None or g2 is None:
        return _result("gender", value_1, value_2, None, NOT_AVAILABLE,
                       note=_unparseable_note(value_1, value_2, g1, g2))
    same = g1 == g2
    return _result("gender", value_1, value_2, 100.0 if same else 0.0,
                   MATCH if same else MISMATCH,
                   normalized_1=g1, normalized_2=g2)


def compare_address(value_1: Any, value_2: Any) -> Dict[str, Any]:
    """Fuzzy address comparison on normalized text."""
    a1, a2 = normalize_address(value_1), normalize_address(value_2)
    if a1 is None or a2 is None:
        return _result("address", value_1, value_2, None, NOT_AVAILABLE,
                       note=_unparseable_note(value_1, value_2, a1, a2))
    score = max(
        fuzz.token_sort_ratio(a1, a2),
        fuzz.token_set_ratio(a1, a2) * ADDRESS_SUBSET_FACTOR,
    )
    status = _status_from_score(score, ADDRESS_MATCH_THRESHOLD, ADDRESS_PARTIAL_THRESHOLD)
    return _result("address", value_1, value_2, score, status,
                   normalized_1=a1, normalized_2=a2)


def compare_pincode(value_1: Any, value_2: Any) -> Dict[str, Any]:
    p1, p2 = normalize_pincode(value_1), normalize_pincode(value_2)
    if p1 is None or p2 is None:
        return _result("pincode", value_1, value_2, None, NOT_AVAILABLE,
                       note=_unparseable_note(value_1, value_2, p1, p2))
    same = p1 == p2
    return _result("pincode", value_1, value_2, 100.0 if same else 0.0,
                   MATCH if same else MISMATCH,
                   normalized_1=p1, normalized_2=p2)


def compare_income(value_1: Any, value_2: Any) -> Dict[str, Any]:
    """
    Numeric income comparison. Equal -> MATCH. Different -> MISMATCH plus the
    absolute and percentage difference (relative to the larger value, so the
    result is symmetric). A difference is NOT automatically fraud; the risk
    engine / LLM decides how to interpret it.
    """
    i1, i2 = normalize_income(value_1), normalize_income(value_2)
    if i1 is None or i2 is None:
        return _result("income", value_1, value_2, None, NOT_AVAILABLE,
                       note=_unparseable_note(value_1, value_2, i1, i2))
    larger = max(i1, i2)
    abs_diff = abs(i1 - i2)
    pct_diff = 0.0 if larger == 0 else abs_diff / larger * 100.0
    status = MATCH if pct_diff <= INCOME_TOLERANCE_PERCENT else MISMATCH
    similarity = max(0.0, 100.0 - pct_diff)
    return _result("income", value_1, value_2, similarity, status,
                   normalized_1=i1, normalized_2=i2,
                   difference_amount=round(abs_diff, 2),
                   difference_percent=round(pct_diff, 2))


FIELD_COMPARATORS = {
    "name": compare_name,
    "dob": compare_dob,
    "gender": compare_gender,
    "address": compare_address,
    "pincode": compare_pincode,
    "income": compare_income,
}

# ---------------------------------------------------------------------------
# 5. PAIRWISE DOCUMENT COMPARISON
# ---------------------------------------------------------------------------

# Case-insensitive lookup so "aadhaar" / "AADHAAR" / "Aadhaar" all work.
_CANONICAL_TYPES = {k.casefold(): k for k in DOCUMENT_COMPARISON_RULES}


def _canonical_doc_type(value: Any) -> Optional[str]:
    if _is_blank(value):
        return None
    return _CANONICAL_TYPES.get(str(value).strip().casefold())


def _weighted_consistency(
    comparisons: List[Dict[str, Any]], weights: Dict[str, float]
) -> Optional[float]:
    """
    Weighted average of field similarities (0-100). Fields that are
    NOT_AVAILABLE are left out and the remaining weights are re-normalized,
    so missing data neither helps nor hurts the score.
    Returns None when nothing could be compared.
    """
    total_weight = 0.0
    accumulated = 0.0
    for c in comparisons:
        if c["status"] == NOT_AVAILABLE or c["similarity"] is None:
            continue
        w = weights.get(c["field"], 0.0)
        if w <= 0:
            continue
        accumulated += w * c["similarity"]
        total_weight += w
    if total_weight == 0:
        return None
    return round(accumulated / total_weight, 1)


def _count(comparisons: List[Dict[str, Any]], status: str) -> int:
    return sum(1 for c in comparisons if c["status"] == status)


def compare_two_documents(
    doc1: Dict[str, Any],
    doc2: Dict[str, Any],
    rules: Optional[Dict[str, List[str]]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compare two extracted-document dictionaries, using only the fields that
    BOTH document types care about (intersection of their rule lists).
    """
    rules = rules or DOCUMENT_COMPARISON_RULES
    weights = weights or FIELD_WEIGHTS

    type_1 = _canonical_doc_type(doc1.get("document_type")) or doc1.get("document_type")
    type_2 = _canonical_doc_type(doc2.get("document_type")) or doc2.get("document_type")

    fields_1 = rules.get(type_1, []) if type_1 else []
    fields_2 = rules.get(type_2, []) if type_2 else []
    common_fields = [f for f in fields_1 if f in fields_2]  # keeps rule order

    field_comparisons: List[Dict[str, Any]] = []
    for field in common_fields:
        comparator = FIELD_COMPARATORS.get(field)
        if comparator is None:
            continue
        field_comparisons.append(comparator(doc1.get(field), doc2.get(field)))

    note = None
    if not common_fields:
        note = "No comparable fields for this document-type combination."

    return {
        "document_1": type_1,
        "document_2": type_2,
        "filename_1": doc1.get("filename"),
        "filename_2": doc2.get("filename"),
        "comparable_fields": common_fields,
        "field_comparisons": field_comparisons,
        "consistency_score": _weighted_consistency(field_comparisons, weights),
        "mismatch_count": _count(field_comparisons, MISMATCH),
        "partial_match_count": _count(field_comparisons, PARTIAL_MATCH),
        "matched_field_count": _count(field_comparisons, MATCH),
        "not_available_count": _count(field_comparisons, NOT_AVAILABLE),
        "note": note,
    }


# ---------------------------------------------------------------------------
# 6. MULTI-DOCUMENT COMPARISON
# ---------------------------------------------------------------------------


def compare_all_documents(
    documents: List[Dict[str, Any]],
    rules: Optional[Dict[str, List[str]]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compare every unique pair of documents (never a document with itself).
    Pairs that share no comparable fields (e.g. PAN vs Invoice) are skipped.
    The overall score pools every available field comparison across all pairs.
    """
    weights = weights or FIELD_WEIGHTS
    documents = [d for d in (documents or []) if isinstance(d, dict)]

    pair_comparisons: List[Dict[str, Any]] = []
    skipped_pairs = 0
    for d1, d2 in combinations(documents, 2):   # unique pairs, no self-pairs
        pair = compare_two_documents(d1, d2, rules=rules, weights=weights)
        if not pair["field_comparisons"]:
            skipped_pairs += 1
            continue
        pair_comparisons.append(pair)

    all_fields = [fc for p in pair_comparisons for fc in p["field_comparisons"]]

    return {
        "documents_compared": len(documents),
        "pair_comparisons": pair_comparisons,
        "pairs_skipped_no_common_fields": skipped_pairs,
        "overall_consistency_score": _weighted_consistency(all_fields, weights),
        "total_matches": _count(all_fields, MATCH),
        "total_partial_matches": _count(all_fields, PARTIAL_MATCH),
        "total_mismatches": _count(all_fields, MISMATCH),
        "total_not_available": _count(all_fields, NOT_AVAILABLE),
    }


# ---------------------------------------------------------------------------
# 8. LLM-READY OUTPUT
# ---------------------------------------------------------------------------


def build_llm_comparison_context(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten compare_all_documents() output into a compact, JSON-serializable
    structure for a future LLM. Contains no authenticity verdicts: anything
    other than MATCH is labelled "Potential inconsistency detected".
    """
    comparisons: List[Dict[str, Any]] = []
    for pair in results.get("pair_comparisons", []):
        for fc in pair.get("field_comparisons", []):
            entry: Dict[str, Any] = {
                "documents": [pair.get("document_1"), pair.get("document_2")],
                "filenames": [pair.get("filename_1"), pair.get("filename_2")],
                "field": fc["field"],
                "value_1": fc["value_1"],
                "value_2": fc["value_2"],
                "similarity": fc["similarity"],
                "status": fc["status"],
            }
            if fc["field"] == "income" and "difference_percent" in fc:
                entry["difference_amount"] = fc["difference_amount"]
                entry["difference_percent"] = fc["difference_percent"]
            if fc["status"] in (PARTIAL_MATCH, MISMATCH):
                entry["flag"] = INCONSISTENCY_MESSAGE
            if fc.get("note"):
                entry["note"] = fc["note"]
            comparisons.append(entry)

    return {
        "summary": {
            "documents_compared": results.get("documents_compared", 0),
            "overall_consistency_score": results.get("overall_consistency_score"),
            "matches": results.get("total_matches", 0),
            "partial_matches": results.get("total_partial_matches", 0),
            "mismatches": results.get("total_mismatches", 0),
            "not_available": results.get("total_not_available", 0),
        },
        "comparisons": comparisons,
        "flagged_comparisons": [c for c in comparisons if "flag" in c],
        "notice": (
            "These results only report field-level inconsistencies between "
            "documents. They are not a determination that any document is "
            "fake or fraudulent. Differences can come from OCR errors, "
            "spelling variants or legitimate changes over time."
        ),
    }


# ---------------------------------------------------------------------------
# 10. SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    aadhaar = {
        "filename": "aadhaar.pdf",
        "document_type": "Aadhaar",
        "name": "Sojitra Akshay Sureshbhai",
        "dob": "02/12/2006",
        "gender": "Male",
        "address": "Shakhpur, Amreli, Gujarat",
        "pincode": "365220",
        "income": None,
    }
    marksheet = {
        "filename": "marksheet.pdf",
        "document_type": "Marksheet",
        "name": "Akshay Sureshbhai Sojitra",
        "dob": "02-12-2006",
        "gender": None,
        "address": None,
        "pincode": None,
        "income": None,
    }
    income_cert = {
        "filename": "income.pdf",
        "document_type": "Income Certificate",
        "name": "Akshay Suresh",                       # shortened name -> partial
        "dob": None,
        "gender": None,
        "address": "Shakhpur, Tal. Amreli, Gujarat 365220",
        "pincode": "365220",
        "income": "Rs. 1,50,000/-",
    }
    scholarship = {
        "filename": "scholarship.pdf",
        "document_type": "Scholarship Application",
        "name": "SOJITRA AKSHAY SURESHBHAI",
        "dob": "02.12.2006",
        "gender": None,
        "address": "Shakhpur Amreli Gujarat",
        "pincode": "365 220",
        "income": "2.4 lakh",                          # differs -> income diff
    }

    def show(title: str, data: Any) -> None:
        print(f"\n===== {title} =====")
        print(json.dumps(data, indent=2, ensure_ascii=False))

    show("NAME COMPARISON", compare_name(aadhaar["name"], marksheet["name"]))
    show("DOB COMPARISON", compare_dob(aadhaar["dob"], marksheet["dob"]))
    show("COMPLETE DOCUMENT COMPARISON (Aadhaar vs Marksheet)",
         compare_two_documents(aadhaar, marksheet))

    all_results = compare_all_documents([aadhaar, marksheet, income_cert, scholarship])
    print("\n===== MULTI-DOCUMENT SUMMARY =====")
    print(json.dumps(
        {k: v for k, v in all_results.items() if k != "pair_comparisons"},
        indent=2))

    show("LLM CONTEXT", build_llm_comparison_context(all_results))
