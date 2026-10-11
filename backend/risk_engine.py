"""
risk_engine.py - VerifyAI rule-based risk / verification engine

Converts the output of comparison.py into:
    1. a risk score (0-100)
    2. a risk level (LOW / MEDIUM / HIGH)
    3. structured risk indicators
    4. human-readable reasons
    5. comparison statistics

IMPORTANT
---------
The score is a *verification-priority* score. It is NOT a probability that a
document is fake and it never claims fraud. It only highlights potential
inconsistencies and recommends manual verification.

How the score is calculated (fully transparent, no ML / LLM)
------------------------------------------------------------
1. Every field comparison with status MISMATCH or PARTIAL_MATCH earns points
   from the tables below. MATCH and NOT_AVAILABLE always earn 0 points.
2. Repeated issue handling (duplicate penalty):
   The same issue = same field + same status (e.g. "name mismatch") seen in
   several document pairs.
       - 1st occurrence       -> full points
       - each repeat          -> REPEAT_FACTOR x base points (default 25%)
       - total for that issue -> capped at REPEAT_CAP_MULTIPLIER x base
                                 points (default 1.5x)
   Example: name mismatch in 4 pairs = 30 + 7.5 + 7.5 + 0 = 45 (not 120).
3. All points are added and clamped to MAX_SCORE (100).
4. The score is mapped to a level using the threshold constants.

Only standard Python libraries are used.
Run the built-in tests with:  python risk_engine.py
"""

import json
import math

# ---------------------------------------------------------------------------
# CONFIGURATION (edit these to tune the engine)
# ---------------------------------------------------------------------------

# Points added for a MISMATCH, per field
MISMATCH_POINTS = {
    "name": 30,
    "dob": 40,
    "address": 15,
    "pincode": 15,
    "gender": 10,
    "income": 10,
}
DEFAULT_MISMATCH_POINTS = 10  # any other / unknown field

# Points added for a PARTIAL_MATCH, per field
PARTIAL_POINTS = {
    "name": 10,
    "address": 7,
}
DEFAULT_PARTIAL_POINTS = 5  # any other / unknown field

# Duplicate penalty settings
REPEAT_FACTOR = 0.25         # a repeat of the same issue adds 25% of base points
REPEAT_CAP_MULTIPLIER = 1.5  # one issue can never exceed 1.5x its base points

# Score range
MIN_SCORE = 0
MAX_SCORE = 100

# Risk level thresholds
#   0  - 24  -> LOW
#   25 - 59  -> MEDIUM
#   60 - 100 -> HIGH
MEDIUM_THRESHOLD = 25
HIGH_THRESHOLD = 60

# Severity of a single indicator is based on its base points
SEVERITY_HIGH_MIN = 25    # base points >= 25 -> HIGH
SEVERITY_MEDIUM_MIN = 10  # base points >= 10 -> MEDIUM, otherwise LOW

# Recommendations
RECOMMENDATIONS = {
    "LOW": "Documents show no significant detected inconsistencies.",
    "MEDIUM": "Some inconsistencies were detected. Manual verification is recommended.",
    "HIGH": ("Multiple significant inconsistencies were detected. "
             "Manual verification is strongly recommended."),
}
INSUFFICIENT_DATA_RECOMMENDATION = (
    "Not enough comparable data was available to assess consistency. "
    "Manual verification is recommended."
)

DISCLAIMER = (
    "This is a verification-priority score based on detected inconsistencies. "
    "It is not a probability that any document is fraudulent."
)

# Field name aliases -> canonical field names (so "date_of_birth" == "dob")
FIELD_ALIASES = {
    "full_name": "name",
    "applicant_name": "name",
    "date_of_birth": "dob",
    "birth_date": "dob",
    "birthdate": "dob",
    "pin": "pincode",
    "pin_code": "pincode",
    "postal_code": "pincode",
    "zip": "pincode",
    "zip_code": "pincode",
    "sex": "gender",
    "annual_income": "income",
    "family_income": "income",
}

# Pretty labels used in messages
FIELD_LABELS = {
    "name": "Name",
    "dob": "DOB",
    "address": "Address",
    "pincode": "Pincode",
    "gender": "Gender",
    "income": "Income",
}

# Status aliases -> canonical statuses
STATUS_ALIASES = {
    "MATCH": "MATCH",
    "PARTIAL_MATCH": "PARTIAL_MATCH",
    "PARTIAL": "PARTIAL_MATCH",
    "MISMATCH": "MISMATCH",
    "NOT_AVAILABLE": "NOT_AVAILABLE",
    "NA": "NOT_AVAILABLE",
    "N/A": "NOT_AVAILABLE",
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _normalize_field(raw_field):
    """Return a canonical lowercase field key, or None if unusable."""
    if raw_field is None:
        return None
    key = str(raw_field).strip().lower().replace("-", "_").replace(" ", "_")
    if not key:
        return None
    return FIELD_ALIASES.get(key, key)


def _field_label(field):
    return FIELD_LABELS.get(field, field.replace("_", " ").title())


def _normalize_status(raw_status):
    """Return MATCH / PARTIAL_MATCH / MISMATCH / NOT_AVAILABLE, or None."""
    if raw_status is None:
        return None
    key = str(raw_status).strip().upper().replace(" ", "_").replace("-", "_")
    return STATUS_ALIASES.get(key)


def _doc_name(raw_name):
    if raw_name is None or not str(raw_name).strip():
        return "Unknown document"
    return str(raw_name).strip()


def _severity(base_points):
    if base_points >= SEVERITY_HIGH_MIN:
        return "HIGH"
    if base_points >= SEVERITY_MEDIUM_MIN:
        return "MEDIUM"
    return "LOW"


def _clean_number(value):
    """Round to 2 decimals and drop '.0' so 30.0 is shown as 30."""
    value = round(value, 2)
    return int(value) if value == int(value) else value


def _risk_level(score):
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    if score >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _empty_stats():
    return {"matches": 0, "partial_matches": 0, "mismatches": 0, "not_available": 0}


# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------

def calculate_risk(comparison_results):
    """
    Convert comparison.py results into a risk assessment.

    Never raises on missing / malformed data; problems are reported in the
    "warnings" list of the result instead.
    """
    warnings = []
    stats = _empty_stats()
    indicators = []

    # (field, status) -> {"count": int, "points": float}
    issue_groups = {}
    raw_total = 0.0

    # ---- validate top-level structure ------------------------------------
    pairs = []
    documents_compared = None
    if not isinstance(comparison_results, dict):
        warnings.append("Comparison results were missing or not a valid object.")
    else:
        documents_compared = comparison_results.get("documents_compared")
        raw_pairs = comparison_results.get("pair_comparisons")
        if raw_pairs is None:
            pairs = []
        elif isinstance(raw_pairs, list):
            pairs = raw_pairs
        else:
            warnings.append("'pair_comparisons' was not a list and was ignored.")

    # ---- process every pair and every field ------------------------------
    for pair_number, pair in enumerate(pairs, start=1):
        if not isinstance(pair, dict):
            warnings.append(f"Pair #{pair_number} skipped: not a valid object.")
            continue

        doc1 = _doc_name(pair.get("document_1"))
        doc2 = _doc_name(pair.get("document_2"))

        field_comparisons = pair.get("field_comparisons")
        if field_comparisons is None:
            field_comparisons = []
        if not isinstance(field_comparisons, list):
            warnings.append(f"Pair #{pair_number} ({doc1} vs {doc2}): "
                            "'field_comparisons' was not a list.")
            continue

        for item in field_comparisons:
            if not isinstance(item, dict):
                warnings.append(f"A field comparison in pair #{pair_number} "
                                "was skipped: not a valid object.")
                continue

            field = _normalize_field(item.get("field"))
            status = _normalize_status(item.get("status"))

            if field is None:
                warnings.append(f"A field comparison in pair #{pair_number} "
                                "was skipped: missing field name.")
                continue
            if status is None:
                warnings.append(f"Field '{field}' in pair #{pair_number} was skipped: "
                                f"unknown status {item.get('status')!r}.")
                continue

            # ---- statistics ----------------------------------------------
            if status == "MATCH":
                stats["matches"] += 1
                continue                      # MATCH = zero risk
            if status == "NOT_AVAILABLE":
                stats["not_available"] += 1
                continue                      # NOT_AVAILABLE = zero risk

            label = _field_label(field)

            if status == "MISMATCH":
                stats["mismatches"] += 1
                base = MISMATCH_POINTS.get(field, DEFAULT_MISMATCH_POINTS)
                message = f"{label} mismatch detected between {doc1} and {doc2}."
            else:  # PARTIAL_MATCH
                stats["partial_matches"] += 1
                base = PARTIAL_POINTS.get(field, DEFAULT_PARTIAL_POINTS)
                message = (f"{label} shows partial similarity between {doc1} and "
                           f"{doc2} and may require verification.")

            # ---- duplicate penalty ---------------------------------------
            group = issue_groups.setdefault((field, status), {"count": 0, "points": 0.0})
            group["count"] += 1

            if group["count"] == 1:
                points = float(base)
            else:
                cap_left = base * REPEAT_CAP_MULTIPLIER - group["points"]
                points = max(0.0, min(base * REPEAT_FACTOR, cap_left))

            group["points"] += points
            raw_total += points

            indicators.append({
                "field": field,
                "status": status,
                "severity": _severity(base),
                "points": _clean_number(points),
                "base_points": base,
                "occurrence": group["count"],
                "document_1": doc1,
                "document_2": doc2,
                "message": message,
            })

    # ---- final score ------------------------------------------------------
    risk_score = int(math.floor(raw_total + 0.5))      # round half up
    risk_score = max(MIN_SCORE, min(MAX_SCORE, risk_score))
    risk_level = _risk_level(risk_score)

    # Show the most important indicators first (stable sort keeps input order)
    indicators.sort(key=lambda ind: -ind["base_points"])

    # ---- data sufficiency -------------------------------------------------
    comparable = stats["matches"] + stats["partial_matches"] + stats["mismatches"]
    insufficient = comparable == 0

    # ---- reasons ----------------------------------------------------------
    reasons = []

    for ind in indicators:
        if ind["status"] == "MISMATCH":
            reasons.append(ind["message"])

    seen_partial_fields = set()
    for ind in indicators:
        if ind["status"] != "PARTIAL_MATCH" or ind["field"] in seen_partial_fields:
            continue
        seen_partial_fields.add(ind["field"])
        count = issue_groups[(ind["field"], "PARTIAL_MATCH")]["count"]
        if count > 1:
            reasons.append(f"{_field_label(ind['field'])} shows partial similarity "
                           "across the submitted documents.")
        else:
            reasons.append(ind["message"])

    if indicators:
        reasons.append(f"Potential inconsistency detected in {len(indicators)} "
                       "field comparison(s).")

    if stats["not_available"] == 1:
        reasons.append("One field was unavailable for comparison.")
    elif stats["not_available"] > 1:
        reasons.append("Some fields were unavailable for comparison "
                       f"({stats['not_available']} comparisons).")

    if insufficient:
        if isinstance(documents_compared, int) and documents_compared < 2:
            reasons.append("At least two documents are required for comparison.")
        else:
            reasons.append("No comparable field data was available.")
    elif not indicators:
        reasons.append("No inconsistencies were detected in the compared fields.")

    # ---- recommendation ---------------------------------------------------
    if insufficient:
        recommendation = INSUFFICIENT_DATA_RECOMMENDATION
    else:
        recommendation = RECOMMENDATIONS[risk_level]

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_indicators": indicators,
        "statistics": stats,
        "reasons": reasons,
        "recommendation": recommendation,
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# TESTS  (run with: python risk_engine.py)
# ---------------------------------------------------------------------------

def _field(field, status, similarity=None):
    entry = {"field": field, "status": status}
    if similarity is not None:
        entry["similarity"] = similarity
    return entry


def _pair(doc1, doc2, fields):
    return {"document_1": doc1, "document_2": doc2, "field_comparisons": fields}


def _results(pairs, documents_compared=None):
    return {
        "documents_compared": documents_compared if documents_compared is not None
        else len(pairs) + 1,
        "pair_comparisons": pairs,
    }


def _show(title, data, expected_level=None):
    print("=" * 70)
    print(title)
    print("=" * 70)
    result = calculate_risk(data)
    print(json.dumps(result, indent=2))
    if expected_level is not None:
        status = "PASS" if result["risk_level"] == expected_level else "FAIL"
        print(f"\n>> Expected {expected_level}, got {result['risk_level']} -> {status}")
    print()
    return result


if __name__ == "__main__":
    # TEST 1: all fields match -> LOW
    test1 = _results([
        _pair("Aadhaar", "Marksheet", [
            _field("name", "MATCH", 95),
            _field("dob", "MATCH", 100),
            _field("address", "MATCH", 92),
        ]),
    ])
    _show("TEST 1: All fields match (expect LOW)", test1, "LOW")

    # TEST 2: one name mismatch -> 30 points -> MEDIUM
    test2 = _results([
        _pair("Aadhaar", "Marksheet", [
            _field("name", "MISMATCH", 40),
            _field("dob", "MATCH", 100),
            _field("address", "MATCH", 90),
        ]),
    ])
    _show("TEST 2: One name mismatch (expect MEDIUM)", test2, "MEDIUM")

    # TEST 3: name + DOB + address mismatches -> 30 + 40 + 15 = 85 -> HIGH
    test3 = _results([
        _pair("Aadhaar", "Marksheet", [
            _field("name", "MISMATCH", 35),
            _field("dob", "MISMATCH", 0),
            _field("address", "MISMATCH", 30),
            _field("gender", "MATCH", 100),
        ]),
    ])
    _show("TEST 3: Name + DOB + Address mismatches (expect HIGH)", test3, "HIGH")

    # TEST 4: same name mismatch across 4 pairs -> capped (30+7.5+7.5+0 = 45)
    test4 = _results([
        _pair("Aadhaar", "Marksheet", [_field("name", "MISMATCH", 30)]),
        _pair("Aadhaar", "Income Certificate", [_field("name", "MISMATCH", 28)]),
        _pair("Aadhaar", "Scholarship Application", [_field("name", "MISMATCH", 33)]),
        _pair("Marksheet", "Income Certificate", [_field("name", "MISMATCH", 31)]),
    ], documents_compared=4)
    _show("TEST 4: Repeated name mismatch in 4 pairs (duplicate penalty, expect MEDIUM)",
          test4, "MEDIUM")

    # TEST 5: partial matches + unavailable fields + unknown field
    test5 = _results([
        _pair("Aadhaar", "Marksheet", [
            _field("address", "PARTIAL_MATCH", 70),
            _field("name", "PARTIAL_MATCH", 78),
            _field("income", "NOT_AVAILABLE"),
        ]),
        _pair("Aadhaar", "Income Certificate", [
            _field("address", "PARTIAL_MATCH", 72),
            _field("father_name", "MISMATCH", 20),
            _field("pincode", "NOT_AVAILABLE"),
        ]),
    ], documents_compared=3)
    # address partial 7 + repeat 1.75 + name partial 10 + unknown-field mismatch 10
    # = 28.75 -> 29 -> MEDIUM (NOT_AVAILABLE adds nothing)
    _show("TEST 5: Partial matches, unknown field, NOT_AVAILABLE (expect MEDIUM)",
          test5, "MEDIUM")

    # TEST 6: edge cases - must never crash
    print("=" * 70)
    print("TEST 6: Edge cases (empty / one document / malformed data)")
    print("=" * 70)
    edge_cases = {
        "None input": None,
        "Empty dict": {},
        "One document only": {"documents_compared": 1, "pair_comparisons": []},
        "Only NOT_AVAILABLE": _results([
            _pair("Aadhaar", "Marksheet", [_field("name", "NOT_AVAILABLE")])]),
        "Malformed data": {
            "pair_comparisons": [
                "not a dict",
                {"document_1": "Aadhaar", "field_comparisons": "oops"},
                {"document_1": "A", "document_2": "B", "field_comparisons": [
                    {"status": "MISMATCH"},                  # no field
                    {"field": "name"},                       # no status
                    {"field": "dob", "status": "WEIRD"},     # unknown status
                    42,                                      # not a dict
                    {"field": "Date of Birth", "status": "mismatch"},  # alias, lowercase
                ]},
            ]
        },
    }
    for name, data in edge_cases.items():
        result = calculate_risk(data)
        print(f"\n--- {name} ---")
        print(json.dumps(result, indent=2))
    print()
    print("All tests executed without crashing.")
