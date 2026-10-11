"""
extraction_validation.py - Deterministic Validation, Grounding, and Normalization Layer.

This module enforces strict anti-hallucination constraints on all extracted document data:
  1. Evidence grounding: verbatim snippet verification in raw OCR via RapidFuzz.
  2. Value grounding: strict containment for identity numbers, token fuzzy-matching for names.
  3. Format validation: Aadhaar (Verhoeff checksum), PAN, IFSC, GSTIN, Email regexes (warning-only).
  4. Structural validation: DOB bounds, gender enum, marks consistency, and marksheet pincode suppression.
  5. Deterministic normalization: Indian day-first DD/MM/YYYY dates, digit cleanup, and schema mapping.
"""

from datetime import date, datetime
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import unicodedata

from rapidfuzz import fuzz

# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

LEGACY_KEYS: List[str] = [
    "document_type", "name", "father_name", "mother_name", "spouse_name", "relation",
    "dob", "age", "gender", "nationality", "address", "pincode", "district", "state",
    "phone", "email", "aadhaar_number", "aadhaar_checksum_valid", "aadhaar_masked",
    "pan_number", "voter_id", "passport_number", "driving_licence_number", "ifsc_code",
    "gstin", "bank_account_number", "income", "income_in_words", "certificate_number",
    "application_number", "reference_number", "issue_date", "application_date", "valid_until",
    "issuing_authority", "place", "caste", "category", "board", "institute", "exam_name",
    "exam_year", "roll_number", "enrollment_number", "marks_obtained", "max_marks",
    "percentage", "cgpa", "result", "invoice_number", "invoice_amount", "all_dates"
]

ADDITIVE_KEYS: List[str] = [
    "city", "village", "taluka", "registration_number", "seat_number", "sid_number",
    "statement_number", "obtained_marks", "total_marks", "exam_month_year", "exam_date",
    "course", "stream", "grade", "income_period", "financial_year", "invoice_date",
    "certificate_date", "annual_income"
]

ALL_SCHEMA_KEYS: List[str] = LEGACY_KEYS + ADDITIVE_KEYS

ALLOWED_DOC_TYPES: Set[str] = {
    "Aadhaar", "PAN Card", "Voter ID", "Driving Licence", "Passport",
    "Income Certificate", "Caste Certificate", "Domicile Certificate", "Birth Certificate",
    "Marksheet", "Bank Statement", "Ration Card", "Scholarship Application", "Invoice", "Unknown"
}

ID_FIELDS: Set[str] = {
    "aadhaar_number", "pan_number", "voter_id", "passport_number", "driving_licence_number",
    "ifsc_code", "gstin", "bank_account_number", "certificate_number", "application_number",
    "registration_number", "reference_number", "enrollment_number", "roll_number",
    "seat_number", "sid_number", "statement_number", "invoice_number"
}

NAME_FIELDS: Set[str] = {
    "name", "father_name", "mother_name", "spouse_name"
}

DATE_FIELDS: Set[str] = {
    "dob", "issue_date", "application_date", "valid_until", "exam_date",
    "invoice_date", "certificate_date"
}

HIGH_PRECISION_RULES_FIELDS: Set[str] = {
    "aadhaar_number", "pan_number", "voter_id", "passport_number",
    "driving_licence_number", "ifsc_code", "gstin", "email", "phone"
}

LLM_SCHEMA_MAPPINGS: Dict[str, Any] = {
    "expiry_date": "valid_until",
    "institution": "institute",
    "total_marks_obtained": [("marks_obtained", "str"), ("obtained_marks", "int")],
    "total_marks": [("max_marks", "str"), ("total_marks", "int")],
}

_HONORIFICS: Set[str] = {
    "mr", "mrs", "ms", "miss", "dr", "shri", "shree", "sri", "smt", "shrimati", "kumari"
}

# =============================================================================
# VERHOEFF ALGORITHM (FOR AADHAAR CHECKSUM VALIDATION)
# =============================================================================

_VERHOEFF_D: List[List[int]] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

_VERHOEFF_P: List[List[int]] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

_VERHOEFF_INV: List[int] = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(number: Any) -> bool:
    """Validate 12-digit Aadhaar number using the Verhoeff dihedral group check."""
    if not number:
        return False
    digits = [int(c) for c in str(number) if c.isdigit()]
    if len(digits) != 12:
        return False
    c = 0
    for i, item in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][item]]
    return c == 0


# =============================================================================
# NORMALIZATION HELPERS
# =============================================================================

def _build_digit_map():
    m = {}
    for cp in list(range(0x0660, 0x066A)) + list(range(0x0900, 0x0E00)):
        if unicodedata.category(chr(cp)) == "Nd":
            m[cp] = str(unicodedata.digit(chr(cp)))
    return m

_DIGIT_MAP = _build_digit_map()


def convert_indic_digits(text: Any) -> str:
    """Convert Gujarati, Devanagari, and other Indic numerals to ASCII digits."""
    if not text:
        return ""
    return str(text).translate(_DIGIT_MAP)


def fix_ocr_digit_confusions(text: str) -> str:
    """
    Fix common OCR character confusion (O/0, l/1, I/1) strictly inside numeric contexts.
    Never applied to generic names or words.
    """
    if not text:
        return ""
    trans = str.maketrans({
        "O": "0", "o": "0",
        "l": "1", "I": "1", "i": "1",
        "S": "5", "s": "5",
        "B": "8",
    })
    return text.translate(trans)


_MONTH_NAME_MAP: Dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def normalize_date_string(raw: Any) -> Optional[str]:
    """
    Normalizes dates into standard DD/MM/YYYY format with Indian day-first assumption.
    Handles delimiters (-, ., /), text month names, 2-digit years, and OCR digit confusions.
    Year-only format 'YYYY' is preserved if only a birth year is present.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"null", "none", "n/a", "unknown", "-"}:
        return None

    # Handle single 4-digit year (e.g. "2006" on Aadhaar)
    if re.fullmatch(r"(19\d\d|20\d\d)", s):
        y = int(s)
        if 1900 <= y <= date.today().year:
            return s

    # First, handle month-name patterns: "06 Dec 2006", "6-December-2006", "06/Dec/2006"
    m_text = re.search(r"(\d{1,2})[\s\-/.]([A-Za-z]{3,9})[\s\-/.]([A-Za-z0-9]{2,4})", s)
    if m_text:
        d_str, mon_str, y_str = m_text.groups()
        mon = _MONTH_NAME_MAP.get(mon_str.lower()) or _MONTH_NAME_MAP.get(mon_str[:3].lower())
        if mon:
            y_clean = fix_ocr_digit_confusions(y_str)
            d_clean = fix_ocr_digit_confusions(d_str)
            if y_clean.isdigit() and d_clean.isdigit():
                d = int(d_clean)
                y = int(y_clean)
                if y < 100:
                    y += 2000 if y <= 30 else 1900
                try:
                    dt = date(y, mon, d)
                    return f"{dt.day:02d}/{dt.month:02d}/{dt.year}"
                except ValueError:
                    pass

    # Numeric pattern with separators: dd/mm/yyyy or yyyy-mm-dd
    # Normalize OCR digit confusion for separator-delimited candidates
    cand = re.sub(r"[^\w\s/.\-]", " ", s)
    m_num = re.search(r"(\w{1,4})[\s/.\-]+(\w{1,2})[\s/.\-]+(\w{2,4})", cand)
    if m_num:
        p1, p2, p3 = m_num.groups()
        p1_fixed = fix_ocr_digit_confusions(p1)
        p2_fixed = fix_ocr_digit_confusions(p2)
        p3_fixed = fix_ocr_digit_confusions(p3)

        if p1_fixed.isdigit() and p2_fixed.isdigit() and p3_fixed.isdigit():
            # Check ISO format: YYYY-MM-DD
            if len(p1_fixed) == 4:
                y, mo, d = int(p1_fixed), int(p2_fixed), int(p3_fixed)
            else:
                d, mo = int(p1_fixed), int(p2_fixed)
                y = int(p3_fixed)
                if y < 100:
                    y += 2000 if y <= 30 else 1900

            try:
                dt = date(y, mo, d)
                return f"{dt.day:02d}/{dt.month:02d}/{dt.year}"
            except ValueError:
                # Might have swapped month/day
                if d <= 12 and mo > 12:
                    try:
                        dt = date(y, d, mo)
                        return f"{dt.day:02d}/{dt.month:02d}/{dt.year}"
                    except ValueError:
                        return None
                return None

    return None


def normalize_clean_name(raw: Any) -> Optional[str]:
    """
    Collapses whitespace and strips noise while preserving printed case and exact spelling.
    If both English and Gujarati/Devanagari appear, prioritizes Latin script.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"null", "none", "n/a", "unknown", "-"}:
        return None

    # If both Latin and native scripts appear, prioritize Latin script
    has_latin = bool(re.search(r"[A-Za-z]", s))
    has_indic = bool(re.search(r"[\u0900-\u0DFF]", s))

    if has_latin and has_indic:
        latin_part = re.sub(r"[^A-Za-z\s.\-]", " ", s)
        latin_cleaned = " ".join(latin_part.split()).strip(" .-")
        if latin_cleaned and len(latin_cleaned) >= 2:
            return latin_cleaned

    # Preserve the printed characters with whitespace collapsed
    cleaned = " ".join(s.split()).strip(" .-")
    return cleaned if len(cleaned) >= 2 else None


def normalize_name_for_compare(name: Any) -> Optional[str]:
    """
    Normalizes a person name for equality comparison:
    Casefold, strip honorifics (Shri, Smt, Mr, Mrs, Ms, Kumari, etc.), remove punctuation, sort tokens.
    Guarantees 'AKSHAY SOJITRA' and 'Akshay Sojitra' or 'Sojitra Akshay' produce identical keys.
    """
    if not name:
        return None
    text = str(name).casefold()
    text = re.sub(r"[\'’`]", "", text)
    text = re.sub(r"[^\w\s]|_", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = [t for t in text.split() if t not in _HONORIFICS]
    tokens.sort()
    return " ".join(tokens) or None


def normalize_pincode_strict(val: Any) -> Optional[str]:
    """Extract clean 6-digit Indian postal pincode (first digit 1-9)."""
    if not val:
        return None
    s = fix_ocr_digit_confusions(str(val))
    digits = re.sub(r"\D", "", s)
    if len(digits) == 6 and digits[0] in "123456789":
        return digits
    m = re.search(r"\b[1-9]\d{5}\b", digits)
    return m.group(0) if m else None


def normalize_phone_strict(val: Any) -> Optional[str]:
    """Extract 10-digit Indian mobile number."""
    if not val:
        return None
    s = fix_ocr_digit_confusions(str(val))
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 10:
        last10 = digits[-10:]
        if last10[0] in "6789":
            return last10
    return None


_INCOME_MULTIPLIERS = [
    (r"\b(crore|crores|cr)\b", 10000000),
    (r"\b(lakh|lakhs|lac|lacs|lpa)\b", 100000),
    (r"\b(thousand|k)\b", 1000),
]


def normalize_income_amount(val: Any) -> Optional[str]:
    """
    Normalizes income representations into numeric digit strings.
    Handles 'Rs. 1,80,000/-', '1.8 Lakh', '180000', and words.
    """
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s or s in {"null", "none", "n/a", "-"}:
        return None

    # Words check (common Indian certificates)
    if "one lakh twenty thousand" in s:
        return "120000"
    if "one lakh fifty thousand" in s:
        return "150000"
    if "one lakh" in s:
        return "100000"
    if "two lakh" in s:
        return "200000"
    if "two lakh fifty thousand" in s:
        return "250000"

    # Numeric multiplier check (e.g. "1.8 Lakh")
    m_num = re.search(r"(\d+(?:\.\d+)?)", s)
    if m_num:
        num_val = float(m_num.group(1))
        for pat, mult in _INCOME_MULTIPLIERS:
            if re.search(pat, s):
                return str(int(num_val * mult))

    # Standard digits extract
    fixed = convert_indic_digits(fix_ocr_digit_confusions(s))
    clean = re.sub(r"[^\d]", "", fixed)
    return clean if clean else None


# =============================================================================
# GROUNDING CHECKS
# =============================================================================

def is_evidence_grounded(snippet: Optional[str], ocr_text: str) -> bool:
    """Verifies that the evidence snippet is grounded in raw OCR text (fuzzy >= 90)."""
    if not snippet or not snippet.strip():
        return False
    snip_norm = convert_indic_digits(" ".join(snippet.split()).casefold())
    ocr_norm = convert_indic_digits(" ".join(ocr_text.split()).casefold())

    if snip_norm in ocr_norm:
        return True

    ratio = fuzz.partial_ratio(snip_norm, ocr_norm)
    return ratio >= 90.0


def is_id_number_grounded(value: Optional[str], ocr_text: str) -> bool:
    """
    Identity numbers must appear in OCR text EXACTLY without digit modification.
    Spaces and zero-width characters are ignored during matching.
    """
    if not value or not str(value).strip():
        return False
    clean_val = re.sub(r"[\s\u200b\u200c\u200d\ufeff\-]", "", str(value)).casefold()
    ocr_ascii = convert_indic_digits(ocr_text)
    clean_ocr = re.sub(r"[\s\u200b\u200c\u200d\ufeff\-]", "", ocr_ascii).casefold()
    return clean_val in clean_ocr


def is_name_grounded(name: Optional[str], ocr_text: str) -> bool:
    """Every token in the name (length >= 2) must fuzzy-match (>= 85) an OCR token."""
    if not name or not str(name).strip():
        return False
    name_tokens = [t.casefold() for t in re.findall(r"[\w]+", str(name)) if len(t) >= 2]
    if not name_tokens:
        return False

    ocr_tokens = [t.casefold() for t in re.findall(r"[\w]+", ocr_text) if len(t) >= 2]
    if not ocr_tokens:
        return False

    for n_tok in name_tokens:
        matched = any(fuzz.ratio(n_tok, o_tok) >= 85 for o_tok in ocr_tokens)
        if not matched:
            return False
    return True


def is_traceable_to_ocr(value: Any, ocr_text: str) -> bool:
    """Verifies that underlying digits/date components exist in the OCR text."""
    if value is None:
        return True
    val_str = str(value)
    digits = re.findall(r"\d+", val_str)
    if not digits:
        return True
    ocr_ascii = convert_indic_digits(ocr_text)
    ocr_digits = "".join(re.findall(r"\d+", ocr_ascii))
    for d in digits:
        if d not in ocr_digits and d not in ocr_ascii and d not in ocr_text:
            return False
    return True


# =============================================================================
# MAIN GROUNDING & VALIDATION PIPELINE
# =============================================================================

def validate_and_ground_fields(
    raw_fields: Dict[str, Any],
    evidence: Dict[str, str],
    ocr_text: str,
    warnings: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validates, grounds, normalizes, and aligns LLM-extracted fields to the legacy schema.
    Returns: (sanitized_dictionary_with_all_keys, list_of_warnings)
    """
    result: Dict[str, Any] = {k: None for k in ALL_SCHEMA_KEYS}
    result["all_dates"] = []
    active_warnings: List[str] = list(warnings or [])

    input_fields = dict(raw_fields or {})

    # 1. Apply Schema Mapping from LLM field names to target schema
    for src_key, target in LLM_SCHEMA_MAPPINGS.items():
        if src_key in input_fields and input_fields[src_key] is not None:
            val = input_fields[src_key]
            if isinstance(target, str):
                if target not in input_fields or input_fields[target] is None:
                    input_fields[target] = val
            elif isinstance(target, list):
                for target_field, target_type in target:
                    if target_type == "int":
                        try:
                            clean_digits = re.sub(r"[^\d]", "", str(val))
                            input_fields[target_field] = int(clean_digits) if clean_digits else None
                        except Exception:
                            input_fields[target_field] = None
                    else:
                        input_fields[target_field] = str(val)

    # 2. Document Type Structural Validation
    raw_type = input_fields.get("document_type")
    if raw_type and str(raw_type) in ALLOWED_DOC_TYPES:
        result["document_type"] = str(raw_type)
    else:
        # Case-insensitive match check
        matched_type = "Unknown"
        if raw_type:
            for allowed in ALLOWED_DOC_TYPES:
                if allowed.casefold() == str(raw_type).strip().casefold():
                    matched_type = allowed
                    break
        result["document_type"] = matched_type

    # 3. Evidence Grounding Verification
    for key, val in list(input_fields.items()):
        if val is None or key in {"document_type", "all_dates"}:
            continue
        ev = evidence.get(key)
        if ev:
            if not is_evidence_grounded(ev, ocr_text):
                input_fields[key] = None
                active_warnings.append(f"ungrounded: {key}")
        # Note: If no evidence string was provided by LLM, we will rely on value grounding below

    # 4. Value Grounding Verification
    # (a) Identity & Reference Numbers: must match OCR text exactly
    for field in ID_FIELDS:
        val = input_fields.get(field)
        if val is not None:
            if not is_id_number_grounded(val, ocr_text):
                input_fields[field] = None
                active_warnings.append(f"ungrounded_value: {field}")

    # (b) Names: every token must fuzzy match (>= 85)
    for field in NAME_FIELDS:
        val = input_fields.get(field)
        if val is not None:
            if not is_name_grounded(val, ocr_text):
                input_fields[field] = None
                active_warnings.append(f"ungrounded_name: {field}")

    # (c) Dates, Marks, Income: digits traceability check
    for field in (DATE_FIELDS | {"income", "marks_obtained", "max_marks", "obtained_marks", "total_marks"}):
        val = input_fields.get(field)
        if val is not None:
            if not is_traceable_to_ocr(val, ocr_text):
                input_fields[field] = None
                active_warnings.append(f"untraceable: {field}")

    # 5. Normalization & Format Validation
    # Names
    for field in NAME_FIELDS:
        if input_fields.get(field):
            result[field] = normalize_clean_name(input_fields[field])

    # Dates
    for field in DATE_FIELDS:
        raw_d = input_fields.get(field)
        if raw_d:
            norm_d = normalize_date_string(raw_d)
            result[field] = norm_d

    # Structural Validation on DOB (no future dates, >= 1900)
    if result["dob"]:
        m_dob = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", result["dob"])
        if m_dob:
            d_d, d_m, d_y = map(int, m_dob.groups())
            today = date.today()
            if d_y < 1900 or date(d_y, d_m, d_d) > today:
                result["dob"] = None
                active_warnings.append("invalid_dob_range")

    # Dates list aggregation
    all_dates_raw = input_fields.get("all_dates") or []
    if isinstance(all_dates_raw, list):
        norm_dates = []
        for d in all_dates_raw:
            nd = normalize_date_string(d)
            if nd and nd not in norm_dates:
                norm_dates.append(nd)
        # Also ensure valid dates from fields are included
        for df in DATE_FIELDS:
            if result.get(df) and result[df] not in norm_dates:
                norm_dates.append(result[df])
        result["all_dates"] = norm_dates

    # Gender validation
    raw_gender = input_fields.get("gender")
    if raw_gender:
        g_low = str(raw_gender).strip().lower()
        if g_low in {"male", "m", "purush", "પુરુષ", "पुरुष"}:
            result["gender"] = "Male"
        elif g_low in {"female", "f", "mahila", "સ્ત્રી", "महिला", "stree"}:
            result["gender"] = "Female"
        elif g_low in {"transgender", "tg", "other", "ત્રીજી જાતિ"}:
            result["gender"] = "Transgender"
        else:
            result["gender"] = None

    # Income
    if input_fields.get("income"):
        result["income"] = normalize_income_amount(input_fields["income"])
        result["annual_income"] = result["income"]

    # Phone
    if input_fields.get("phone"):
        result["phone"] = normalize_phone_strict(input_fields["phone"])

    # Email format validation
    if input_fields.get("email"):
        em = str(input_fields["email"]).strip()
        if re.fullmatch(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", em):
            result["email"] = em
        else:
            result["email"] = em
            active_warnings.append("format_warning: email")

    # PAN format validation (Warning-only, preserves malformed value)
    if input_fields.get("pan_number"):
        pan = re.sub(r"\s+", "", str(input_fields["pan_number"])).upper()
        result["pan_number"] = pan
        if not re.fullmatch(r"^[A-Z]{5}[0-9]{4}[A-Z]$", pan):
            active_warnings.append("format_warning: pan_number")

    # Aadhaar format & Verhoeff validation
    if input_fields.get("aadhaar_number"):
        raw_a = re.sub(r"\s+", "", str(input_fields["aadhaar_number"]))
        if len(raw_a) == 12 and raw_a.isdigit():
            result["aadhaar_number"] = f"{raw_a[:4]} {raw_a[4:8]} {raw_a[8:]}"
            result["aadhaar_masked"] = f"XXXX XXXX {raw_a[-4:]}"
            is_v = verhoeff_valid(raw_a)
            result["aadhaar_checksum_valid"] = is_v
            if not is_v:
                active_warnings.append("format_warning: aadhaar_checksum_invalid")
        else:
            result["aadhaar_number"] = str(input_fields["aadhaar_number"]).strip()
            result["aadhaar_checksum_valid"] = False
            active_warnings.append("format_warning: aadhaar_length_invalid")

    # IFSC format validation
    if input_fields.get("ifsc_code"):
        ifsc = re.sub(r"\s+", "", str(input_fields["ifsc_code"])).upper()
        result["ifsc_code"] = ifsc
        if not re.fullmatch(r"^[A-Z]{4}0[A-Z0-9]{6}$", ifsc):
            active_warnings.append("format_warning: ifsc_code")

    # GSTIN format validation
    if input_fields.get("gstin"):
        gstin = re.sub(r"\s+", "", str(input_fields["gstin"])).upper()
        result["gstin"] = gstin
        if len(gstin) != 15:
            active_warnings.append("format_warning: gstin")

    # Identity and Reference numbers assignment
    for f in [
        "voter_id", "passport_number", "driving_licence_number", "bank_account_number",
        "certificate_number", "application_number", "reference_number", "registration_number",
        "enrollment_number", "roll_number", "seat_number", "sid_number", "statement_number",
        "invoice_number"
    ]:
        if input_fields.get(f):
            result[f] = str(input_fields[f]).strip()

    # Marksheet marks consistency & integer conversion
    obt_raw = input_fields.get("obtained_marks") or input_fields.get("marks_obtained")
    tot_raw = input_fields.get("total_marks") or input_fields.get("max_marks")

    obt_int = None
    tot_int = None
    if obt_raw is not None:
        try:
            obt_int = int(re.sub(r"[^\d]", "", str(obt_raw)))
        except ValueError:
            obt_int = None
    if tot_raw is not None:
        try:
            tot_int = int(re.sub(r"[^\d]", "", str(tot_raw)))
        except ValueError:
            tot_int = None

    if obt_int is not None and tot_int is not None:
        if obt_int > tot_int:
            active_warnings.append("marks_obtained_greater_than_max")
            obt_int = None
            obt_raw = None

    result["obtained_marks"] = obt_int
    result["marks_obtained"] = str(obt_int) if obt_int is not None else (str(obt_raw) if obt_raw else None)
    result["total_marks"] = tot_int
    result["max_marks"] = str(tot_int) if tot_int is not None else (str(tot_raw) if tot_raw else None)

    # Percentage structural check (0 to 100)
    if input_fields.get("percentage") is not None:
        try:
            p_clean = re.sub(r"[^\d.]", "", str(input_fields["percentage"]))
            p_val = float(p_clean)
            if 0.0 <= p_val <= 100.0:
                result["percentage"] = str(round(p_val, 2))
            else:
                result["percentage"] = None
                active_warnings.append("invalid_percentage_range")
        except ValueError:
            result["percentage"] = None

    # Pincode validation & Marksheet collision prevention
    raw_pin = input_fields.get("pincode")
    norm_pin = normalize_pincode_strict(raw_pin) if raw_pin else None

    # Check collision: Pincode must NOT equal or be contained in seat_number, roll_number,
    # sid_number, statement_number, certificate_number, phone, or aadhaar_number.
    if norm_pin:
        conflict = False
        conflict_sources = [
            result.get("seat_number"), result.get("roll_number"), result.get("sid_number"),
            result.get("statement_number"), result.get("certificate_number"), result.get("phone"),
            result.get("aadhaar_number")
        ]
        for src in conflict_sources:
            if src:
                src_digits = re.sub(r"\D", "", str(src))
                if norm_pin in src_digits or src_digits in norm_pin:
                    conflict = True
                    break

        if conflict or result["document_type"] == "Marksheet":
            norm_pin = None
            active_warnings.append("pincode_suppressed_as_id_or_marksheet_seat")

    result["pincode"] = norm_pin

    # Fill remaining textual fields
    remaining_text_fields = [
        "address", "village", "taluka", "city", "district", "state", "nationality",
        "relation", "age", "income_in_words", "income_period", "financial_year",
        "issuing_authority", "place", "caste", "category", "board", "institute",
        "exam_name", "exam_year", "exam_month_year", "course", "stream", "grade",
        "result", "cgpa", "invoice_amount"
    ]
    for rf in remaining_text_fields:
        if input_fields.get(rf) is not None and result.get(rf) is None:
            result[rf] = str(input_fields[rf]).strip()

    return result, active_warnings
