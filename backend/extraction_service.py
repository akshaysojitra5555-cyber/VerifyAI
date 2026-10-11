"""
extraction_service.py - Orchestrator for Multi-Mode Document Extraction in VerifyAI.

Supports three operational modes configured via EXTRACTION_MODE (.env):
  - 'hybrid' (default): Gemini 2.5 Flash is primary. If Gemini returns null,
    high-precision identity fields (Aadhaar, PAN, Voter ID, Passport, DL, IFSC, GSTIN, Email, Phone)
    are filled from deterministic rules ONLY if they pass validation.
    Name, address, pincode, dates, marks, and income are NEVER filled from rules when Gemini says null.
  - 'llm': Gemini only; falls back to rules if Gemini encounters an unrecoverable failure.
  - 'rules': Pure deterministic rule extraction via extractor.py (zero Gemini API calls).

Guarantees 100% backward and schema compatibility with comparison.py and risk_engine.py.
"""

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

import extractor
from extractor_llm import execute_llm_extraction, CONFIGURED_MODEL
from extraction_validation import (
    validate_and_ground_fields,
    verhoeff_valid,
    HIGH_PRECISION_RULES_FIELDS,
    ALL_SCHEMA_KEYS,
)

load_dotenv()

logger = logging.getLogger("VerifyAI.extraction_service")
logger.setLevel(logging.INFO)

# Global configuration
EXTRACTION_MODE = os.getenv("EXTRACTION_MODE", "hybrid").strip().lower()
MAX_CONCURRENCY = int(os.getenv("GEMINI_MAX_CONCURRENCY", "3"))

_async_semaphore: Optional[asyncio.Semaphore] = None


def get_semaphore() -> asyncio.Semaphore:
    """Returns an asyncio.Semaphore configured with GEMINI_MAX_CONCURRENCY."""
    global _async_semaphore
    if _async_semaphore is None:
        _async_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    return _async_semaphore


# Diagnostics store for non-intrusive metadata retrieval
_last_diagnostics: List[Dict[str, Any]] = []


def get_last_extraction_meta() -> List[Dict[str, Any]]:
    """Returns metadata for the most recent extraction calls (method, latency, warnings)."""
    return list(_last_diagnostics)


def _clear_diagnostics() -> None:
    global _last_diagnostics
    _last_diagnostics = []


def _enrich_rule_extraction(rule_res: Dict[str, Any], ocr_text: str) -> Dict[str, Any]:
    """
    Enriches deterministic rule-based output with high-confidence pattern heuristics
    for Indian documents (e.g. Aadhaar, PAN, certificates).
    Specifically handles real-world OCR layouts where labels like 'Name:' are absent.
    """
    enriched = dict(rule_res)
    text = ocr_text or ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1. Aadhaar Checksum & Document Type Identification
    if enriched.get("aadhaar_number"):
        clean_a = re.sub(r"\D", "", str(enriched["aadhaar_number"]))
        if len(clean_a) == 12:
            is_v = verhoeff_valid(clean_a)
            enriched["aadhaar_checksum_valid"] = is_v
            if is_v:
                enriched["aadhaar_masked"] = f"XXXX XXXX {clean_a[-4:]}"
                if enriched.get("document_type") in (None, "Unknown"):
                    enriched["document_type"] = "Aadhaar"

    # Also detect Aadhaar/PAN from text keywords if document_type is still Unknown
    if enriched.get("document_type") in (None, "Unknown"):
        low_t = text.lower()
        if any(k in low_t for k in ("aadhaar", "uidai", "mera aadhaar", "aadhar", "unique identification")):
            enriched["document_type"] = "Aadhaar"
        elif "income tax department" in low_t or "permanent account number" in low_t or enriched.get("pan_number"):
            enriched["document_type"] = "PAN Card"

    # 2. Name Extraction Heuristics (if name is missing in rule_res)
    if not enriched.get("name"):
        cand_name = None

        # Heuristic A: Look for name line directly preceding DOB / Date of Birth
        # On standard Indian Aadhaar cards, the English name is printed 1-2 lines above DOB.
        for i, l in enumerate(lines):
            if re.search(r"DOB|Date\s*of\s*Birth|जन्म|તારીખ", l, re.I):
                for prev in reversed(lines[max(0, i - 3):i]):
                    clean = re.sub(r"[^A-Za-z\s.'-]", "", prev).strip()
                    words = clean.split()
                    if len(words) >= 2 and all(len(w) >= 2 for w in words):
                        low_c = clean.lower()
                        if not any(k in low_c for k in [
                            "government", "india", "male", "female", "dob", "date",
                            "aadhaar", "uidai", "authority", "card", "mera", "help",
                            "enrollment", "birth", "yob", "year", "tariq"
                        ]):
                            cand_name = clean
                            break
                if cand_name:
                    break

        # Heuristic B: Explicit label regex (Candidate Name, Card Holder, etc.)
        if not cand_name:
            m_lbl = re.search(
                r"(?:Candidate'?s?\s*Name|Name|Card\s*Holder|Full\s*Name|Name\s*of\s*Applicant)\s*[:\-]\s*([A-Za-z\s.'-]+)",
                text,
                re.I
            )
            if m_lbl:
                clean_lbl = re.sub(r"[^A-Za-z\s.'-]", "", m_lbl.group(1)).strip()
                words = clean_lbl.split()
                if len(words) >= 1 and len(clean_lbl) >= 3:
                    cand_name = clean_lbl

        if cand_name:
            enriched["name"] = cand_name

    # 3. Father / Guardian Name & Relation
    if not enriched.get("father_name"):
        m_rel = re.search(
            r"\b(S/o|D/o|W/o|C/o|Care\s*of|Son\s*of|Daughter\s*of|Wife\s*of)\s*[:\-]?\s*([A-Za-z\s.'-]+?)(?:,|\n|;|\bPO\b|\bDIST\b|\bVILL\b|\bTALUKA\b|\bPIN\b|\d|$)",
            text,
            re.I
        )
        if m_rel:
            rel_type = m_rel.group(1).strip()
            clean_f = re.sub(r"[^A-Za-z\s.'-]", "", m_rel.group(2)).strip()
            words_f = clean_f.split()
            if len(words_f) >= 1 and len(clean_f) >= 3:
                enriched["father_name"] = clean_f
                if not enriched.get("relation"):
                    enriched["relation"] = rel_type

    # 4. District & State
    if not enriched.get("district"):
        m_dist = re.search(r"\bDIST\s*[:\-]?\s*([A-Za-z]+)", text, re.I)
        if m_dist:
            enriched["district"] = m_dist.group(1).strip()

    if not enriched.get("state"):
        known_states = [
            "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
            "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
            "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
            "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
            "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi"
        ]
        low_t = text.lower()
        for st in known_states:
            if re.search(r"\b" + re.escape(st.lower()) + r"\b", low_t):
                enriched["state"] = st
                break

    # 5. Pincode
    if not enriched.get("pincode") and enriched.get("document_type") != "Marksheet":
        m_pin = re.search(r"\b[1-9]\d{5}\b", text)
        if m_pin:
            pin_cand = m_pin.group(0)
            aadhaar_digits = re.sub(r"\D", "", str(enriched.get("aadhaar_number") or ""))
            if pin_cand not in aadhaar_digits:
                enriched["pincode"] = pin_cand

    # 6. Address
    if not enriched.get("address"):
        m_addr = re.search(
            r"((?:S/o|D/o|W/o|C/o|Address\s*:)[\s\S]+?(?:\b\d{6}\b|$))",
            text,
            re.I
        )
        if m_addr:
            addr_str = re.sub(r"\s+", " ", m_addr.group(1)).strip()
            if len(addr_str) >= 15:
                enriched["address"] = addr_str

    return enriched


# =============================================================================
# SYNCHRONOUS DOCUMENT EXTRACTION
# =============================================================================

def extract_document_fields(
    ocr_text: str,
    filename: Optional[str] = None,
    client_override: Optional[Any] = None,
    mode_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Primary extraction entry point.
    Extracts structured fields from raw OCR text using the configured mode (rules, llm, or hybrid).
    Always returns a dictionary containing all legacy and additive schema keys.
    """
    mode = (mode_override or EXTRACTION_MODE).lower()
    t0 = time.time()
    filename_str = filename or "document"

    diagnostics: Dict[str, Any] = {
        "filename": filename_str,
        "mode": mode,
        "model": CONFIGURED_MODEL if mode in ("llm", "hybrid") else None,
        "warnings": [],
        "latency_sec": 0.0,
        "method_used": "unknown",
    }

    # -------------------------------------------------------------------------
    # MODE 1: RULES ONLY (ZERO GEMINI CALLS)
    # -------------------------------------------------------------------------
    if mode == "rules":
        rule_res = extractor.extract_fields(ocr_text)
        rule_res = _enrich_rule_extraction(rule_res, ocr_text)
        # Ground and ensure all schema keys are present
        sanitized, warnings = validate_and_ground_fields(
            raw_fields=rule_res,
            evidence={},
            ocr_text=ocr_text,
            warnings=["extraction_mode: rules"],
        )
        sanitized["filename"] = filename_str
        sanitized["extraction_method"] = "rules"

        diagnostics["method_used"] = "rules"
        diagnostics["warnings"] = warnings
        diagnostics["latency_sec"] = round(time.time() - t0, 3)
        _last_diagnostics.append(diagnostics)
        return sanitized

    # -------------------------------------------------------------------------
    # MODE 2 & 3: LLM OR HYBRID EXTRACTION
    # -------------------------------------------------------------------------
    llm_output = execute_llm_extraction(
        ocr_text=ocr_text,
        filename=filename_str,
        client_override=client_override,
    )

    # Emergency Fallback to Rules if LLM call failed or returned None
    if not llm_output or not isinstance(llm_output, dict):
        logger.warning(f"LLM returned no output for {filename_str}. Triggering rules fallback.")
        rule_res = extractor.extract_fields(ocr_text)
        rule_res = _enrich_rule_extraction(rule_res, ocr_text)
        sanitized, warnings = validate_and_ground_fields(
            raw_fields=rule_res,
            evidence={},
            ocr_text=ocr_text,
            warnings=[f"llm_failed_fallback_to_rules", f"mode: {mode}"],
        )
        sanitized["filename"] = filename_str
        sanitized["extraction_method"] = f"{mode}_fallback_to_rules"

        diagnostics["method_used"] = f"{mode}_fallback_to_rules"
        diagnostics["warnings"] = warnings
        diagnostics["latency_sec"] = round(time.time() - t0, 3)
        _last_diagnostics.append(diagnostics)
        return sanitized

    # Process and ground LLM extraction
    raw_doc_type = llm_output.get("document_type", "Unknown")
    raw_fields = llm_output.get("fields", {})
    if not isinstance(raw_fields, dict):
        raw_fields = {}
    raw_fields["document_type"] = raw_doc_type

    evidence = llm_output.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}

    llm_warnings = llm_output.get("warnings", [])
    if not isinstance(llm_warnings, list):
        llm_warnings = []

    sanitized, warnings = validate_and_ground_fields(
        raw_fields=raw_fields,
        evidence=evidence,
        ocr_text=ocr_text,
        warnings=llm_warnings,
    )

    # -------------------------------------------------------------------------
    # HYBRID ENRICHMENT: HIGH-PRECISION FALLBACK FOR NULL FIELDS
    # -------------------------------------------------------------------------
    if mode == "hybrid":
        rule_res = extractor.extract_fields(ocr_text)
        enriched_count = 0

        for field in HIGH_PRECISION_RULES_FIELDS:
            # Only fill if Gemini returned null / None
            if sanitized.get(field) is None and rule_res.get(field):
                rule_val = rule_res.get(field)

                # Validate Aadhaar with Verhoeff before backfilling
                if field == "aadhaar_number":
                    clean_a = str(rule_val).replace(" ", "")
                    if len(clean_a) == 12 and verhoeff_valid(clean_a):
                        sanitized["aadhaar_number"] = f"{clean_a[:4]} {clean_a[4:8]} {clean_a[8:]}"
                        sanitized["aadhaar_checksum_valid"] = True
                        sanitized["aadhaar_masked"] = f"XXXX XXXX {clean_a[-4:]}"
                        enriched_count += 1
                elif field in ("pan_number", "voter_id", "passport_number", "driving_licence_number",
                               "ifsc_code", "gstin", "email", "phone"):
                    sanitized[field] = str(rule_val).strip()
                    enriched_count += 1

        sanitized["extraction_method"] = "hybrid"
        diagnostics["method_used"] = "hybrid"
        if enriched_count > 0:
            warnings.append(f"hybrid_enriched_{enriched_count}_fields_from_rules")
    else:
        sanitized["extraction_method"] = "llm"
        diagnostics["method_used"] = "llm"

    sanitized["filename"] = filename_str

    # Final guarantee: all legacy and additive keys are present in dict
    for k in ALL_SCHEMA_KEYS:
        if k not in sanitized:
            sanitized[k] = [] if k == "all_dates" else None

    diagnostics["warnings"] = warnings
    diagnostics["latency_sec"] = round(time.time() - t0, 3)
    _last_diagnostics.append(diagnostics)

    return sanitized


# =============================================================================
# ASYNCHRONOUS EXTRACTION WITH THREADPOOL & CONCURRENCY LIMIT
# =============================================================================

async def extract_document_fields_async(
    ocr_text: str,
    filename: Optional[str] = None,
    client_override: Optional[Any] = None,
    mode_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Non-blocking async wrapper around extract_document_fields.
    Utilizes asyncio.to_thread and respects GEMINI_MAX_CONCURRENCY.
    """
    sem = get_semaphore()
    async with sem:
        return await asyncio.to_thread(
            extract_document_fields,
            ocr_text,
            filename,
            client_override,
            mode_override,
        )


async def extract_documents_concurrently(
    documents_data: List[Tuple[str, str]],
    client_override: Optional[Any] = None,
    mode_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts multiple documents concurrently while strictly preserving input order.
    Parameters:
      documents_data: List of (ocr_text, filename) tuples.
    """
    tasks = [
        extract_document_fields_async(
            ocr_text=text,
            filename=fname,
            client_override=client_override,
            mode_override=mode_override,
        )
        for text, fname in documents_data
    ]
    return await asyncio.gather(*tasks)
