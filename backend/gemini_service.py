"""
gemini_service.py - VerifyAI Gemini LLM Primary Extraction & Intelligence Service.

Responsible for:
  - Primary LLM-based semantic extraction from OCR text
  - Multilingual label understanding (English + Gujarati)
  - Anti-hallucination enforcement & field-level confidence scoring
  - Optional semantic cross-document comparison
  - Factual verification explanation generation
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
CONFIGURED_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# Ordered model candidates for fallback if configured model returns 404/deprecation
MODEL_CANDIDATES = list(dict.fromkeys([
    CONFIGURED_MODEL,
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]))

_client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    try:
        _client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"[VerifyAI] Gemini Client active. Primary model: {CONFIGURED_MODEL}")
    except Exception as e:
        print(f"[VerifyAI] ERROR initializing Gemini Client: {e}", file=sys.stderr)
        _client = None
else:
    print("[VerifyAI] WARNING: GEMINI_API_KEY not found in environment.", file=sys.stderr)


def is_gemini_available() -> bool:
    """Return True if Gemini client is successfully initialized and has an API key."""
    return _client is not None and bool(GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# STRUCTURED SCHEMAS FOR STRICT EXTRACTION
# ---------------------------------------------------------------------------

class FieldConfidence(BaseModel):
    document_type: float = 0.0
    name: float = 0.0
    dob: float = 0.0
    gender: float = 0.0
    address: float = 0.0
    pincode: float = 0.0
    income: float = 0.0
    aadhaar_number: float = 0.0
    pan_number: float = 0.0
    seat_number: float = 0.0
    roll_number: float = 0.0
    sid_number: float = 0.0
    statement_number: float = 0.0
    total_marks: float = 0.0
    obtained_marks: float = 0.0


class StructuredExtractionOutput(BaseModel):
    document_type: str = Field(
        description="One of: Aadhaar, PAN Card, Voter ID, Driving Licence, Passport, "
                    "Income Certificate, Caste Certificate, Domicile Certificate, Birth Certificate, "
                    "Marksheet, Bank Statement, Ration Card, Scholarship Application, Invoice, or Unknown"
    )
    name: Optional[str] = Field(default=None, description="Full legal name of candidate / holder / student")
    father_name: Optional[str] = None
    mother_name: Optional[str] = None
    spouse_name: Optional[str] = None
    relation: Optional[str] = None
    dob: Optional[str] = Field(default=None, description="Date of birth in DD/MM/YYYY format")
    age: Optional[str] = None
    gender: Optional[str] = Field(default=None, description="Male, Female, or Transgender")
    nationality: Optional[str] = None

    address: Optional[str] = None
    village: Optional[str] = None
    taluka: Optional[str] = None
    district: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = Field(default=None, description="Postal 6-digit pincode only. Never use seat/roll numbers.")

    phone: Optional[str] = None
    email: Optional[str] = None

    aadhaar_number: Optional[str] = None
    aadhaar_masked: Optional[str] = None
    pan_number: Optional[str] = None
    voter_id: Optional[str] = None
    driving_licence_number: Optional[str] = None
    passport_number: Optional[str] = None

    certificate_number: Optional[str] = None
    application_number: Optional[str] = None
    registration_number: Optional[str] = None
    reference_number: Optional[str] = None

    income: Optional[str] = Field(default=None, description="Annual/total numeric income (e.g. 150000)")
    income_period: Optional[str] = None
    income_in_words: Optional[str] = None

    institution: Optional[str] = None
    board: Optional[str] = None
    course: Optional[str] = None
    stream: Optional[str] = None
    exam_year: Optional[str] = None

    roll_number: Optional[str] = None
    seat_number: Optional[str] = None
    sid_number: Optional[str] = None
    statement_number: Optional[str] = None
    enrollment_number: Optional[str] = None

    obtained_marks: Optional[str] = Field(default=None, description="Marks obtained by student")
    total_marks: Optional[str] = Field(default=None, description="Maximum possible marks")
    percentage: Optional[str] = None
    cgpa: Optional[str] = None
    grade: Optional[str] = None
    result: Optional[str] = None

    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    invoice_amount: Optional[str] = None

    issue_date: Optional[str] = None
    application_date: Optional[str] = None
    valid_until: Optional[str] = None
    issuing_authority: Optional[str] = None
    place: Optional[str] = None
    caste: Optional[str] = None
    category: Optional[str] = None

    all_dates: List[str] = Field(default_factory=list)
    confidence: Dict[str, float] = Field(default_factory=dict)
    extraction_notes: List[str] = Field(default_factory=list)


class SemanticComparisonOutput(BaseModel):
    semantic_assessment: str = Field(description="LIKELY_CONSISTENT, POTENTIAL_INCONSISTENCY, or INCONCLUSIVE")
    confidence: float
    observations: List[str]


# ---------------------------------------------------------------------------
# RESILIENT EXECUTION WRAPPER
# ---------------------------------------------------------------------------

def _execute_gemini_call(prompt: str, schema: Optional[Any] = None, temperature: float = 0.0) -> Optional[str]:
    """Execute Gemini request with automatic failover across valid model endpoints."""
    if not is_gemini_available():
        return None

    config_params: Dict[str, Any] = {"temperature": temperature}
    if schema:
        config_params["response_mime_type"] = "application/json"
        config_params["response_schema"] = schema

    config = types.GenerateContentConfig(**config_params)

    for model_name in MODEL_CANDIDATES:
        try:
            response = _client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            if response and response.text:
                return response.text
        except APIError as e:
            err_str = str(e)
            code = getattr(e, "code", None)
            if code == 404 or "404" in err_str or code == 429 or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                print(f"[VerifyAI] Model {model_name} unavailable ({code or 'quota'}). Failing over to next candidate...", file=sys.stderr)
                continue
            print(f"[VerifyAI] Gemini API error ({model_name}): {e}", file=sys.stderr)
            break
        except Exception as e:
            err_str = str(e)
            if "404" in err_str or "quota" in err_str.lower() or "429" in err_str or "resource_exhausted" in err_str.lower():
                print(f"[VerifyAI] Model {model_name} unavailable. Failing over to next candidate...", file=sys.stderr)
                continue
            print(f"[VerifyAI] Unexpected error calling Gemini ({model_name}): {e}", file=sys.stderr)
            break

    return None


# ---------------------------------------------------------------------------
# 1. PRIMARY LLM-BASED EXTRACTION
# ---------------------------------------------------------------------------

def extract_document_with_gemini(ocr_text: str, filename: str = "") -> Optional[Dict[str, Any]]:
    """
    Primary semantic extractor. Reads noisy, layout-corrupted OCR text in English and Gujarati
    and extracts structured JSON conforming to StructuredExtractionOutput.
    Never invents data.
    """
    if not is_gemini_available() or not ocr_text or not ocr_text.strip():
        return None

    prompt = f"""You are the primary document extraction engine for VerifyAI, an official document verification platform in India.
Your task is to accurately extract identity and certificate fields from the OCR text of an uploaded document.

SOURCE FILENAME: {filename}

RAW OCR TEXT (May contain character errors, broken labels, multi-column noise, and mixed English/Gujarati):
\"\"\"
{ocr_text}
\"\"\"

STRICT ANTI-HALLUCINATION & EXTRACTION RULES:
1. DO NOT INVENT INFORMATION. If a field is not present or cannot be reliably determined, return null.
2. OCR CORRUPTIONS: Understand misspelled or compressed labels semantically:
   - "Canddate'sName", "Candidate's Name", "Candidate Name", "Student Name", "ઉમેદવારનું નામ" -> name
   - "Father's Name", "Pita ka Naam", "પિતાનું નામ" -> father_name
   - "Mother's Name", "Mata ka Naam", "માતાનું નામ" -> mother_name
   - "DOB", "D.O.B", "Birth Date", "જન્મ તારીખ", "Year of Birth" -> dob
   - "Gender", "Sex", "પુરુષ/સ્ત્રી" -> gender
   - "Seat No", "બેઠક ક્રમાંક" -> seat_number
   - "Roll No", "રોલ નંબર" -> roll_number
   - "S.I.D. No", "SID" -> sid_number
   - "Sr. No. of Statement", "Statement No", "પત્રક અનુક્રમ" -> statement_number
   - "Annual Income", "Gross Income", "વાર્ષિક આવક" -> income
3. CANDIDATE NAME INTEGRITY:
   - Extract the full name accurately as it appears.
   - Do NOT confuse student name with subjects, words like "Zero Five Seven", or adjacent column IDs.
   - Strip stray alphanumeric IDs like "24V-137803" if they were mistakenly merged on the same line.
4. MARKSHEETS VS IDENTITY DOCUMENTS:
   - For Marksheets, correctly identify obtain/total marks (e.g. 413/500 -> obtained_marks: "413", total_marks: "500").
   - Marksheets DO NOT have postal pincodes. Never label Seat numbers (e.g. "B 244319") as pincodes.
5. DOCUMENT TYPE IDENTIFICATION:
   - Select strictly from: Aadhaar, PAN Card, Voter ID, Driving Licence, Passport, Income Certificate, Caste Certificate, Domicile Certificate, Birth Certificate, Marksheet, Bank Statement, Ration Card, Scholarship Application, Invoice, or Unknown.
6. CONFIDENCE & NOTES:
   - For each extracted field, provide a confidence float between 0.0 and 1.0 based on how clear and uncorrupted the OCR was.
   - Provide concise notes in `extraction_notes` if OCR anomalies were resolved.
"""

    response_text = _execute_gemini_call(
        prompt=prompt,
        schema=StructuredExtractionOutput,
        temperature=0.0
    )

    if response_text:
        try:
            parsed = json.loads(response_text)
            parsed["_extraction_source"] = "llm"
            print(f"[VerifyAI] LLM extraction succeeded for {filename}. Type: {parsed.get('document_type')}, Name: {parsed.get('name')}")
            return parsed
        except Exception as e:
            print(f"[VerifyAI] Failed to parse LLM structured output for {filename}: {e}", file=sys.stderr)

    return None


# ---------------------------------------------------------------------------
# 2. SEMANTIC CROSS-DOCUMENT COMPARISON
# ---------------------------------------------------------------------------

def semantic_compare_documents(
    documents: List[Dict[str, Any]],
    deterministic_comparison: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Evaluates cross-document inconsistencies (e.g. name token permutations,
    initial expansions) with semantic context without altering deterministic results.
    """
    if not is_gemini_available() or not documents or len(documents) < 2:
        return None

    # Redact sensitive IDs for prompt hygiene
    sanitized_docs = []
    for doc in documents:
        sd = {k: v for k, v in doc.items() if not k.startswith("_") and k != "ocr_text"}
        if sd.get("aadhaar_number"):
            sd["aadhaar_number"] = "[Aadhaar Redacted]"
        sanitized_docs.append(sd)

    prompt = f"""You are the semantic comparison engine for VerifyAI.
Review these extracted documents alongside the deterministic comparison results.

Rules:
1. Provide semantic context: distinguish harmless ordering/initial variations from conflicting identities.
2. Note exact or material mismatches as potential inconsistencies.
3. Do NOT make definitive legal accusations or state that documents are fake.

Extracted Documents:
{json.dumps(sanitized_docs, indent=2, default=str)}

Deterministic Comparison Results:
{json.dumps(deterministic_comparison, indent=2, default=str)}
"""

    response_text = _execute_gemini_call(
        prompt=prompt,
        schema=SemanticComparisonOutput,
        temperature=0.1
    )

    if response_text:
        try:
            return json.loads(response_text)
        except Exception as e:
            print(f"[VerifyAI] Failed to parse semantic comparison: {e}", file=sys.stderr)

    return None


# ---------------------------------------------------------------------------
# 3. VERIFICATION EXPLANATION GENERATION
# ---------------------------------------------------------------------------

def generate_verification_explanation(
    documents: List[Dict[str, Any]],
    comparison: Dict[str, Any],
    risk: Dict[str, Any],
    semantic: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generates an objective, non-accusatory summary explanation of findings.
    Never declares a document 'fake' or claims 'fraud'.
    """
    fallback_reasons = risk.get("reasons", [])
    recommendation = risk.get("recommendation", "")
    default_explanation = (
        f"{' '.join(fallback_reasons)} {recommendation}".strip()
        or "Verification completed using deterministic rule checks."
    )

    if not is_gemini_available():
        return default_explanation

    prompt = f"""You are the verification report writer for VerifyAI.
Generate a concise, objective summary (2 to 4 sentences) explaining the document verification findings for a reviewer.

STRICT TONE REQUIREMENTS:
- State only factual discrepancies detected across documents.
- Use cautious, professional terminology: "potential inconsistency", "verification concern", "manual verification recommended".
- NEVER say "this document is fake", "fraud occurred", or cite a fraud probability.

Verification Data:
Risk Level: {risk.get('risk_level')}
Risk Score: {risk.get('risk_score')}/100
Risk Reasons: {json.dumps(risk.get('reasons', []))}
Semantic Context: {json.dumps(semantic or {})}
Comparison Summary: Matches={comparison.get('total_matches')}, Partial={comparison.get('total_partial_matches')}, Mismatches={comparison.get('total_mismatches')}
"""

    response_text = _execute_gemini_call(prompt=prompt, schema=None, temperature=0.2)
    return response_text.strip() if response_text else default_explanation