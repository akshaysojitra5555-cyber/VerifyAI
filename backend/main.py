"""
VerifyAI Backend - FastAPI Application.
AI-Powered Multi-Document Fraud Detection & Verification System.

Pipeline:
    Upload Documents
        ↓
    Tesseract OCR
        ↓
    Gemini LLM Primary Semantic Extraction
        ↓
    Validation / Normalization
        ↓
    Deterministic Multi-Document Comparison
        ↓
    Gemini Semantic Comparison (Optional Layer)
        ↓
    Deterministic Risk Engine
        ↓
    Gemini Verification Explanation
        ↓
    Final Verification Report
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ocr import extract_text
from extraction_service import extract_document_fields
from comparison import compare_all_documents
from risk_engine import calculate_risk
import time
from gemini_service import (
    extract_document_with_gemini,
    semantic_compare_documents,
    generate_verification_explanation,
    is_gemini_available,
    CONFIGURED_MODEL,
)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILES = 5
MAX_SIZE = 10 * 1024 * 1024  # 10 MB per file

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

# ------------------------------------------------------------
# FASTAPI APPLICATION
# ------------------------------------------------------------

app = FastAPI(
    title="VerifyAI Backend",
    description="AI-Powered Multi-Document Fraud Detection & Verification System",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "message": "VerifyAI backend is running",
        "gemini_available": is_gemini_available(),
        "gemini_model": CONFIGURED_MODEL if is_gemini_available() else None,
    }


def _validate_file(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename}: Only PDF, JPG, JPEG, and PNG files are supported."
        )


# ------------------------------------------------------------
# MAIN MULTI-DOCUMENT VERIFICATION ENDPOINT
# ------------------------------------------------------------

@app.post("/api/compare-documents")
async def compare_documents(
    file1: Optional[UploadFile] = File(None),
    file2: Optional[UploadFile] = File(None),
    file3: Optional[UploadFile] = File(None),
    file4: Optional[UploadFile] = File(None),
    file5: Optional[UploadFile] = File(None),
):
    raw_files = [file1, file2, file3, file4, file5]
    uploaded_files: List[UploadFile] = [f for f in raw_files if f is not None]

    if not uploaded_files:
        raise HTTPException(status_code=400, detail="Please upload at least one document.")

    if len(uploaded_files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} documents allowed.")

    documents: List[Dict[str, Any]] = []
    gemini_active = is_gemini_available()

    for file in uploaded_files:
        _validate_file(file)

        try:
            data = await file.read(MAX_SIZE + 1)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not read {file.filename}: {str(e)}")

        if len(data) > MAX_SIZE:
            raise HTTPException(status_code=413, detail=f"{file.filename} exceeds 10 MB limit.")

        # Save locally
        try:
            (UPLOAD_DIR / file.filename).write_bytes(data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not save {file.filename}: {str(e)}")

        # Step 1: OCR
        try:
            ocr_text = extract_text(file.filename, data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR failed for {file.filename}: {str(e)}")

        # Step 2: Extraction Layer (LLM Primary, Anti-Hallucination Grounding & Hybrid Fallback)
        final_fields = extract_document_fields(ocr_text, filename=file.filename)
        final_fields["file_size"] = len(data)
        final_fields["ocr_text"] = ocr_text

        documents.append(final_fields)

    # Step 4: Deterministic Multi-Document Comparison (Authoritative Baseline)
    try:
        comparison_results = compare_all_documents(documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison engine failed: {str(e)}")

    # Step 5: Optional Gemini Semantic Comparison Layer
    semantic_results = None
    if gemini_active:
        try:
            semantic_results = semantic_compare_documents(documents, comparison_results)
        except Exception as e:
            print(f"[VerifyAI] Semantic comparison error: {e}")
            semantic_results = None

    # Step 6: Deterministic Risk Engine (Authoritative Risk Score)
    try:
        risk_results = calculate_risk(comparison_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk engine failed: {str(e)}")

    # Step 7: Verification Report Explanation
    try:
        explanation = generate_verification_explanation(
            documents=documents,
            comparison=comparison_results,
            risk=risk_results,
            semantic=semantic_results,
        )
    except Exception as e:
        print(f"[VerifyAI] Explanation generation error: {e}")
        explanation = "Verification completed using deterministic document comparison and risk analysis."

    return {
        "success": True,
        "message": "Documents verified successfully.",
        "gemini": {
            "available": gemini_active,
            "model": CONFIGURED_MODEL if gemini_active else None,
            "extraction_used": any(d.get("extraction_method") == "llm" for d in documents),
            "semantic_comparison_used": semantic_results is not None,
        },
        "documents": documents,
        "comparison": comparison_results,
        "risk": risk_results,
        "semantic_comparison": semantic_results,
        "explanation": explanation,
    }


# ------------------------------------------------------------
# SINGLE-DOCUMENT UPLOAD / EXTRACTION ENDPOINT
# ------------------------------------------------------------

@app.post("/api/upload")
async def upload_single_document(file: UploadFile = File(...)):
    _validate_file(file)

    data = await file.read(MAX_SIZE + 1)
    if len(data) > MAX_SIZE:
        raise HTTPException(status_code=413, detail=f"{file.filename} exceeds 10 MB limit.")

    try:
        (UPLOAD_DIR / file.filename).write_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save {file.filename}: {str(e)}")

    try:
        ocr_text = extract_text(file.filename, data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed for {file.filename}: {str(e)}")

    final_fields = extract_document_fields(ocr_text, filename=file.filename)
    extraction_method = final_fields.get("extraction_method", "hybrid")

    return {
        "success": True,
        "filename": file.filename,
        "extraction_method": extraction_method,
        "gemini_available": gemini_active,
        "extracted_fields": final_fields,
        "ocr_text": ocr_text,
    }


# ------------------------------------------------------------
# FRONTEND STATIC FILES MOUNT
# ------------------------------------------------------------

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {
            "project": "VerifyAI",
            "description": "AI-Powered Multi-Document Fraud Detection & Verification System",
            "status": "running",
        }