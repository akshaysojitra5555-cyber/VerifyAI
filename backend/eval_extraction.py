"""
eval_extraction.py - Benchmark and Evaluation Harness for VerifyAI Extraction Engines.

Compares:
  1. Pure Rules-Based Extraction (legacy regex extractor.py)
  2. Pure LLM-Based Extraction (Gemini 2.5 Flash with strict anti-hallucination grounding)
  3. Hybrid Pipeline (LLM primary + high-precision deterministic fallback)

Evaluates per-field accuracy against expected JSON ground truth files.
Generates an ASCII comparison table suitable for project presentations and viva questions.

Usage:
  python eval_extraction.py
  python eval_extraction.py --eval-dir path/to/dataset
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

# Reconfigure stdout for UTF-8 so Indic/Gujarati text displays properly on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from extraction_service import extract_document_fields
from extraction_validation import normalize_name_for_compare, normalize_date_string


DEFAULT_BENCHMARK_SAMPLES = [
    {
        "id": "sample_01_marksheet",
        "ocr_text": (
            "GUJARAT SECONDARY AND HIGHER SECONDARY EDUCATION BOARD, GANDHINAGAR\n"
            "STATEMENT OF MARKS\n"
            "Canddate'sName, SOUJITRA AKSHAY SURESHBHAI\n"
            "SEAT NO. B 244319\n"
            "SID.No. 24V-137803\n"
            "SR.NO. OF STATEMENT 433501\n"
            "MONTH & YEAR OF THE EXAM. MAR-2024\n"
            "THEORY TOTAL 413/500\n"
            "Dt: 09/05/2024"
        ),
        "expected": {
            "document_type": "Marksheet",
            "name": "SOUJITRA AKSHAY SURESHBHAI",
            "seat_number": "B 244319",
            "sid_number": "24V-137803",
            "statement_number": "433501",
            "marks_obtained": "413",
            "obtained_marks": 413,
            "max_marks": "500",
            "total_marks": 500,
            "exam_year": "2024",
            "exam_date": "09/05/2024",
            "pincode": None,
        }
    },
    {
        "id": "sample_02_income_cert_gujarati",
        "ocr_text": (
            "મામલતદાર કચેરી, અમરેલી\n"
            "આવકનું પ્રમાણપત્ર\n"
            "પ્રમાણપત્ર નંબર: INC/2024/98412\n"
            "આથી પ્રમાણપત્ર આપવામાં આવે છે કે પ્રિયંકા મહેશભાઈ શાહ\n"
            "પિતાનું નામ: મહેશભાઈ શાહ\n"
            "ગામ: રામપુર, તાલુકો: ગોંડલ, જિલ્લો: રાજકોટ\n"
            "વાર્ષિક આવક: રૂ. ૧,૫૦,૦૦૦/-\n"
            "તારીખ: 15/06/2024"
        ),
        "expected": {
            "document_type": "Income Certificate",
            "name": "પ્રિયંકા મહેશભાઈ શાહ",
            "father_name": "મહેશભાઈ શાહ",
            "village": "રામપુર",
            "taluka": "ગોંડલ",
            "district": "રાજકોટ",
            "income": "150000",
            "certificate_number": "INC/2024/98412",
            "issue_date": "15/06/2024"
        }
    },
    {
        "id": "sample_03_aadhaar_card",
        "ocr_text": (
            "GOVERNMENT OF INDIA\n"
            "UNIQUE IDENTIFICATION AUTHORITY OF INDIA\n"
            "Enrollment No: 1024/99120/12345\n"
            "To, PRIYA PATEL\n"
            "DOB: 12/04/2002\n"
            "Female\n"
            "Address: 42, Anand Nagar, Satellite, Ahmedabad, Gujarat - 380015\n"
            "2182 0986 7545"
        ),
        "expected": {
            "document_type": "Aadhaar",
            "name": "PRIYA PATEL",
            "dob": "12/04/2002",
            "gender": "Female",
            "pincode": "380015",
            "state": "Gujarat",
            "aadhaar_number": "2182 0986 7545",
            "aadhaar_checksum_valid": True,
        }
    },
    {
        "id": "sample_04_pan_card",
        "ocr_text": (
            "INCOME TAX DEPARTMENT\n"
            "GOVT. OF INDIA\n"
            "Permanent Account Number Card\n"
            "ABCDE1234F\n"
            "Name: RAJESH KUMAR SHAH\n"
            "Father's Name: DINESHBHAI SHAH\n"
            "Date of Birth: 25/11/1988"
        ),
        "expected": {
            "document_type": "PAN Card",
            "pan_number": "ABCDE1234F",
            "name": "RAJESH KUMAR SHAH",
            "father_name": "DINESHBHAI SHAH",
            "dob": "25/11/1988",
        }
    }
]


def load_dataset_from_directory(data_dir: Path) -> List[Dict[str, Any]]:
    """Loads OCR .txt and ground-truth .json pairs from directory."""
    samples = []
    txt_files = list(data_dir.glob("*.txt"))
    for tf in txt_files:
        jf = tf.with_suffix(".json")
        if jf.exists():
            ocr_content = tf.read_text(encoding="utf-8")
            expected_data = json.loads(jf.read_text(encoding="utf-8"))
            samples.append({
                "id": tf.stem,
                "ocr_text": ocr_content,
                "expected": expected_data,
            })
    return samples


def are_values_equivalent(field: str, extracted: Any, expected: Any) -> bool:
    """Compares extracted and ground truth values with smart semantic tolerance."""
    if extracted is None and expected is None:
        return True
    if extracted is None or expected is None:
        return False

    # Name comparison: order and case-insensitive check
    if field in ("name", "father_name", "mother_name", "spouse_name"):
        norm_ext = normalize_name_for_compare(extracted)
        norm_exp = normalize_name_for_compare(expected)
        return norm_ext == norm_exp

    # Date comparison: DD/MM/YYYY equivalence
    if "date" in field or field == "dob":
        d_ext = normalize_date_string(extracted)
        d_exp = normalize_date_string(expected)
        return d_ext == d_exp

    # String comparison (strip and casefold)
    str_ext = str(extracted).strip().replace(" ", "").casefold()
    str_exp = str(expected).strip().replace(" ", "").casefold()
    return str_ext == str_exp


def evaluate_dataset(samples: List[Dict[str, Any]]) -> None:
    """Executes evaluation across Rules, LLM, and Hybrid engines and displays metrics."""
    modes = ["rules", "llm", "hybrid"]
    field_totals: Dict[str, int] = {}
    field_correct: Dict[str, Dict[str, int]] = {m: {} for m in modes}
    latencies: Dict[str, float] = {m: 0.0 for m in modes}

    print("=" * 80)
    print(" VerifyAI Document Extraction Evaluation Harness")
    print(" Benchmark Evaluation: Rules vs LLM vs Hybrid")
    print(f" Total Evaluation Samples: {len(samples)}")
    print("=" * 80)

    for sample in samples:
        ocr_text = sample["ocr_text"]
        expected = sample["expected"]

        for f, exp_val in expected.items():
            if exp_val is not None:
                field_totals[f] = field_totals.get(f, 0) + 1

        for mode in modes:
            t0 = time.time()
            extracted = extract_document_fields(ocr_text, filename=sample["id"], mode_override=mode)
            dur = time.time() - t0
            latencies[mode] += dur

            for f, exp_val in expected.items():
                if exp_val is None:
                    continue
                actual_val = extracted.get(f)
                if are_values_equivalent(f, actual_val, exp_val):
                    field_correct[mode][f] = field_correct[mode].get(f, 0) + 1

    # Print summary table
    print("\n" + "-" * 80)
    print(f"{'Field':<28} | {'Rules (%)':<12} | {'LLM (%)':<12} | {'Hybrid (%)':<12} | Count")
    print("-" * 80)

    all_fields = sorted(field_totals.keys())
    total_expected_checks = sum(field_totals.values())
    total_correct_checks: Dict[str, int] = {m: sum(field_correct[m].values()) for m in modes}

    for f in all_fields:
        count = field_totals[f]
        r_pct = (field_correct["rules"].get(f, 0) / count) * 100.0
        l_pct = (field_correct["llm"].get(f, 0) / count) * 100.0
        h_pct = (field_correct["hybrid"].get(f, 0) / count) * 100.0
        print(f"{f:<28} | {r_pct:>10.1f}% | {l_pct:>10.1f}% | {h_pct:>10.1f}% | {count:>5}")

    print("=" * 80)
    for mode in modes:
        overall_pct = (total_correct_checks[mode] / total_expected_checks) * 100.0 if total_expected_checks else 0.0
        avg_lat = (latencies[mode] / len(samples)) if samples else 0.0
        print(f"Overall Accuracy [{mode.upper():<6}]: {overall_pct:6.2f}% | Avg Latency: {avg_lat:.3f}s / doc")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="VerifyAI Document Extraction Evaluation Harness")
    parser.add_argument("--eval-dir", type=str, default="", help="Path to folder containing .txt and .json files")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir) if args.eval_dir else BASE_DIR / "eval_data"

    if eval_dir.exists() and list(eval_dir.glob("*.txt")):
        print(f"[VerifyAI] Loading custom evaluation dataset from: {eval_dir}")
        samples = load_dataset_from_directory(eval_dir)
    else:
        print("[VerifyAI] Using built-in representative document samples (Marksheet, Income, Aadhaar, PAN)")
        # Automatically generate eval_data directory and files so they are persisted on disk
        eval_dir.mkdir(parents=True, exist_ok=True)
        for s in DEFAULT_BENCHMARK_SAMPLES:
            txt_file = eval_dir / f"{s['id']}.txt"
            json_file = eval_dir / f"{s['id']}.json"
            if not txt_file.exists():
                txt_file.write_text(s["ocr_text"], encoding="utf-8")
            if not json_file.exists():
                json_file.write_text(json.dumps(s["expected"], indent=2, ensure_ascii=False), encoding="utf-8")
        samples = DEFAULT_BENCHMARK_SAMPLES

    evaluate_dataset(samples)


if __name__ == "__main__":
    main()
