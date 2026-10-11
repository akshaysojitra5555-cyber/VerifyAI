"""
test_hybrid_pipeline.py - Verification test suite for VerifyAI hybrid pipeline.
"""
import os
import unittest
from extractor import extract_fields
from comparison import compare_all_documents, compare_name
from risk_engine import calculate_risk
from gemini_service import (
    improve_extraction_with_gemini,
    merge_rule_and_gemini_extraction,
    generate_verification_explanation,
    is_gemini_available,
)

class TestVerifyAIHybridPipeline(unittest.TestCase):

    def test_1_marksheet_extraction(self):
        """Test extraction on Gujarati marksheet OCR sample."""
        ocr_text = """
        Candidate's Name, SOJITRA AKSHAY SURESHBHAI
        MONTH & YEAR OF THE EXAM. MAR-2024
        SEAT NO. B 244319
        SID.No. 24V-1937803
        SR.NO. OF STATEMENT 433501
        TOTAL MARKS 413/500
        """
        rule_data = extract_fields(ocr_text)

        # Verify deterministic extraction handles seat, statement, marks
        self.assertEqual(rule_data.get("name"), "Sojitra Akshay Sureshbhai")
        self.assertEqual(rule_data.get("seat_number"), "B 244319")
        self.assertEqual(rule_data.get("sid_number"), "24V-1937803")
        self.assertEqual(rule_data.get("statement_number"), "433501")
        self.assertEqual(rule_data.get("obtained_marks"), "413")
        self.assertEqual(rule_data.get("total_marks"), "500")

        # Hybrid merge test
        gemini_mock = {
            "name": "SOJITRA AKSHAY SURESHBHAI",
            "seat_number": "B 244319",
            "sid_number": "24V-1937803",
            "statement_number": "433501",
            "obtained_marks": "413",
            "total_marks": "500",
        }
        merged = merge_rule_and_gemini_extraction(rule_data, gemini_mock)
        self.assertEqual(merged["seat_number"], "B 244319")
        self.assertEqual(merged["_extraction_source"], "hybrid")

    def test_2_income_certificate_extraction(self):
        """Test extraction of income certificate values."""
        sample = """
        Office of the Mamlatdar, Amreli
        Income Certificate
        This is to certify that Akshay Sureshbhai Sojitra
        Annual Income: Rs. 1,50,000/-
        Certificate No: INC/2026/8942
        Date: 12/06/2026
        """
        rule_data = extract_fields(sample)
        self.assertEqual(rule_data.get("income"), "150000")
        self.assertEqual(rule_data.get("certificate_number"), "INC/2026/8942")
        self.assertEqual(rule_data.get("document_type"), "Income Certificate")

    def test_3_gemini_failure_fallback(self):
        """Test that missing or None Gemini data safely falls back to rule data."""
        rule_data = {
            "name": "Akshay Patel",
            "dob": "02/12/2006",
            "gender": "Male",
            "document_type": "Aadhaar"
        }
        # Simulate complete Gemini failure / None return
        merged = merge_rule_and_gemini_extraction(rule_data, None)
        self.assertEqual(merged["name"], "Akshay Patel")
        self.assertEqual(merged["_extraction_source"], "rule_based")

    def test_4_deterministic_comparison_intact(self):
        """Test deterministic comparison logic."""
        res = compare_name("Akshay Sojitra", "Sojitra Akshay Sureshbhai")
        self.assertIn(res["status"], ["MATCH", "PARTIAL_MATCH"])

    def test_5_risk_engine_intact(self):
        """Ensure risk engine computes authoritative scores without mutation."""
        comparison_mock = {
            "documents_compared": 2,
            "pair_comparisons": [{
                "document_1": "Aadhaar",
                "document_2": "Marksheet",
                "field_comparisons": [
                    {"field": "name", "status": "MISMATCH"},
                    {"field": "dob", "status": "MISMATCH"},
                ]
            }]
        }
        risk = calculate_risk(comparison_mock)
        # Name (30) + DOB (40) = 70 -> HIGH
        self.assertEqual(risk["risk_score"], 70)
        self.assertEqual(risk["risk_level"], "HIGH")

if __name__ == "__main__":
    unittest.main()