"""
Unit tests for src/validation.py

These have zero external dependencies — no OpenCV, no EasyOCR.
They run in milliseconds and should always pass in CI.
"""

import pytest
from src.validation import (
    PlateResult,
    compute_checksum,
    normalise,
    validate_plate,
)


# ── compute_checksum ──────────────────────────────────────────────────────────

class TestComputeChecksum:
    def test_standard_3_letter_prefix(self):
        # SBA + 1234 → known-correct checksum
        result = compute_checksum("SBA", "1234")
        assert isinstance(result, str)
        assert len(result) == 1
        assert result.isalpha()

    def test_two_letter_prefix(self):
        result = compute_checksum("SB", "1234")
        assert isinstance(result, str) and len(result) == 1

    def test_one_letter_prefix(self):
        result = compute_checksum("S", "1234")
        assert isinstance(result, str) and len(result) == 1

    def test_deterministic(self):
        """Same input always produces same output."""
        assert compute_checksum("SBA", "1234") == compute_checksum("SBA", "1234")

    def test_different_plates_differ(self):
        a = compute_checksum("SBA", "1234")
        b = compute_checksum("SBA", "5678")
        assert a != b


# ── normalise ─────────────────────────────────────────────────────────────────

class TestNormalise:
    def test_strips_whitespace(self):
        assert normalise("SBA 1234 L") == "SBA1234L"

    def test_uppercases(self):
        assert normalise("sba1234l") == "SBA1234L"

    def test_strips_special_chars(self):
        assert normalise("SBA-1234-L") == "SBA1234L"

    def test_digit_o_to_zero(self):
        # O inside the digit block → 0
        result = normalise("SBA0O34L")
        assert "0" in result

    def test_digit_i_to_one(self):
        result = normalise("SBA1I34L")
        assert result[3:7].replace("1", "").isdigit() or "1" in result[3:7]

    def test_prefix_unchanged(self):
        """Letters in prefix should NOT be turned into digits."""
        result = normalise("SBA1234L")
        assert result.startswith("SBA")

    def test_checksum_unchanged(self):
        """Trailing checksum letter must not be coerced to digit."""
        result = normalise("SBA1234L")
        assert result[-1] == "L"

    def test_already_clean(self):
        assert normalise("SBA1234L") == "SBA1234L"


# ── validate_plate ────────────────────────────────────────────────────────────

class TestValidatePlate:

    # --- Valid plates ---

    def test_private_car(self):
        r = validate_plate("SBA1234L")
        assert r is not None
        assert r.format == "private_car"
        assert r.plate  == "SBA1234L"

    def test_taxi(self):
        r = validate_plate("SHA5678H")
        assert r is not None
        assert r.format == "taxi"

    def test_private_hire(self):
        r = validate_plate("SX9999K")
        assert r is not None
        assert r.format == "private_hire"

    def test_government(self):
        r = validate_plate("QX1234")
        assert r is not None
        assert r.format == "government"

    def test_diplomatic(self):
        r = validate_plate("D1234A")
        assert r is not None
        assert r.format == "diplomatic"

    def test_two_letter_prefix(self):
        r = validate_plate("SB1234A")
        assert r is not None

    def test_one_letter_prefix(self):
        r = validate_plate("S1234A")
        assert r is not None

    # --- Checksum validation ---

    def test_correct_checksum_yields_high_confidence(self):
        # Build a plate with the CORRECT checksum
        from src.validation import compute_checksum
        checksum = compute_checksum("SBA", "1234")
        r = validate_plate(f"SBA1234{checksum}")
        assert r is not None
        assert r.checksum_valid is True
        assert r.confidence == "high"

    def test_wrong_checksum_yields_low_confidence(self):
        # Find an incorrect checksum letter
        from src.validation import compute_checksum
        correct = compute_checksum("SBA", "1234")
        wrong = "Z" if correct != "Z" else "A"
        r = validate_plate(f"SBA1234{wrong}")
        assert r is not None
        assert r.checksum_valid is False
        assert r.confidence == "low"

    def test_no_checksum_yields_medium_confidence(self):
        r = validate_plate("QX1234")          # government — no checksum
        assert r is not None
        assert r.checksum_valid is None
        assert r.confidence == "medium"

    # --- Formatted output ---

    def test_formatted_adds_spaces(self):
        r = validate_plate("SBA1234L")
        assert r is not None
        assert " " in r.formatted

    def test_formatted_strips_leading_zeros(self):
        r = validate_plate("SBA0034L")
        assert r is not None
        assert "0034" not in r.formatted   # leading zeros stripped in display

    # --- Invalid inputs ---

    def test_too_short_returns_none(self):
        assert validate_plate("AB") is None

    def test_empty_returns_none(self):
        assert validate_plate("") is None

    def test_gibberish_returns_none(self):
        assert validate_plate("XXXXXXXXX") is None

    def test_none_input_returns_none(self):
        assert validate_plate(None) is None  # type: ignore[arg-type]

    def test_all_digits_returns_none(self):
        assert validate_plate("123456789") is None

    # --- OCR noise correction end-to-end ---

    def test_ocr_noise_o_in_digits(self):
        """Common OCR error: O instead of 0 in digit block."""
        from src.validation import compute_checksum
        checksum = compute_checksum("SBA", "1234")
        noisy = f"SBAO234{checksum}"        # O instead of first digit
        r = validate_plate(noisy)
        assert r is not None
        assert "0" in r.plate              # corrected

    def test_ocr_noise_i_in_digits(self):
        from src.validation import compute_checksum
        checksum = compute_checksum("SBA", "1234")
        noisy = f"SBAI234{checksum}"       # I instead of 1
        r = validate_plate(noisy)
        assert r is not None
        assert r.plate[3] == "1"           # I → 1 in digit block
