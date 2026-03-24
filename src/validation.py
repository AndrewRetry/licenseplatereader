"""
Singapore license plate validation.

Supported formats:
  Private car   : SBA 1234 L
  Taxi          : SHA 5678 H
  Private hire  : SX 9999 K
  Motorcycle    : FBA 123
  Government    : QX 1234
  Diplomatic    : D 1234
"""

import re
from dataclasses import dataclass
from typing import Optional

CHECKSUM_LETTERS = "AZYXUTSRPMLKJHGEDCB"
WEIGHTS = [9, 4, 5, 4, 3, 2]

PATTERNS = [
    # Standard: SBA1234A | S1234A | SA1234A
    re.compile(r"^([A-Z]{1,3})(\d{1,4})([A-Z])$"),
    # Motorcycle: FBA123 | FB123A
    re.compile(r"^(F[A-Z])(\d{1,4})([A-Z]?)$"),
    # Government: QX1234
    re.compile(r"^(QX)(\d{4})$"),
    # Diplomatic: D1234A
    re.compile(r"^(D)(\d{4})([A-Z]?)$"),
]

PREFIX_PADDING = {1: "  ", 2: " ", 3: ""}


@dataclass
class PlateResult:
    plate: str
    formatted: str
    prefix: str
    digits: str
    checksum: Optional[str]
    checksum_valid: Optional[bool]
    format: str
    confidence: str


def compute_checksum(prefix: str, digits: str) -> str:
    padded = (PREFIX_PADDING.get(len(prefix), "") + prefix)[-3:]
    nums = [
        *(0 if c == " " else ord(c) - 64 for c in padded),
        *map(int, digits.zfill(4)[:3]),
    ]
    total = sum(n * w for n, w in zip(nums, WEIGHTS))
    return CHECKSUM_LETTERS[total % 19]


def normalise(raw: str) -> str:
    """Clean OCR noise and apply scoped O→0 / I→1 substitutions."""
    clean = re.sub(r"[^A-Z0-9]", "", raw.upper())

    # Pre-pass: SG prefixes are at most 3 letters.  If the cleaned string
    # has O or I past position 2, they are almost certainly misread digits —
    # substitute them before the structural split so the regex sees digits.
    if len(clean) > 3:
        candidate = clean[:2] + clean[2:].replace("O", "0").replace("I", "1")
        m_pre = re.match(r"^([A-Z]{1,3})(\d+)([A-Z]?)$", candidate)
        if m_pre:
            clean = candidate

    m = re.match(r"^([A-Z]*)(\d+)([A-Z]?)$", clean)
    if not m:
        return clean

    prefix, digits, checksum = m.group(1), m.group(2), m.group(3)

    # O/I → digit only inside the digit block (belt-and-suspenders)
    fixed_digits = digits.replace("O", "0").replace("I", "1")

    # 8 → B only inside the prefix (between letters)
    fixed_prefix = re.sub(r"(?<=[A-Z])8(?=[A-Z]|$)", "B", prefix)

    return fixed_prefix + fixed_digits + checksum


def _resolve_format(prefix: str) -> str:
    if prefix == "QX":
        return "government"
    if prefix == "D":
        return "diplomatic"
    if prefix.startswith("F"):
        return "motorcycle"
    if prefix.startswith("SH"):
        return "taxi"
    if prefix.startswith("SX"):
        return "private_hire"
    return "private_car"


def _format_plate(prefix: str, digits: str, checksum: Optional[str]) -> str:
    stripped = digits.lstrip("0") or "0"
    return f"{prefix} {stripped} {checksum}" if checksum else f"{prefix} {stripped}"


def validate_plate(raw: str) -> Optional[PlateResult]:
    if not raw or len(raw) < 3:
        return None

    candidate = normalise(raw)

    for pattern in PATTERNS:
        m = pattern.match(candidate)
        if not m:
            continue

        prefix = m.group(1)
        digits = m.group(2)
        checksum = m.group(3) if m.lastindex >= 3 else None
        if checksum == "":
            checksum = None

        checksum_valid: Optional[bool] = None
        if checksum:
            checksum_valid = compute_checksum(prefix, digits) == checksum

        fmt = _resolve_format(prefix)
        confidence = (
            "high" if checksum_valid is True
            else "low" if checksum_valid is False
            else "medium"
        )

        return PlateResult(
            plate=candidate,
            formatted=_format_plate(prefix, digits, checksum),
            prefix=prefix,
            digits=digits,
            checksum=checksum,
            checksum_valid=checksum_valid,
            format=fmt,
            confidence=confidence,
        )

    return None
