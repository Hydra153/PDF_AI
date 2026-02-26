"""
Field Validators

Validates and normalizes extracted field values.
Returns (is_valid, normalized_value, error_message).

Usage:
    from validators import validate_field
    
    is_valid, normalized, error = validate_field("01/02/1990", "Date")
"""

import re
from datetime import datetime
from typing import Tuple, Optional


def validate_date(value: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate and normalize date to ISO format.
    Returns (is_valid, normalized_value, error).
    """
    if not value or value.lower() in ("none", "n/a", ""):
        return (False, value, "Empty date")
    
    value = value.strip()
    
    # Date formats — US (MM/DD) first, no ambiguous DD/MM numeric formats
    formats = [
        "%m/%d/%Y",      # 01/02/2024  (US standard)
        "%Y-%m-%d",      # 2024-01-02  (ISO)
        "%m-%d-%Y",      # 01-02-2024
        "%m.%d.%Y",      # 01.02.2024
        "%B %d, %Y",     # January 02, 2024  (unambiguous)
        "%b %d, %Y",     # Jan 02, 2024      (unambiguous)
        "%d %B %Y",      # 02 January 2024   (unambiguous)
        "%d %b %Y",      # 02 Jan 2024       (unambiguous)
        "%m/%d/%y",      # 01/02/24
        "%Y/%m/%d",      # 2024/01/02
    ]
    
    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt)
            # Basic sanity check
            if 1900 < parsed.year < 2100:
                return (True, parsed.strftime("%Y-%m-%d"), None)
        except ValueError:
            continue
    
    return (False, value, "Could not parse date format")


def validate_phone(value: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate and normalize phone number.
    Returns (is_valid, normalized_digits, error).
    """
    if not value or value.lower() in ("none", "n/a", ""):
        return (False, value, "Empty phone")
    
    # Extract digits only
    digits = re.sub(r"[^\d]", "", value)
    
    # Valid phone should have 7-15 digits
    if len(digits) < 7:
        return (False, value, "Too few digits for phone")
    if len(digits) > 15:
        return (False, value, "Too many digits for phone")
    
    return (True, digits, None)


def validate_email(value: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate email format.
    Returns (is_valid, normalized_email, error).
    """
    if not value or value.lower() in ("none", "n/a", ""):
        return (False, value, "Empty email")
    
    value = value.strip().lower()
    
    # Basic email pattern
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    
    if re.match(pattern, value):
        return (True, value, None)
    
    return (False, value, "Invalid email format")


def validate_name(value: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate name (mostly alphabetic, reasonable length).
    Returns (is_valid, normalized_name, error).
    """
    if not value or value.lower() in ("none", "n/a", ""):
        return (False, value, "Empty name")
    
    value = value.strip()
    
    # Name should be at least 2 characters
    if len(value) < 2:
        return (False, value, "Name too short")
    
    # Name should be mostly letters (allow up to 30% non-letters for titles, Jr., etc.)
    letter_count = sum(c.isalpha() or c.isspace() for c in value)
    letter_ratio = letter_count / len(value)
    
    if letter_ratio < 0.7:
        return (False, value, "Too many non-letter characters for name")
    
    # Check digit ratio (names shouldn't have many digits)
    digit_ratio = sum(c.isdigit() for c in value) / len(value)
    if digit_ratio > 0.2:
        return (False, value, "Too many digits for name")
    
    return (True, value, None)


def validate_age(value: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate age (reasonable number).
    Returns (is_valid, normalized_age, error).
    """
    if not value or value.lower() in ("none", "n/a", ""):
        return (False, value, "Empty age")
    
    # Extract digits
    digits = re.sub(r"[^\d]", "", value)
    
    if not digits:
        return (False, value, "No digits found in age")
    
    try:
        age = int(digits)
        if 0 < age < 150:
            return (True, str(age), None)
        else:
            return (False, value, "Age out of reasonable range")
    except ValueError:
        return (False, value, "Could not parse age")


def validate_address(value: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate address (basic sanity check).
    """
    if not value or value.lower() in ("none", "n/a", ""):
        return (False, value, "Empty address")
    
    value = value.strip()
    
    # Address should have some length
    if len(value) < 5:
        return (False, value, "Address too short")
    
    # Address should have some alphanumeric content
    if not any(c.isalnum() for c in value):
        return (False, value, "No alphanumeric content")
    
    return (True, value, None)


def validate_yes_no(value: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate yes/no field.
    """
    if not value:
        return (False, value, "Empty value")
    
    value_lower = value.strip().lower()
    
    if value_lower in ("yes", "y", "true", "1", "x", "checked"):
        return (True, "Yes", None)
    elif value_lower in ("no", "n", "false", "0", "", "unchecked"):
        return (True, "No", None)
    else:
        return (False, value, "Could not parse as yes/no")


def validate_policy(value: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate policy/ID number.
    """
    if not value or value.lower() in ("none", "n/a", ""):
        return (False, value, "Empty policy number")
    
    value = value.strip()
    
    # Policy should have some alphanumeric content
    if len(value) < 3:
        return (False, value, "Policy number too short")
    
    # Should be mostly alphanumeric
    alnum_count = sum(c.isalnum() for c in value)
    if alnum_count / len(value) < 0.5:
        return (False, value, "Policy should be mostly alphanumeric")
    
    return (True, value.upper(), None)


# ─── Validator Registry ───
# Maps field name keywords → validator functions.
# find_validator() uses keyword containment + difflib fuzzy matching,
# so "Patient Date of Birth" will match "Date of Birth" → validate_date.
VALIDATORS = {
    # Date validators
    "Date": validate_date,
    "DOB": validate_date,
    "Date of Birth": validate_date,
    "Birth Date": validate_date,
    "Collected Date": validate_date,
    "Collection Date": validate_date,
    "Received Date": validate_date,
    "Report Date": validate_date,
    "Admission Date": validate_date,
    "Discharge Date": validate_date,
    
    # Phone validators
    "Phone": validate_phone,
    "Phone Number": validate_phone,
    "Telephone": validate_phone,
    "Tel": validate_phone,
    "Fax": validate_phone,
    "Cell": validate_phone,
    "Mobile": validate_phone,
    
    # Email validators
    "Email": validate_email,
    "E-mail": validate_email,
    "Email Address": validate_email,
    
    # Name validators
    "Name": validate_name,
    "Patient": validate_name,
    "Patient Name": validate_name,
    "Physician": validate_name,
    "Physician Name": validate_name,
    "Doctor": validate_name,
    "Doctor Name": validate_name,
    "Ordering Physician": validate_name,
    "Referring Physician": validate_name,
    "Insured Name": validate_name,
    "Subscriber Name": validate_name,
    
    # Age validators
    "Age": validate_age,
    "Patient Age": validate_age,
    
    # Address validators
    "Address": validate_address,
    "Physician Address": validate_address,
    "Patient Address": validate_address,
    "Mailing Address": validate_address,
    "Street": validate_address,
    
    # Yes/No validators
    "Smoke": validate_yes_no,
    "Smoking": validate_yes_no,
    "Smoker": validate_yes_no,
    "Drink": validate_yes_no,
    "Alcohol": validate_yes_no,
    "Pregnant": validate_yes_no,
    
    # Policy/ID validators
    "Policy": validate_policy,
    "Policy #": validate_policy,
    "Policy Number": validate_policy,
    "Member ID": validate_policy,
    "Group Number": validate_policy,
    "Account Number": validate_policy,
    "SSN": validate_policy,
    "MRN": validate_policy,
    "NPI": validate_policy,
}


def find_validator(field_name: str):
    """
    Find the best matching validator for a given field name.
    
    Uses a two-tier matching strategy (no external dependencies):
    
    1. **Keyword containment** — checks if any registry key appears as a
       substring in the field name (case-insensitive). Longer matches are
       preferred: "Date of Birth" beats "Date" for "Patient Date of Birth".
    
    2. **Fuzzy matching** — falls back to difflib.get_close_matches() with
       a 0.6 cutoff for typo tolerance ("Patnient Name" → "Patient Name").
    
    Reference: https://docs.python.org/3/library/difflib.html#difflib.get_close_matches
    
    Args:
        field_name: The field name from the extraction (e.g. "Patient DOB")
    
    Returns:
        Validator function or None if no match found
    """
    if not field_name:
        return None
    
    field_lower = field_name.lower().strip()
    
    # Tier 1: Exact match (fastest)
    if field_name in VALIDATORS:
        return VALIDATORS[field_name]
    
    # Tier 2: Case-insensitive exact match
    for key, validator in VALIDATORS.items():
        if key.lower() == field_lower:
            return validator
    
    # Tier 3: Keyword containment — prefer longest matching key
    # "Patient Date of Birth" contains "Date of Birth" (len 13) and "Date" (len 4)
    # We want the longer, more specific match.
    best_match = None
    best_match_len = 0
    for key, validator in VALIDATORS.items():
        key_lower = key.lower()
        if key_lower in field_lower and len(key_lower) > best_match_len:
            best_match = validator
            best_match_len = len(key_lower)
    
    if best_match is not None:
        return best_match
    
    # No match found — field passes through without validation
    return None


def validate_field(value: str, field_name: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate and normalize a field value.
    
    Uses find_validator() for smart field-to-validator routing:
    - Exact match: "DOB" → validate_date
    - Substring:   "Patient DOB" → validate_date  (contains "DOB")
    - Fuzzy:       "Patnient Name" → validate_name (close to "Patient Name")
    - No match:    passes through as-is
    
    Args:
        value: The value to validate
        field_name: The field name (e.g. "Patient Name", "DOB", etc.)
    
    Returns:
        (is_valid, normalized_value, error_message)
    """
    validator = find_validator(field_name)
    
    if validator is None:
        # No validator found — accept non-empty values as-is
        if value and value.lower() not in ("none", "n/a", ""):
            return (True, value, None)
        return (False, value, "Empty value")
    
    return validator(value)


def validate_and_select_best(
    candidates: list, 
    field_name: str
) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Try candidates in order, return first that validates.
    
    Args:
        candidates: List of (text, score) tuples
        field_name: Field name for validation routing
    
    Returns:
        (best_value, confidence, error_or_None)
    """
    for text, score in candidates:
        is_valid, normalized, error = validate_field(text, field_name)
        if is_valid:
            return (normalized, score, None)
    
    # No valid candidate found
    if candidates:
        return (candidates[0][0], candidates[0][1] * 0.5, "Validation failed")
    
    return (None, 0, "No candidates")

