"""Completeness and consistency helpers."""

from __future__ import annotations

from typing import Dict, List

from claims_poc.state import ClaimState

# Values that should be treated as empty/not filled
EMPTY_VALUES = {None, "", "unknown", "not provided"}


def compute_completeness(state: ClaimState, required_fields: List[str]) -> float:
    """Compute completeness score (0.0 to 1.0) based on required fields.
    
    Only counts fields with meaningful values - placeholder values like
    "not provided" or "unknown" are treated as empty.
    """
    optional_fields = {"other_vehicle_plate", "estimated_damage_cost"}
    truly_required = [f for f in required_fields if f not in optional_fields]
    # Count fields that have meaningful values (exclude placeholders)
    filled = sum(
        1 for field in truly_required 
        if getattr(state, field) not in EMPTY_VALUES
    )
    return round(filled / len(truly_required), 2) if truly_required else 1.0


def find_inconsistencies(state: ClaimState, doc_fields: Dict[str, str]) -> List[str]:
    """Find inconsistencies between user-provided state and document-extracted fields.
    
    Checks the following fields for mismatches:
    - date
    - time
    - location
    - injuries
    - other_vehicle_plate
    
    Only flags mismatches when both values are present and non-empty.
    """
    flags: List[str] = []
    # Fields to check for inconsistencies
    checkable_fields = ["date", "time", "location", "injuries", "other_vehicle_plate"]
    
    for field in checkable_fields:
        doc_value = doc_fields.get(field)
        state_value = getattr(state, field, None)
        
        # Only flag mismatch if both values exist and are non-empty, and they differ
        if (doc_value and 
            doc_value not in EMPTY_VALUES and
            state_value and 
            state_value not in EMPTY_VALUES and
            str(doc_value).strip() != str(state_value).strip()):
            flags.append(f"{field}_mismatch")
    
    return flags

