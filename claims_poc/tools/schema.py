
from typing import Dict, List

DEFAULT_SCHEMA = [
    "date",
    "time",
    "location",
    "other_vehicle_involved",
    "injuries",
    "description",
]

CLAIM_TYPE_SCHEMAS: Dict[str, List[str]] = {
    "motor_accident": DEFAULT_SCHEMA + ["other_vehicle_plate", "estimated_damage_cost"],
    "theft": ["date", "location", "description", "estimated_damage_cost"],
}


def get_required_fields_for_claim_type(claim_type: str) -> List[str]:
    return CLAIM_TYPE_SCHEMAS.get(claim_type, DEFAULT_SCHEMA)

