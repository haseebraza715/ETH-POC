"""State definitions for the claims PoC."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ClaimState:
    """Dataclass that tracks the evolving claim during the workflow."""

    claim_type: str = "motor_accident"
    date: Optional[str] = None
    time: Optional[str] = None
    location: Optional[str] = None
    other_vehicle_involved: Optional[bool] = None
    other_vehicle_plate: Optional[str] = None
    injuries: Optional[str] = None
    description: Optional[str] = None
    estimated_damage_cost: Optional[float] = None

    fields_source: Dict[str, str] = field(default_factory=dict)
    documents: List[str] = field(default_factory=list)
    completeness_score: float = 0.0
    consistency_flags: List[str] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    doc_extracted_fields: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None
    reasoning_summary: Optional[str] = None
    collection_attempts: int = 0  # Track how many times we've tried to collect info
    validation_cycles: int = 0  # Track how many times we've validated
    previous_completeness: float = 0.0  # Track if we're making progress

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the claim."""
        return asdict(self)

    def add_reasoning(self, entry: str) -> None:
        """Append a single reasoning entry."""
        self.reasoning_trace.append(entry)

    def add_message(self, role: str, content: str) -> None:
        """Append a chat-style message to the history."""
        self.messages.append({"role": role, "content": content})

    def set_field(self, field_name: str, value: Any, source: str) -> None:
        """Utility to set a field and remember its source."""
        setattr(self, field_name, value)
        self.fields_source[field_name] = source


GraphState = Dict[str, ClaimState]

