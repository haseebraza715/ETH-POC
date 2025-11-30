"""Field extraction helpers."""

from __future__ import annotations

import json
from typing import Dict

from claims_poc.llm_client import RuleBasedExtractor, call_llm

EXTRACTION_PROMPT = """You will extract structured information from an insurance-related document.

Claim type: {claim_type}

Document text:
----------------
{doc_text}
----------------

From this document, extract ONLY the following fields, if they are present:
- date
- time
- location
- other_vehicle_plate
- injuries
- description

Return a single JSON object with EXACTLY these keys:
{{
  "date": ...,
  "time": ...,
  "location": ...,
  "other_vehicle_plate": ...,
  "injuries": ...,
  "description": ...
}}

Rules:
- If the document does NOT clearly contain a field, set its value to null.
- Do NOT guess or infer missing fields.
- Do NOT add any extra keys or text outside the JSON.
Important:
- If a value is not explicitly stated in the text, set it to null.
- Do NOT guess or infer values based on common sense."""


def extract_fields_from_doc(doc_text: str, claim_type: str) -> tuple[Dict[str, str], bool]:
    """Use an LLM to extract structured data from a document.
    
    Falls back to rule-based extraction if JSON parsing fails.
    
    Returns:
        tuple: (extracted_fields_dict, used_fallback_flag)
    """
    prompt = EXTRACTION_PROMPT.format(doc_text=doc_text[:6000], claim_type=claim_type)
    response = call_llm(prompt, response_format="json", temperature=0.0)
    
    # Try to parse JSON response
    try:
        if isinstance(response, str):
            parsed = json.loads(response)
        else:
            parsed = response
        
        # Validate that we got a dict
        if isinstance(parsed, dict):
            return parsed, False
        else:
            # Response is not a dict, fall back to rule-based extraction
            return RuleBasedExtractor.run(prompt), True
    except json.JSONDecodeError:
        # JSON parsing failed, fall back to rule-based extraction
        return RuleBasedExtractor.run(prompt), True

