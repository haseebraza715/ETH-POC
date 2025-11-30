"""LangGraph definition for the claims PoC."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Optional, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from claims_poc.llm_client import call_llm
from claims_poc.state import ClaimState
from claims_poc.tools import consistency, doc_parser, extractor, schema
from claims_poc.tools.consistency import EMPTY_VALUES


class IOInterface(Protocol):
    """Minimal IO contract so nodes can prompt the operator."""

    def ask(self, prompt: str) -> str:
        ...

    def notify(self, message: str) -> None:
        ...


class NeedUserInput(Exception):
    """Exception raised when the workflow needs user input to continue.
    
    This exception is raised by StreamlitIO.ask() when no answer is available
    and the workflow should pause to wait for user input.
    """
    def __init__(self, question: str, current_state=None):
        self.question = question
        self.current_state = current_state  # Optional: current claim state when exception was raised
        super().__init__(f"Need user input for question: {question}")


class NeedMultiUserInput(Exception):
    """Exception raised when the workflow needs multiple user inputs at once (GUI mode).
    
    This exception is raised when there are multiple clarification questions
    that should be shown together in the GUI, allowing the user to answer all at once.
    """
    def __init__(self, questions: List[Dict[str, str]], current_state=None):
        """
        Args:
            questions: List of dicts with keys: 'field', 'question', 'user_value', 'doc_value'
            current_state: Optional current claim state when exception was raised
        """
        self.questions = questions
        self.current_state = current_state
        super().__init__(f"Need user input for {len(questions)} questions")


class TranscriptIO(IOInterface):
    """IO handler that logs prompts/responses for non-interactive runs."""

    def __init__(self, scripted_answers: Optional[List[str]] = None) -> None:
        self._answers = iter(scripted_answers or [])
        self.events: List[str] = []
        self._ask_count = 0

    def ask(self, prompt: str) -> str:
        self._ask_count += 1
        self.events.append(f"question: {prompt}")
        # Prevent infinite loops: if we've asked too many times, return empty
        if self._ask_count > 20:
            self.events.append("answer: [stopped asking to prevent infinite loop]")
            return ""
        answer = next(self._answers, "")
        if answer:
            self.events.append(f"answer: {answer}")
        else:
            # For document path requests, provide a default
            if "police report" in prompt.lower() or "document" in prompt.lower():
                answer = "claims_poc/sample_data/police_report_example.txt"
                self.events.append(f"answer: {answer} [default]")
                return answer
            self.events.append("answer: [no response provided]")
        return answer

    def notify(self, message: str) -> None:
        self.events.append(f"info: {message}")


QUESTION_PROMPT = """You are helping to collect the minimum information needed to file an insurance claim.

Here is the current structured claim state as JSON:
----------------
{claim_state_json}
----------------

Here is the list of required fields for this claim type:
{required_fields_list}

Your task:
- Identify which required fields are still missing (value is null or empty).
- Ask the user 1–3 SHORT, CLEAR questions to collect the MOST important missing fields first.
- Questions should be friendly but professional.
- Ask ONE question per line.

Important:
- Do NOT ask about fields that already have a non-null value.
- Do NOT mention internal field names or JSON.
- Do NOT ask more than 3 questions at once.

Return ONLY the questions, one per line, without any extra commentary."""

CLARIFY_PROMPT = """You are helping to resolve inconsistencies in an insurance claim.

Current claim state (after user answers and document extraction):
----------------
{claim_state_json}
----------------

Detected inconsistencies (each item has: field, user_value, doc_value):
----------------
{inconsistencies_json}
----------------

Your task:
- For EACH inconsistency, generate a clear question asking the user to confirm which value is correct or to clarify the situation.
- Be polite and professional.
- Mention both conflicting values in a natural way.

Rules:
- Ask ONE question per inconsistency.
- One question per line.
- Do NOT restate the full JSON or technical terms.
- Do NOT propose your own answer; just ask for clarification.

Return ONLY the questions, one per line, without any additional text."""

SUMMARY_PROMPT = """You are preparing a short summary of an insurance claim for a human claims handler.

Here is the final structured claim state as JSON:
----------------
{final_claim_state_json}
----------------

Your task:
- Write a concise, natural-language summary (120–200 words) that a claims handler can quickly read and understand.
- Focus on the key facts: what happened, when and where, who was involved, injuries, damage, and any important context from the documents.
- Write in clear, professional English - avoid technical jargon or JSON references.
- If some information is missing or unknown, briefly note that.
- Make it read like a brief case report, not a data dump.

Style:
- Professional, neutral, and factual.
- Use natural flowing sentences and short paragraphs.
- Write as if you're briefing a colleague verbally.

Return ONLY the summary text, no headers or labels."""

TRACE_PROMPT = """You are creating a short reasoning trace that explains how the claim assistant processed this case.

Final claim state:
----------------
{final_claim_state_json}
----------------

Internal processing events:
----------------
{internal_events_list}
----------------

Turn these technical events into a clear, chronological reasoning trace (5–10 bullet points) written in natural, professional English.

Focus on explaining:
- How the assistant collected initial information from the user
- How it processed and extracted data from the police report/document
- How it detected and resolved any inconsistencies between user input and document
- How it validated completeness and consistency
- How it finalized the claim

Guidelines:
- Write in natural, flowing language - avoid technical terms like "validation cycles" or "completeness scores"
- Use past tense and active voice
- Make it read like a brief explanation to a colleague, not a technical log
- Focus on the "what" and "why", not the "how" of implementation
- If inconsistencies were found and resolved, explain that clearly

Output format: bullet points ("- ..." per line), nothing else."""


class LangGraphState(TypedDict, total=False):
    claim: ClaimState
    _io_mode: Optional[str]  # "gui" or "cli" - determines if document request should be skipped


def _parse_boolean(answer: str) -> bool | None:
    lowered = answer.strip().lower()
    if lowered in {"yes", "y", "true", "t", "1"}:
        return True
    if lowered in {"no", "n", "false", "f", "0"}:
        return False
    return None


def _collect_basic_info_node(io: IOInterface, state: LangGraphState) -> LangGraphState:
    claim = state["claim"]
    claim.collection_attempts += 1
    required_fields = schema.get_required_fields_for_claim_type(claim.claim_type)
    optional_fields = {"other_vehicle_plate", "estimated_damage_cost"}
    truly_required = [f for f in required_fields if f not in optional_fields]
    missing_fields = [field for field in truly_required if getattr(claim, field) in EMPTY_VALUES]
    
    if not missing_fields:
        io.notify("All required fields collected from user.")
        return {"claim": claim}
    
    # If we've tried before, set defaults immediately instead of asking again
    if claim.collection_attempts > 1:
        io.notify(f"Setting defaults for remaining fields (attempt {claim.collection_attempts})")
        for field in missing_fields:
            if field == "other_vehicle_involved":
                claim.set_field(field, False, "default")
                claim.add_reasoning(f"Set default for {field}: False")
            else:
                claim.set_field(field, "not provided", "default")
                claim.add_reasoning(f"Set placeholder for {field} (no answer after {claim.collection_attempts} attempts)")
        return {"claim": claim}
    
    # First attempt: ask questions
    to_ask = missing_fields[:3]
    claim_state_json = json.dumps(claim.to_dict(), indent=2)
    required_fields_json = json.dumps(required_fields)
    prompt = QUESTION_PROMPT.format(
        claim_state_json=claim_state_json,
        required_fields_list=required_fields_json,
    )
    fallback_questions = "\n".join(
        f"Could you provide the {field.replace('_', ' ')}?" for field in to_ask
    )
    questions_text = call_llm(prompt, temperature=0.2, fallback_text=fallback_questions)
    question_lines = [line.strip() for line in questions_text.splitlines() if line.strip()]
    if len(question_lines) < len(to_ask):
        question_lines.extend(
            f"Could you share the {field.replace('_', ' ')}?"
            for field in to_ask[len(question_lines) :]
        )

    for field, question in zip(to_ask, question_lines):
        answer = io.ask(f"{question.strip()} ").strip()
        claim.add_message("assistant", question)
        claim.add_message("user", answer)
        
        if not answer:
            # Set defaults immediately on empty answer
            if field == "other_vehicle_involved":
                claim.set_field(field, False, "default")
                claim.add_reasoning(f"Set default for {field}: False (no answer)")
            elif field not in optional_fields:
                claim.set_field(field, "not provided", "default")
                claim.add_reasoning(f"Set placeholder for {field} (no answer)")
            continue
        
        # Process non-empty answers normally
        value = answer
        if field == "other_vehicle_involved":
            parsed = _parse_boolean(answer)
            if parsed is None:
                io.notify("Could not parse yes/no answer, keeping as text.")
            else:
                value = parsed
        elif field == "estimated_damage_cost":
            try:
                value = float(answer.replace(",", ""))
            except ValueError:
                io.notify("Could not parse number, storing raw text.")
        claim.set_field(field, value, "user")
        claim.add_reasoning(f"Collected {field} from user.")
    return {"claim": claim}


def _request_documents_node(io: IOInterface, state: LangGraphState) -> LangGraphState:
    claim = state["claim"]
    if claim.documents:
        io.notify("Documents already on file, skipping request.")
        return {"claim": claim}
    prompt = "Enter path to police report (blank to use sample_data/police_report_example.txt): "
    answer = io.ask(prompt).strip()
    claim.add_message("assistant", prompt)
    if not answer:
        answer = "claims_poc/sample_data/police_report_example.txt"
    claim.documents = [answer]
    claim.add_reasoning(f"Document queued for processing: {answer}")
    return {"claim": claim}


def _process_documents_node(io: IOInterface, state: LangGraphState) -> LangGraphState:
    claim = state["claim"]
    
    # Handle empty document list explicitly
    if not claim.documents:
        io.notify("No documents provided; skipping document processing.")
        claim.add_reasoning("No documents provided; document processing skipped.")
        return {"claim": claim}
    
    for doc_path in claim.documents:
        try:
            doc_text = doc_parser.parse_document(doc_path)
        except Exception as exc:
            io.notify(f"Failed to parse {doc_path}: {exc}")
            claim.add_reasoning(f"Failed to parse {doc_path}: {exc}")
            continue
        io.notify(f"Parsed {doc_path}, extracting fields via LLM.")
        doc_fields, used_fallback = extractor.extract_fields_from_doc(doc_text, claim.claim_type)
        if used_fallback:
            io.notify("LLM JSON extraction failed; using rule-based fallback extractor.")
            claim.add_reasoning(f"Extracted fields from {doc_path} using rule-based fallback (LLM JSON parsing failed)")
        else:
            claim.add_reasoning(f"Extracted fields from {doc_path} via LLM")
        claim.doc_extracted_fields.update(doc_fields)
        for field, value in doc_fields.items():
            if value in (None, "", "null"):
                continue
            current_value = getattr(claim, field, None)
            if current_value in (None, ""):
                claim.set_field(field, value, "document")
    return {"claim": claim}


def _validate_claim_node(state: LangGraphState) -> LangGraphState:
    claim = state["claim"]
    required_fields = schema.get_required_fields_for_claim_type(claim.claim_type)
    score = consistency.compute_completeness(claim, required_fields)
    flags = consistency.find_inconsistencies(claim, claim.doc_extracted_fields)
    
    # Track validation cycles and detect staleness
    claim.validation_cycles += 1
    is_stale = abs(score - claim.previous_completeness) < 0.01  # No improvement
    claim.previous_completeness = score
    
    claim.completeness_score = score
    claim.consistency_flags = flags
    optional_fields = {"other_vehicle_plate", "estimated_damage_cost"}
    truly_required = [f for f in required_fields if f not in optional_fields]
    missing = [field for field in truly_required if getattr(claim, field) in EMPTY_VALUES]
    
    # Create more human-readable trace message
    if missing:
        trace_message = f"Validation cycle {claim.validation_cycles}: Completeness {score:.0%}, missing fields: {', '.join(missing)}"
    elif flags:
        trace_message = f"Validation cycle {claim.validation_cycles}: Completeness {score:.0%}, found inconsistencies: {', '.join(flags)}"
    else:
        trace_message = f"Validation cycle {claim.validation_cycles}: Completeness {score:.0%}, all fields complete and consistent"
    
    if is_stale and claim.validation_cycles > 1:
        trace_message += " [No progress since last cycle]"
    claim.add_reasoning(trace_message)
    return {"claim": claim}


def _validate_answer_format(answer: str, field: str) -> bool:
    """Validate that the answer format matches the expected field type.
    
    Returns True if the format is valid for the field, False otherwise.
    """
    answer = answer.strip()
    
    if field == "date":
        # Date should be in YYYY-MM-DD format or similar date patterns
        # Check for common date formats: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, etc.
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',   # DD/MM/YYYY or MM/DD/YYYY
            r'^\d{4}/\d{2}/\d{2}$',   # YYYY/MM/DD
            r'^\d{2}-\d{2}-\d{4}$',   # DD-MM-YYYY
        ]
        # Reject time-like patterns (HH:MM)
        if re.match(r'^\d{1,2}:\d{2}', answer):
            return False
        # Check if it matches any date pattern
        return any(re.match(pattern, answer) for pattern in date_patterns)
    
    elif field == "time":
        # Time should be in HH:MM format
        time_pattern = r'^\d{1,2}:\d{2}(?::\d{2})?$'  # HH:MM or HH:MM:SS
        return bool(re.match(time_pattern, answer))
    
    # For other fields, accept any non-empty answer
    return bool(answer)


def _normalize_clarification_answer(answer: str, doc_value: Any, field: str, io: Optional[IOInterface] = None) -> Any:
    """Normalize user clarification answer to actual value.
    
    If user indicates they want to use the document value (e.g., "report one", "the report"),
    return the actual doc_value. Otherwise, validate the format and return the answer.
    
    If the format is invalid for the field type, will use document value as fallback.
    """
    if not answer:
        return doc_value if doc_value else answer
    
    answer_lower = answer.lower().strip()
    # Check if user wants to use document value
    doc_indicators = ["report", "document", "doc", "the report", "report one", "from report"]
    if any(indicator in answer_lower for indicator in doc_indicators):
        # User wants to use document value
        return doc_value
    
    # Validate answer format for date/time fields
    if field in ("date", "time") and not _validate_answer_format(answer, field):
        # Format doesn't match expected type - use document value as fallback
        if io:
            io.notify(
                f"Answer '{answer}' doesn't match expected format for {field}. "
                f"Using document value '{doc_value}' instead."
            )
        return doc_value if doc_value else answer
    
    # User provided a specific value with valid format, use it as-is
    return answer


def _clarify_inconsistencies_node(io: IOInterface, state: LangGraphState) -> LangGraphState:
    claim = state["claim"]
    inconsistencies = []
    for flag in claim.consistency_flags:
        field = flag.replace("_mismatch", "")
        inconsistencies.append(
            {
                "field": field,
                "user_value": getattr(claim, field),
                "doc_value": claim.doc_extracted_fields.get(field),
            }
        )
    if not inconsistencies:
        return {"claim": claim}

    claim_json = json.dumps(claim.to_dict(), indent=2)
    inconsistencies_json = json.dumps(inconsistencies, indent=2)
    prompt = CLARIFY_PROMPT.format(
        claim_state_json=claim_json,
        inconsistencies_json=inconsistencies_json,
    )
    def _default_clarify_question(item: Dict[str, object]) -> str:
        return (
            f"You mentioned {item['field']} as {item['user_value']}, "
            f"but the report lists {item['doc_value']}. Which is correct?"
        )

    fallback_questions = "\n".join(_default_clarify_question(item) for item in inconsistencies)
    questions_text = call_llm(prompt, temperature=0.2, fallback_text=fallback_questions)
    question_lines = [line.strip() for line in questions_text.splitlines() if line.strip()]
    if len(question_lines) < len(inconsistencies):
        remaining = inconsistencies[len(question_lines) :]
        question_lines.extend(
            f"Could you confirm the correct value for {item['field']} (you said {item['user_value']}, "
            f"document shows {item['doc_value']})?"
            for item in remaining
        )

    # Check if we're in GUI mode
    io_mode = state.get("_io_mode", "cli")
    
    if io_mode == "gui" and len(inconsistencies) > 1:
        # GUI mode with multiple inconsistencies: ask all questions at once
        # Build list of questions with field identifiers
        questions_with_fields = []
        for item, question in zip(inconsistencies, question_lines):
            field = item["field"]
            question_with_field = f"[FIELD:{field}] {question}"
            questions_with_fields.append({
                "field": field,
                "question": question_with_field,
                "display_question": question,  # Question without field tag for display
                "user_value": item["user_value"],
                "doc_value": item["doc_value"],
            })
        
        # Check if we already have answers for all questions
        if hasattr(io, "get_multi_answers"):
            answers = io.get_multi_answers(questions_with_fields)
            if answers and all(q["question"] in answers for q in questions_with_fields):
                # All answers available, process them
                for q_info in questions_with_fields:
                    field = q_info["field"]
                    question = q_info["question"]
                    display_question = q_info["display_question"]
                    doc_value = q_info["doc_value"]
                    answer = answers.get(question, "").strip()
                    
                    claim.add_message("assistant", display_question)
                    claim.add_message("user", answer)
                    
                    if answer:
                        normalized_value = _normalize_clarification_answer(answer, doc_value, field, io)
                        setattr(claim, field, normalized_value)
                        claim.fields_source[field] = "clarified"
                        
                        if normalized_value == doc_value:
                            if answer.lower() in ["report", "document", "doc", "the report", "report one", "from report"]:
                                claim.add_reasoning(
                                    f"Clarified {field}: user confirmed document's value '{normalized_value}'"
                                )
                            else:
                                claim.add_reasoning(
                                    f"Clarified {field}: answer '{answer}' had invalid format, using document value '{normalized_value}'"
                                )
                        else:
                            claim.add_reasoning(f"Clarified {field} to '{normalized_value}'")
                    else:
                        if doc_value:
                            setattr(claim, field, doc_value)
                            claim.fields_source[field] = "clarified"
                            claim.add_reasoning(
                                f"Clarified {field}: no answer provided, using document value '{doc_value}'"
                            )
            else:
                # Not all answers available, raise exception to get user input
                raise NeedMultiUserInput(questions_with_fields, current_state=state)
        else:
            # IO handler doesn't support multi-answers, raise exception
            raise NeedMultiUserInput(questions_with_fields, current_state=state)
    else:
        # CLI mode or single inconsistency: process one at a time (existing behavior)
        if inconsistencies and question_lines:
            item = inconsistencies[0]
            question = question_lines[0]
            field = item["field"]
            doc_value = item["doc_value"]
            
            question_with_field = f"[FIELD:{field}] {question}"
            answer = io.ask(f"{question_with_field} ").strip()
            claim.add_message("assistant", question)
            claim.add_message("user", answer)
            
            if answer:
                normalized_value = _normalize_clarification_answer(answer, doc_value, field, io)
                setattr(claim, field, normalized_value)
                claim.fields_source[field] = "clarified"
                
                if normalized_value == doc_value:
                    if answer.lower() in ["report", "document", "doc", "the report", "report one", "from report"]:
                        claim.add_reasoning(
                            f"Clarified {field}: user confirmed document's value '{normalized_value}'"
                        )
                    else:
                        claim.add_reasoning(
                            f"Clarified {field}: answer '{answer}' had invalid format, using document value '{normalized_value}'"
                        )
                else:
                    claim.add_reasoning(f"Clarified {field} to '{normalized_value}'")
            else:
                if doc_value:
                    setattr(claim, field, doc_value)
                    claim.fields_source[field] = "clarified"
                    claim.add_reasoning(
                        f"Clarified {field}: no answer provided, using document value '{doc_value}'"
                    )
    
    # Re-run consistency check after clarifications to update flags
    updated_flags = consistency.find_inconsistencies(claim, claim.doc_extracted_fields)
    claim.consistency_flags = updated_flags
    
    if not updated_flags:
        claim.add_reasoning("All inconsistencies resolved after clarification.")
    
    return {"claim": claim}


def get_summary_ready_state(claim: ClaimState) -> Dict[str, Any]:
    """Return a cleaned state dict with only user-facing fields for summary generation.
    
    Filters out internal debugging/technical fields that shouldn't appear in handler summaries.
    """
    internal_fields = {
        "collection_attempts",
        "validation_cycles",
        "previous_completeness",
        "messages",
        "reasoning_trace",
        "doc_extracted_fields",
    }
    d = claim.to_dict()
    return {k: v for k, v in d.items() if k not in internal_fields}


def filter_technical_reasoning_entries(reasoning_trace: List[str]) -> List[str]:
    """Filter out technical/internal entries from reasoning trace before LLM processing.
    
    Removes entries containing:
    - "Completeness=" (completeness scores)
    - "cycles=" or "cycle" (validation cycles)
    - "attempts" (collection attempts)
    - Technical metrics and internal state information
    
    Keeps human-friendly entries about actual processing steps.
    """
    technical_patterns = [
        "completeness=",
        "completeness ",
        "cycles=",
        "cycle",
        "attempts",
        "validation cycle",
        "no progress",
        "reached maximum",
        "finalizing with completeness",
    ]
    
    filtered = []
    for entry in reasoning_trace:
        entry_lower = entry.lower()
        # Skip entries that contain technical patterns
        if any(pattern in entry_lower for pattern in technical_patterns):
            continue
        # Keep human-friendly entries
        filtered.append(entry)
    
    return filtered


def _finalize_claim_node(io: IOInterface, state: LangGraphState) -> LangGraphState:
    claim = state["claim"]
    # Use cleaned state (without internal fields) for summary generation
    clean_state = get_summary_ready_state(claim)
    final_json = json.dumps(clean_state, indent=2)
    summary_prompt = SUMMARY_PROMPT.format(final_claim_state_json=final_json)
    summary_fallback = "Offline summary: claim finalized with available data."
    summary_text = call_llm(summary_prompt, temperature=0.2, fallback_text=summary_fallback)
    claim.summary = summary_text

    # Filter out technical entries before sending to LLM for trace generation
    filtered_trace = filter_technical_reasoning_entries(claim.reasoning_trace)
    events_list = json.dumps(filtered_trace, indent=2)
    # Use cleaned state for trace prompt as well to avoid technical jargon
    trace_prompt = TRACE_PROMPT.format(
        final_claim_state_json=final_json,
        internal_events_list=events_list,
    )
    # Use filtered trace for fallback as well
    trace_fallback = "\n".join(f"- {event}" for event in filtered_trace[-6:]) if filtered_trace else "- Processed claim."
    reasoning_summary = call_llm(trace_prompt, temperature=0.1, fallback_text=trace_fallback or "- Processed claim.")
    claim.reasoning_summary = reasoning_summary
    claim.add_reasoning("Finalized claim with summary and reasoning trace.")
    io.notify("Summary ready.")
    return {"claim": claim}


def _route_after_collect(state: LangGraphState) -> str:
    """Route after collecting basic info: skip document request in GUI mode if no docs."""
    claim = state["claim"]
    io_mode = state.get("_io_mode", "cli")
    
    # In GUI mode, if no documents are provided, skip the request node
    if io_mode == "gui" and not claim.documents:
        claim.add_reasoning("GUI mode: no document uploaded, skipping document request.")
        return "process_documents"
    
    # Otherwise, go to request_documents (CLI mode or GUI with documents)
    return "request_documents"


def _route_validation(state: LangGraphState) -> str:
    claim = state["claim"]
    
    # PRIORITY 1: If complete and consistent, finalize
    if claim.completeness_score >= 1.0 and not claim.consistency_flags:
        claim.add_reasoning("Claim is complete and consistent; proceeding to finalization.")
        return "finalize_claim"
    
    # PRIORITY 2: Force finalization if we've validated too many times
    if claim.validation_cycles >= 3:
        claim.add_reasoning(
            f"Reached maximum validation cycles ({claim.validation_cycles}); finalizing with current state."
        )
        return "finalize_claim"
    
    # PRIORITY 3: Force finalization if state is stale (no progress) and we have acceptable completeness
    if (claim.validation_cycles > 1 and 
        abs(claim.completeness_score - claim.previous_completeness) < 0.01 and
        claim.completeness_score >= 0.8):
        claim.add_reasoning(
            f"No progress detected after {claim.validation_cycles} cycles; "
            f"finalizing with completeness {claim.completeness_score:.0%}."
        )
        return "finalize_claim"
    
    # PRIORITY 4: Force finalization if we've collected multiple times with acceptable completeness
    if claim.collection_attempts >= 2 and claim.completeness_score >= 0.8:
        claim.add_reasoning(
            f"After {claim.collection_attempts} collection attempts, "
            f"completeness {claim.completeness_score:.0%} is acceptable; finalizing."
        )
        return "finalize_claim"
    
    # Normal routing
    if claim.completeness_score < 1.0:
        return "collect_basic_info"
    if claim.consistency_flags:
        return "clarify_inconsistencies"
    return "finalize_claim"


def build_graph(io_handler: IOInterface) -> Callable[[LangGraphState], LangGraphState]:
    """Compile the LangGraph with injected IO handler."""

    def collect_node(state: LangGraphState) -> LangGraphState:
        return _collect_basic_info_node(io_handler, state)

    def request_node(state: LangGraphState) -> LangGraphState:
        return _request_documents_node(io_handler, state)

    def process_node(state: LangGraphState) -> LangGraphState:
        return _process_documents_node(io_handler, state)

    def clarify_node(state: LangGraphState) -> LangGraphState:
        return _clarify_inconsistencies_node(io_handler, state)

    def finalize_node(state: LangGraphState) -> LangGraphState:
        return _finalize_claim_node(io_handler, state)

    graph = StateGraph(LangGraphState)
    graph.add_node("collect_basic_info", collect_node)
    graph.add_node("request_documents", request_node)
    graph.add_node("process_documents", process_node)
    graph.add_node("validate_claim", _validate_claim_node)
    graph.add_node("clarify_inconsistencies", clarify_node)
    graph.add_node("finalize_claim", finalize_node)

    graph.add_edge(START, "collect_basic_info")
    graph.add_conditional_edges("collect_basic_info", _route_after_collect)
    graph.add_edge("request_documents", "process_documents")
    graph.add_edge("process_documents", "validate_claim")
    graph.add_conditional_edges("validate_claim", _route_validation)
    graph.add_edge("clarify_inconsistencies", "validate_claim")
    graph.add_edge("finalize_claim", END)
    workflow = graph.compile()
    return workflow


def _apply_initial_answers(claim: ClaimState, initial_answers: Dict[str, Any]) -> None:
    for field, value in initial_answers.items():
        if not hasattr(claim, field):
            continue
        # Skip empty strings and placeholder values
        if value in EMPTY_VALUES:
            continue
        # Handle boolean fields
        if field == "other_vehicle_involved" and isinstance(value, str):
            parsed = _parse_boolean(value)
            if parsed is not None:
                value = parsed
        claim.set_field(field, value, "user")


def _materialize_doc(doc_bytes: bytes, doc_name: Optional[str]) -> str:
    suffix = Path(doc_name or "").suffix or ".txt"
    temp = NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(doc_bytes)
    temp.flush()
    temp.close()
    return temp.name


def run_claim_flow(
    *,
    initial_answers: Dict[str, Any],
    doc_bytes: Optional[bytes] = None,
    doc_name: Optional[str] = None,
    io_handler: Optional[IOInterface] = None,
) -> tuple[ClaimState, str, List[str], List[str]]:
    """Run the LangGraph workflow for programmatic callers (Streamlit/tests/etc.)."""

    if io_handler is None:
        io_handler = TranscriptIO()
    workflow = build_graph(io_handler)
    claim = ClaimState()
    _apply_initial_answers(claim, initial_answers)

    temp_doc_path: Optional[str] = None
    try:
        if doc_bytes:
            temp_doc_path = _materialize_doc(doc_bytes, doc_name)
            claim.documents = [temp_doc_path]
            claim.add_reasoning(f"Uploaded document processed: {doc_name or Path(temp_doc_path).name}")

        final_state = workflow.invoke(
            {"claim": claim, "_io_mode": "gui"},
            config={"recursion_limit": 100},
        )["claim"]
    finally:
        if temp_doc_path:
            try:
                os.unlink(temp_doc_path)
            except OSError:
                pass

    summary = final_state.summary or ""
    reasoning_display = (
        final_state.reasoning_summary.splitlines()
        if final_state.reasoning_summary
        else final_state.reasoning_trace
    )
    events = io_handler.events + final_state.reasoning_trace if hasattr(io_handler, "events") else final_state.reasoning_trace
    return final_state, summary, reasoning_display, events

