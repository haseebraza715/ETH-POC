"""Streamlit UI for the claims PoC."""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

import streamlit as st

from claims_poc.graph import NeedUserInput, NeedMultiUserInput, run_claim_flow
from claims_poc.state import ClaimState
from claims_poc.streamlit_io import StreamlitIO


def render_json(state: ClaimState) -> str:
    if hasattr(state, "to_dict"):
        return json.dumps(state.to_dict(), indent=2)
    return json.dumps(state.__dict__, indent=2)


def main() -> None:
    st.set_page_config(page_title="Claims Intake & Validation Assistant", layout="wide")
    st.title("Agentic Claims Intake & Validation Assistant (PoC)")
    st.caption("Zurich Insurance × ETH Agentic Systems Lab – PoC by Haseeb")

    # Check LLM configuration
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    if not has_openrouter and not has_openai:
        st.warning(
            "⚠️ No LLM API key detected. Set OPENROUTER_API_KEY or OPENAI_API_KEY for full functionality. "
            "Running in offline mode with rule-based extraction."
        )

    # Initialize session state
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = "input"  # input, processing, answering_questions, complete
    if "final_result" not in st.session_state:
        st.session_state.final_result = None
    if "workflow_iteration" not in st.session_state:
        st.session_state.workflow_iteration = 0
    if "current_claim_state" not in st.session_state:
        st.session_state.current_claim_state = None

    # Handle workflow states
    if st.session_state.workflow_state == "processing":
        # Run the workflow - it may ask questions during execution
        # Collect answers from session state if available
        answer_map = st.session_state.get("question_answers_map", {})
        answer_queue = list(answer_map.values()) if answer_map else []
        
        io_handler = StreamlitIO(answer_queue=answer_queue)
        with st.spinner("Processing claim with agentic workflow..."):
            try:
                final_state, summary, reasoning_trace, events = run_claim_flow(
                    initial_answers=st.session_state.initial_answers,
                    doc_bytes=st.session_state.doc_bytes,
                    doc_name=st.session_state.get("doc_name"),
                    io_handler=io_handler,
                )
                
                # Check if there are pending questions that need user input
                pending_questions = io_handler.get_pending_questions()
                
                if pending_questions:
                    # Questions were asked but not answered - need user input
                    st.session_state.current_claim_state = (final_state, summary, reasoning_trace, events)
                    st.session_state.workflow_state = "answering_questions"
                    st.rerun()
                else:
                    # No pending questions - workflow is complete
                    # Clear any remaining pending questions
                    if "pending_questions" in st.session_state:
                        st.session_state.pending_questions = []
                    st.session_state.final_result = (final_state, summary, reasoning_trace, events)
                    st.session_state.workflow_state = "complete"
                    st.rerun()
            except NeedMultiUserInput as e:
                # Workflow needs multiple user inputs at once (GUI mode)
                questions = e.questions
                # Store multi-question info in session state
                st.session_state.multi_questions = questions
                # Don't store current_state from exception - it's a LangGraphState dict, not the tuple we need
                # The current_claim_state will be updated when we have the full result
                # Switch to answering_questions state with multi-question mode
                st.session_state.workflow_state = "answering_questions"
                st.session_state.multi_question_mode = True
                st.rerun()
            except NeedUserInput as e:
                # Workflow needs user input - pause and show question
                question = e.question
                # Ensure question is in pending_questions list
                if "pending_questions" not in st.session_state:
                    st.session_state.pending_questions = []
                if question not in st.session_state.pending_questions:
                    st.session_state.pending_questions.append(question)
                # Don't store current_state from exception - it's a LangGraphState dict, not the tuple we need
                # The current_claim_state will be updated when we have the full result
                # Switch to answering_questions state
                st.session_state.workflow_state = "answering_questions"
                st.session_state.multi_question_mode = False
                st.rerun()
            except Exception as e:
                error_msg = str(e)
                # Check if it's a rate limit error that wasn't caught by llm_client
                if "429" in error_msg or "rate" in error_msg.lower() or "rate-limit" in error_msg.lower():
                    st.warning(
                        "⚠️ Rate limit reached. Please wait a moment and try again, or the system will use fallback extraction. "
                        "If you have your own API key, you can add it to OpenRouter settings to avoid rate limits."
                    )
                    st.session_state.workflow_state = "input"
                else:
                    st.error(f"Error processing claim: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                    st.session_state.workflow_state = "input"

    elif st.session_state.workflow_state == "answering_questions":
        # Display pending questions and collect answers
        io_handler = StreamlitIO()
        
        # Check if we're in multi-question mode
        multi_question_mode = st.session_state.get("multi_question_mode", False)
        
        if multi_question_mode and "multi_questions" in st.session_state:
            # Multi-question mode: show all clarification questions at once
            questions = st.session_state.multi_questions
            st.markdown("### 🔍 Clarification Needed")
            st.info("The agent found inconsistencies between your input and the document. Please clarify the correct values for each field.")
            
            answers_provided = {}
            for idx, q_info in enumerate(questions):
                field = q_info["field"]
                question = q_info["question"]  # Full question with [FIELD:...] prefix
                display_question = q_info["display_question"]  # Clean question for display
                field_label = field.replace("_", " ").title()
                
                st.markdown(f"**Question {idx + 1} - {field_label}:**")
                st.write(display_question)
                
                # Check if answer already exists in session state
                existing_answer = st.session_state.question_answers_map.get(question, "")
                answer = st.text_input(
                    f"Your answer for {field_label}:",
                    value=existing_answer,
                    key=f"multi_answer_{st.session_state.workflow_iteration}_{field}",
                    label_visibility="visible"
                )
                answers_provided[question] = answer
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Submit All Answers", type="primary", key="submit_multi_answers"):
                    # Store all answers
                    all_answered = True
                    for question, answer in answers_provided.items():
                        if not answer.strip():
                            all_answered = False
                            # Find the field for better error message
                            for q_info in questions:
                                if q_info["question"] == question:
                                    field_label = q_info["field"].replace("_", " ").title()
                                    st.error(f"Please provide an answer for {field_label}")
                                    break
                            break
                        io_handler.set_answer(question, answer.strip())
                    
                    if all_answered:
                        # Store field-to-answer mapping for reliable matching
                        # This ensures answers are matched by field even if question text varies
                        if "field_answers_map" not in st.session_state:
                            st.session_state.field_answers_map = {}
                        for q_info in questions:
                            field = q_info["field"]
                            question = q_info["question"]
                            if question in answers_provided:
                                st.session_state.field_answers_map[field] = answers_provided[question].strip()
                        
                        # Clear multi-question mode and re-run workflow
                        st.session_state.workflow_iteration += 1
                        st.session_state.workflow_state = "processing"
                        if "multi_question_mode" in st.session_state:
                            del st.session_state.multi_question_mode
                        if "multi_questions" in st.session_state:
                            del st.session_state.multi_questions
                        st.rerun()
            
            with col2:
                if st.button("Skip All (Use Document Values)", key="skip_multi_answers"):
                    # Set empty answers for all questions (will use document values)
                    for q_info in questions:
                        io_handler.set_answer(q_info["question"], "")
                    st.session_state.workflow_iteration += 1
                    st.session_state.workflow_state = "processing"
                    if "multi_question_mode" in st.session_state:
                        del st.session_state.multi_question_mode
                    if "multi_questions" in st.session_state:
                        del st.session_state.multi_questions
                    st.rerun()
            
            # Show current progress if available
            if st.session_state.current_claim_state:
                with st.expander("View Current Claim State"):
                    try:
                        # Ensure current_claim_state is a tuple of 4 values
                        if isinstance(st.session_state.current_claim_state, tuple) and len(st.session_state.current_claim_state) == 4:
                            final_state, summary, reasoning_trace, events = st.session_state.current_claim_state
                            st.json(final_state.to_dict())
                        else:
                            # If it's not the expected format, just show it as-is
                            st.json(st.session_state.current_claim_state)
                    except (ValueError, TypeError):
                        # If unpacking fails, just show the raw state
                        st.json(st.session_state.current_claim_state)
        else:
            # Single question mode (legacy behavior)
            pending_questions = io_handler.get_pending_questions()
            
            if not pending_questions:
                # No more questions, continue processing
                st.session_state.workflow_state = "processing"
                st.rerun()
                return
            
            st.markdown("### 🔍 Clarification Needed")
            st.info("The agent needs some additional information to process your claim.")
            
            # Show all pending questions
            answers_provided = {}
            for idx, question in enumerate(pending_questions):
                # Extract field identifier if present and display clean question
                display_question = question
                field_label = None
                if question.startswith("[FIELD:") and "]" in question:
                    field_end = question.index("]")
                    field_name = question[7:field_end]
                    display_question = question[field_end + 1:].strip()
                    # Create a user-friendly field label
                    field_label = field_name.replace("_", " ").title()
                
                # Display question with field label if available
                if field_label:
                    st.markdown(f"**Question {idx + 1} - {field_label}:**")
                else:
                    st.markdown(f"**Question {idx + 1}:**")
                st.write(display_question)
                
                # Check if answer already exists in session state
                existing_answer = st.session_state.question_answers_map.get(question, "")
                answer = st.text_input(
                    f"Your answer:",
                    value=existing_answer,
                    key=f"answer_{st.session_state.workflow_iteration}_{idx}",
                    label_visibility="collapsed"
                )
                answers_provided[question] = answer
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Submit Answers", type="primary", key="submit_answers"):
                    # Store all answers
                    all_answered = True
                    for question, answer in answers_provided.items():
                        if not answer.strip():
                            all_answered = False
                            st.error(f"Please provide an answer for: {question}")
                            break
                        io_handler.set_answer(question, answer.strip())
                    
                    if all_answered:
                        # Clear pending questions and re-run workflow with answers
                        st.session_state.workflow_iteration += 1
                        st.session_state.workflow_state = "processing"
                        st.rerun()
            
            with col2:
                if st.button("Skip (Use Defaults)", key="skip_answers"):
                    # Clear pending questions and continue with defaults
                    for question in pending_questions:
                        io_handler.set_answer(question, "")
                    st.session_state.workflow_iteration += 1
                    st.session_state.workflow_state = "processing"
                    st.rerun()
            
            # Show current progress if available
            if st.session_state.current_claim_state:
                with st.expander("View Current Claim State"):
                    try:
                        # Ensure current_claim_state is a tuple of 4 values
                        if isinstance(st.session_state.current_claim_state, tuple) and len(st.session_state.current_claim_state) == 4:
                            final_state, summary, reasoning_trace, events = st.session_state.current_claim_state
                            st.json(final_state.to_dict())
                        else:
                            # If it's not the expected format, just show it as-is
                            st.json(st.session_state.current_claim_state)
                    except (ValueError, TypeError):
                        # If unpacking fails, just show the raw state
                        st.json(st.session_state.current_claim_state)

    elif st.session_state.workflow_state == "complete":
        # Show results
        if st.session_state.final_result:
            final_state, summary, reasoning_trace, events = st.session_state.final_result
            st.success("✅ Claim processed successfully!")
            
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Structured Claim JSON", "Handler Summary", "Reasoning Trace", "Event Log"]
            )

            with tab1:
                st.subheader("Final Claim State (JSON)")
                st.code(render_json(final_state), language="json")

            with tab2:
                st.subheader("Summary for Claims Handler")
                st.write(summary or "No summary generated.")

            with tab3:
                st.subheader("Reasoning Trace (Human-Readable)")
                if reasoning_trace:
                    for step in reasoning_trace:
                        st.markdown(f"- {step}")
                else:
                    st.write("No reasoning trace available.")

            with tab4:
                st.subheader("Raw Event Log (Debug)")
                if events:
                    for event in events:
                        st.text(event)
                else:
                    st.write("No events logged.")
            
            if st.button("Start New Claim"):
                # Reset session state
                keys_to_keep = []
                for key in list(st.session_state.keys()):
                    if key not in keys_to_keep:
                        del st.session_state[key]
                st.session_state.workflow_state = "input"
                st.rerun()

    else:  # input state
        # Input form
        st.markdown("### Claim Information")
        col1, col2 = st.columns(2)
        with col1:
            date = st.text_input("Date of incident", placeholder="2024-05-18", key="input_date")
            time = st.text_input("Time of incident", placeholder="22:25", key="input_time")
            location = st.text_input("Location", placeholder="Zurich Central Station", key="input_location")
            injuries = st.selectbox(
                "Injuries",
                options=["none", "minor", "serious", "unknown"],
                index=0,
                key="input_injuries",
            )
        with col2:
            description = st.text_area(
                "Short description of what happened",
                height=150,
                placeholder="Rear-end collision at traffic light, no injuries...",
                key="input_description",
            )
            uploaded_file = st.file_uploader(
                "Police report / incident document (TXT or PDF)",
                type=["txt", "pdf"],
                key="input_file",
            )

        st.markdown("---")

        if st.button("Run Claim Assistant", type="primary"):
            if not date or not time or not location or not description:
                st.error("Please fill at least date, time, location, and description.")
            else:
                # Store inputs and start processing
                st.session_state.initial_answers = {
                    "date": date,
                    "time": time,
                    "location": location,
                    "injuries": injuries if injuries != "unknown" else None,
                    "description": description,
                }
                st.session_state.doc_bytes = uploaded_file.read() if uploaded_file is not None else None
                st.session_state.doc_name = uploaded_file.name if uploaded_file is not None else None
                # Initialize question handling
                if "question_answers_map" not in st.session_state:
                    st.session_state.question_answers_map = {}
                if "pending_questions" not in st.session_state:
                    st.session_state.pending_questions = []
                st.session_state.workflow_iteration = 0
                st.session_state.workflow_state = "processing"
                st.rerun()


if __name__ == "__main__":
    main()

