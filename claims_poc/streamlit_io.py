"""Streamlit-compatible IO handler for interactive workflows."""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

from claims_poc.graph import IOInterface, NeedUserInput
from typing import Dict


class StreamlitIO(IOInterface):
    """IO handler that uses Streamlit session state for interactive questions.
    
    This handler stores questions in session state and retrieves answers from session state.
    The Streamlit app must check for pending questions and prompt the user.
    """

    def __init__(self, answer_queue: Optional[List[str]] = None) -> None:
        """Initialize with optional pre-populated answer queue."""
        self.answer_queue = answer_queue or []
        self.questions: List[str] = []
        self.notifications: List[str] = []
        self._question_index = 0
        
        # Initialize session state keys if not present
        if "pending_questions" not in st.session_state:
            st.session_state.pending_questions = []
        if "question_answers_map" not in st.session_state:
            st.session_state.question_answers_map = {}

    def ask(self, prompt: str) -> str:
        """Store question in session state and return answer if available.
        
        Raises NeedUserInput exception if no answer is available and workflow should pause.
        
        IMPORTANT: Each question should be answered exactly once. Answers are consumed
        after being returned to prevent reuse across multiple questions.
        """
        self.questions.append(prompt)
        
        # Extract field identifier if present (format: [FIELD:field_name] question text)
        field = None
        question_text = prompt
        if prompt.startswith("[FIELD:") and "]" in prompt:
            field_end = prompt.index("]")
            field = prompt[7:field_end]  # Extract field name from [FIELD:field_name]
            question_text = prompt[field_end + 1:].strip()  # Remove field identifier from question
        
        # Track consumed answers to prevent reuse
        if "consumed_answers" not in st.session_state:
            st.session_state.consumed_answers = set()
        
        # First, check if answer already exists in session state (from previous interaction)
        # This takes priority so we can resume workflows with existing answers
        if "question_answers_map" in st.session_state:
            # For questions with field identifiers, ONLY use exact match to prevent cross-field reuse
            if field:
                # Questions with field identifiers must match exactly
                if prompt in st.session_state.question_answers_map:
                    answer = st.session_state.question_answers_map[prompt]
                    # Check if this answer has already been consumed
                    answer_key = f"{prompt}::{answer}"
                    if answer and answer_key not in st.session_state.consumed_answers:
                        # Mark as consumed and return
                        st.session_state.consumed_answers.add(answer_key)
                        return answer
            else:
                # For questions without field identifiers, try exact match first
                if prompt in st.session_state.question_answers_map:
                    answer = st.session_state.question_answers_map[prompt]
                    answer_key = f"{prompt}::{answer}"
                    if answer and answer_key not in st.session_state.consumed_answers:
                        st.session_state.consumed_answers.add(answer_key)
                        return answer
                # Then try fuzzy match (only for questions without field identifiers)
                for existing_q, existing_a in st.session_state.question_answers_map.items():
                    if existing_a:
                        # Skip questions with field identifiers in fuzzy matching
                        if existing_q.startswith("[FIELD:") and "]" in existing_q:
                            continue
                        answer_key = f"{existing_q}::{existing_a}"
                        if answer_key not in st.session_state.consumed_answers and self._questions_similar(prompt, existing_q):
                            st.session_state.consumed_answers.add(answer_key)
                            return existing_a
        
        # Check if we have a pre-populated answer queue
        if self._question_index < len(self.answer_queue):
            answer = self.answer_queue[self._question_index]
            self._question_index += 1
            return answer
        
        # Store field-to-question mapping if field is present
        if field:
            if "field_question_map" not in st.session_state:
                st.session_state.field_question_map = {}
            st.session_state.field_question_map[field] = prompt
        
        # Store question in session state for Streamlit UI to display
        if "pending_questions" not in st.session_state:
            st.session_state.pending_questions = []
        if prompt not in st.session_state.pending_questions:
            st.session_state.pending_questions.append(prompt)
        
        # Raise exception to pause workflow - Streamlit app will catch and show question
        raise NeedUserInput(prompt)
    
    @staticmethod
    def _questions_similar(q1: str, q2: str) -> bool:
        """Check if two questions are similar enough to be considered the same."""
        # Simple similarity check: if one question contains most words of the other
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        if not words1 or not words2:
            return False
        # If 70% of words overlap, consider them similar
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap > 0.7

    def notify(self, message: str) -> None:
        """Store notification in both local list and session state."""
        self.notifications.append(message)
        if "notifications" not in st.session_state:
            st.session_state.notifications = []
        st.session_state.notifications.append(message)

    def get_questions(self) -> List[str]:
        """Get all questions asked so far."""
        return self.questions
    
    def get_pending_questions(self) -> List[str]:
        """Get questions that are waiting for user answers."""
        return st.session_state.get("pending_questions", [])
    
    def has_pending_questions(self) -> bool:
        """Check if there are questions waiting for answers."""
        pending = st.session_state.get("pending_questions", [])
        return len(pending) > 0
    
    def set_answer(self, question: str, answer: str) -> None:
        """Store an answer for a question.
        
        The answer will be available for the next ask() call with the exact same question.
        Once consumed, it won't be reused for other questions.
        """
        if "question_answers_map" not in st.session_state:
            st.session_state.question_answers_map = {}
        st.session_state.question_answers_map[question] = answer
        # Remove from pending if present
        if "pending_questions" in st.session_state:
            if question in st.session_state.pending_questions:
                st.session_state.pending_questions.remove(question)
        # Note: We don't mark it as consumed here - it will be marked when ask() returns it
        # This allows the same answer to be retrieved if ask() is called again with the same question
        # (which shouldn't happen in normal flow, but provides safety)

    def get_multi_answers(self, questions: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Get answers for multiple questions at once.
        
        Args:
            questions: List of dicts with 'question' key (the full question with field identifier)
            
        Returns:
            Dict mapping question text to answer (empty string if no answer), or None if not all questions have been answered
        """
        if "question_answers_map" not in st.session_state:
            return None
        
        answers = {}
        for q_info in questions:
            question = q_info["question"]
            if question in st.session_state.question_answers_map:
                # Include answer even if empty (empty means use document value)
                answer = st.session_state.question_answers_map[question]
                answers[question] = answer.strip() if answer else ""
            else:
                # Question not in map means user hasn't answered yet
                return None
        
        # Return answers only if we have all of them (including empty ones)
        if len(answers) == len(questions):
            return answers
        return None

    def clear(self) -> None:
        """Clear questions and notifications."""
        self.questions = []
        self.notifications = []
        self._question_index = 0
        if "pending_questions" in st.session_state:
            st.session_state.pending_questions = []
        if "question_answers_map" in st.session_state:
            st.session_state.question_answers_map = {}
        if "notifications" in st.session_state:
            st.session_state.notifications = []

