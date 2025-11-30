
from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

from claims_poc.graph import IOInterface, NeedUserInput


class StreamlitIO(IOInterface):
    """
    This handler stores questions in session state and retrieves answers from session state.
    The Streamlit app must check for pending questions and prompt the user.
    """

    def __init__(self, answer_queue: Optional[List[str]] = None) -> None:
        self.answer_queue = answer_queue or []
        self.questions: List[str] = []
        self.notifications: List[str] = []
        self._question_index = 0
        

        if "pending_questions" not in st.session_state:
            st.session_state.pending_questions = []
        if "question_answers_map" not in st.session_state:
            st.session_state.question_answers_map = {}

    def ask(self, prompt: str) -> str:
        """
        Raises NeedUserInput exception if no answer is available and workflow should pause.
        
        IMPORTANT: Each question should be answered exactly once. Answers are consumed
        after being returned to prevent reuse across multiple questions.
        """
        self.questions.append(prompt)
        

        field = None
        question_text = prompt
        if prompt.startswith("[FIELD:") and "]" in prompt:
            field_end = prompt.index("]")
            field = prompt[7:field_end]
            question_text = prompt[field_end + 1:].strip()
        

        if "consumed_answers" not in st.session_state:
            st.session_state.consumed_answers = set()
        


        if "question_answers_map" in st.session_state:

            if field:

                if prompt in st.session_state.question_answers_map:
                    answer = st.session_state.question_answers_map[prompt]

                    answer_key = f"{prompt}::{answer}"
                    if answer and answer_key not in st.session_state.consumed_answers:

                        st.session_state.consumed_answers.add(answer_key)
                        return answer
            else:

                if prompt in st.session_state.question_answers_map:
                    answer = st.session_state.question_answers_map[prompt]
                    answer_key = f"{prompt}::{answer}"
                    if answer and answer_key not in st.session_state.consumed_answers:
                        st.session_state.consumed_answers.add(answer_key)
                        return answer

                for existing_q, existing_a in st.session_state.question_answers_map.items():
                    if existing_a:

                        if existing_q.startswith("[FIELD:") and "]" in existing_q:
                            continue
                        answer_key = f"{existing_q}::{existing_a}"
                        if answer_key not in st.session_state.consumed_answers and self._questions_similar(prompt, existing_q):
                            st.session_state.consumed_answers.add(answer_key)
                            return existing_a
        

        if self._question_index < len(self.answer_queue):
            answer = self.answer_queue[self._question_index]
            self._question_index += 1
            return answer
        

        if field:
            if "field_question_map" not in st.session_state:
                st.session_state.field_question_map = {}
            st.session_state.field_question_map[field] = prompt
        

        if "pending_questions" not in st.session_state:
            st.session_state.pending_questions = []
        if prompt not in st.session_state.pending_questions:
            st.session_state.pending_questions.append(prompt)
        

        raise NeedUserInput(prompt)
    
    @staticmethod
    def _questions_similar(q1: str, q2: str) -> bool:

        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        if not words1 or not words2:
            return False

        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap > 0.7

    def notify(self, message: str) -> None:
        self.notifications.append(message)
        if "notifications" not in st.session_state:
            st.session_state.notifications = []
        st.session_state.notifications.append(message)

    def get_questions(self) -> List[str]:
        return self.questions
    
    def get_pending_questions(self) -> List[str]:
        return st.session_state.get("pending_questions", [])
    
    def has_pending_questions(self) -> bool:
        pending = st.session_state.get("pending_questions", [])
        return len(pending) > 0
    
    def set_answer(self, question: str, answer: str) -> None:
        """
        The answer will be available for the next ask() call with the exact same question.
        Once consumed, it won't be reused for other questions.
        """
        if "question_answers_map" not in st.session_state:
            st.session_state.question_answers_map = {}
        st.session_state.question_answers_map[question] = answer

        if "pending_questions" in st.session_state:
            if question in st.session_state.pending_questions:
                st.session_state.pending_questions.remove(question)




    def get_multi_answers(self, questions: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """
        Args:
            questions: List of dicts with 'question' key (the full question with field identifier)
            
        Returns:
            Dict mapping question text to answer (empty string if no answer), or None if not all questions have been answered
        """
        answers = {}
        

        if "question_answers_map" in st.session_state:
            for q_info in questions:
                question = q_info["question"]
                if question in st.session_state.question_answers_map:
                    answer = st.session_state.question_answers_map[question]
                    answers[question] = answer.strip() if answer else ""
        


        if len(answers) < len(questions) and "field_answers_map" in st.session_state:
            field_answers = st.session_state.field_answers_map
            for q_info in questions:
                question = q_info["question"]
                field = q_info["field"]

                if question not in answers and field in field_answers:
                    answer = field_answers[field]
                    answers[question] = answer.strip() if answer else ""

                    if "question_answers_map" not in st.session_state:
                        st.session_state.question_answers_map = {}
                    st.session_state.question_answers_map[question] = answer
        

        if len(answers) == len(questions):
            return answers
        return None

    def clear_field_answers(self) -> None:
        if "field_answers_map" in st.session_state:
            del st.session_state.field_answers_map

    def clear(self) -> None:
        self.questions = []
        self.notifications = []
        self._question_index = 0
        if "pending_questions" in st.session_state:
            st.session_state.pending_questions = []
        if "question_answers_map" in st.session_state:
            st.session_state.question_answers_map = {}
        if "field_answers_map" in st.session_state:
            st.session_state.field_answers_map = {}
        if "notifications" in st.session_state:
            st.session_state.notifications = []

