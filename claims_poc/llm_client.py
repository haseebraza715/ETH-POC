"""LLM client helper with OpenRouter/OpenAI + offline fallback."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError

try:  # pragma: no cover - runtime dependency
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional
    ChatOpenAI = None  # type: ignore

DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant helping to process insurance claims.\n"
    "Your role is to:\n"
    "- extract structured information from text,\n"
    "- check for inconsistencies,\n"
    "- ask clear follow-up questions,\n"
    "- and generate concise summaries for human claims handlers.\n"
    "You MUST:\n"
    "- be precise and conservative,\n"
    "- never invent facts that are not clearly supported by the input,\n"
    "- and follow the requested output format exactly (especially JSON).\n"
    "If information is missing or unclear, explicitly mark it as null or \"unknown\" instead of guessing."
)


class RuleBasedExtractor:
    """Heuristic extractor for offline runs."""

    DATE_PATTERN = re.compile(r"(20\d{2}-\d{2}-\d{2})")
    TIME_PATTERN = re.compile(r"(\d{1,2}:\d{2})")

    @classmethod
    def run(cls, prompt: str) -> Dict[str, Optional[str]]:
        text_match = re.search(r"Document text:\n----------------\n(.+)\n----------------", prompt, re.S)
        text = text_match.group(1) if text_match else prompt
        return {
            "date": cls._match(cls.DATE_PATTERN, text),
            "time": cls._match(cls.TIME_PATTERN, text),
            "location": cls._guess_location(text),
            "other_vehicle_plate": cls._guess_plate(text),
            "injuries": "minor" if "injury" in text.lower() else None,
            "description": text[:200],
        }

    @staticmethod
    def _match(pattern: re.Pattern[str], text: str) -> Optional[str]:
        match = pattern.search(text)
        return match.group(1) if match else None

    @staticmethod
    def _guess_location(text: str) -> Optional[str]:
        for line in text.splitlines():
            if "location" in line.lower():
                return line.split(":", 1)[-1].strip()
        return None

    @staticmethod
    def _guess_plate(text: str) -> Optional[str]:
        maybe = re.search(r"([A-Z]{2}\s?\d{3,6})", text)
        return maybe.group(1) if maybe else None


def has_remote_llm() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _call_openai(messages: list[HumanMessage | SystemMessage], temperature: float) -> str:
    if ChatOpenAI is None:
        raise RuntimeError("langchain_openai is not installed")
    model = os.getenv("LANGGRAPH_MODEL", "gpt-4o-mini")
    client = ChatOpenAI(model=model, temperature=temperature)
    message: AIMessage = client.invoke(messages)
    if isinstance(message.content, list):
        text = "".join(
            part["text"]
            for part in message.content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        return text
    return str(message.content)


def _call_openrouter(messages: list[dict[str, str]], temperature: float) -> str:
    """Call OpenRouter API, raising exceptions for caller to handle."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    model = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-small-3.2-24b-instruct:free")
    headers: Dict[str, str] = {}
    referer = os.getenv("OPENROUTER_SITE_URL")
    title = os.getenv("OPENROUTER_SITE_NAME")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    kwargs: Dict[str, Any] = {}
    if headers:
        kwargs["extra_headers"] = headers
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return completion.choices[0].message.content or ""


def _fallback_text(prompt: str) -> str:
    snippet = prompt.strip().splitlines()[-1]
    return f"Offline response: {snippet[:200]}"


def call_llm(
    user_prompt: str,
    *,
    response_format: str = "text",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
    fallback_text: Optional[str] = None,
) -> Union[str, Dict[str, Any]]:
    """Call the configured LLM or fallback heuristic.
    
    Gracefully handles rate limits and API errors by falling back to rule-based extraction.
    """
    if os.getenv("OPENROUTER_API_KEY"):
        try:
            text = _call_openrouter(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature,
            )
        except OpenAIRateLimitError:
            # Rate limited - fall back to rule-based extraction
            if response_format == "json":
                return RuleBasedExtractor.run(user_prompt)
            return fallback_text or _fallback_text(user_prompt)
        except Exception as e:
            # Handle other API errors (429, rate limit messages, etc.)
            error_msg = str(e)
            if "429" in error_msg or "rate" in error_msg.lower() or "rate-limit" in error_msg.lower():
                # Rate limited - fall back to rule-based extraction
                if response_format == "json":
                    return RuleBasedExtractor.run(user_prompt)
                return fallback_text or _fallback_text(user_prompt)
            # For other errors, re-raise to let caller handle
            raise
    elif os.getenv("OPENAI_API_KEY"):
        try:
            text = _call_openai(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
                temperature,
            )
        except OpenAIRateLimitError:
            # Rate limited - fall back to rule-based extraction
            if response_format == "json":
                return RuleBasedExtractor.run(user_prompt)
            return fallback_text or _fallback_text(user_prompt)
        except Exception as e:
            # Handle other API errors (429, rate limit messages, etc.)
            error_msg = str(e)
            if "429" in error_msg or "rate" in error_msg.lower() or "rate-limit" in error_msg.lower():
                # Rate limited - fall back to rule-based extraction
                if response_format == "json":
                    return RuleBasedExtractor.run(user_prompt)
                return fallback_text or _fallback_text(user_prompt)
            # For other errors, re-raise to let caller handle
            raise
    else:
        # No API key - use fallback
        if response_format == "json":
            return RuleBasedExtractor.run(user_prompt)
        text = fallback_text or _fallback_text(user_prompt)
    
    if response_format == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If JSON parsing fails, fall back to rule-based extraction
            return RuleBasedExtractor.run(user_prompt)
    return text

