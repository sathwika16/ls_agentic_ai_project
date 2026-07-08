import os
print("LOADED:", __file__)
print("########## AGENTS.PY LOADED ##########")

import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_API_KEY
print(">>> NEW agents.py LOADED <<<")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in week4/.env")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY,
)

planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a study planner. Create a concise, practical revision plan."),
        ("user", "{query}"),
    ]
)

tutor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a tutor. Answer ONLY using the provided context. If the context is insufficient, clearly state that.",
        ),
        (
            "user",
            """Question:
{query}

Context:
{context}
""",
        ),
    ]
)

validator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an answer validator.

Return ONLY valid JSON.

Example:

{
  "confidence": 0.92,
  "needs_human_review": false,
  "reason": "The answer is supported by the retrieved context."
}

Rules:
- confidence must be between 0 and 1.
- Do NOT wrap the JSON inside ```json.
- Do NOT explain anything.
- Output ONLY JSON.
""",
        ),
        (
            "user",
            """
Question:
{query}

Context:
{context}

Answer:
{answer}
""",
        ),
    ]
)

fallback_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful tutor. Explain politely when the uploaded notes do not contain enough information.",
        ),
        ("user", "{query}"),
    ]
)


def _safe(chain, payload, fallback_text=""):
    try:
        return chain.invoke(payload).content
    except Exception as e:
        print("LLM ERROR:", e)
        return fallback_text


def plan_revision(query: str) -> str:
    return _safe(
        planner_prompt | llm,
        {"query": query},
        f"""Revision Plan

• Understand {query}
• Read notes
• Make short notes
• Solve practice questions
• Revise weak topics
• Final revision""",
    )


def tutor_answer(query: str, context: str) -> str:
    if not context.strip():
        return (
            "I could not find relevant uploaded notes for this question. "
            "Please upload a PDF first."
        )

    return _safe(
        tutor_prompt | llm,
        {
            "query": query,
            "context": context,
        },
        "Unable to generate an answer.",
    )


def validate_answer(query: str, context: str, answer: str):

    raw = _safe(
        validator_prompt | llm,
        {
            "query": query,
            "context": context,
            "answer": answer,
        },
        "",
    )

    print("Validator Output:")
    print(raw)

    try:

        raw = raw.strip()

        if raw.startswith("```"):
            raw = raw.replace("```json", "")
            raw = raw.replace("```", "")
            raw = raw.strip()

        data = json.loads(raw)

        confidence = float(data.get("confidence", 0.8))
        needs_human_review = bool(data.get("needs_human_review", False))
        reason = data.get("reason", "")

        return {
            "confidence": confidence,
            "needs_human_review": needs_human_review,
            "reason": reason,
        }

    except Exception as e:

        print("Validation parse failed:", e)

        # If retrieval exists and answer is meaningful,
        # trust it instead of rejecting it.

        if context.strip() and len(answer.strip()) > 50:

            return {
                "confidence": 0.80,
                "needs_human_review": False,
                "reason": "Answer generated using retrieved context.",
            }

        return {
            "confidence": 0.30,
            "needs_human_review": True,
            "reason": "Insufficient supporting context.",
        }


def fallback_answer(query: str):

    return _safe(
        fallback_prompt | llm,
        {"query": query},
        f"""I could not retrieve enough information for:

{query}

Please upload a relevant PDF or ask a more specific question.""",
    )