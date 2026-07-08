import os
print("GRAPH:", __file__)
print("########## GRAPH.PY LOADED ##########")

from langgraph.graph import StateGraph, END

from state import CopilotState
from tools import retrieve_docs
from agents import (
    plan_revision,
    tutor_answer,
    validate_answer,
    fallback_answer,
)
print(">>> NEW graph.py LOADED <<<")

MAX_RETRIES = 2


def supervisor_node(state: CopilotState):
    query = state.get("user_query", "").lower()

    planner_keywords = [
        "plan",
        "schedule",
        "revision",
        "revise",
        "study plan",
        "timetable",
        "prepare",
    ]

    if any(word in query for word in planner_keywords):
        return {"route": "planner"}

    return {"route": "retrieve"}


def route_from_state(state: CopilotState):
    return state.get("route", "retrieve")


def planner_node(state: CopilotState):
    answer = plan_revision(state["user_query"])

    return {
        "final_answer": answer,
        "confidence": 0.95,
        "needs_human_review": False,
        "route": "done",
    }


def retrieve_node(state: CopilotState):
    context = retrieve_docs(state["user_query"])

    if not context:
        return {
            "retrieved_context": "",
            "route": "fallback",
            "last_error": "No relevant context found.",
        }

    return {
        "retrieved_context": context,
        "route": "tutor",
    }


def tutor_node(state: CopilotState):
    answer = tutor_answer(
        state["user_query"],
        state["retrieved_context"],
    )

    return {
        "draft_answer": answer,
        "route": "validate",
    }


def validate_node(state: CopilotState):

    result = validate_answer(
        state["user_query"],
        state["retrieved_context"],
        state["draft_answer"],
    )

    confidence = result.get("confidence", 0.8)
    needs_review = result.get("needs_human_review", False)

    # Good answer -> finish
    if not needs_review:
        return {
            "confidence": confidence,
            "needs_human_review": False,
            "final_answer": state["draft_answer"],
            "route": "done",
        }

    # Retry once or twice
    if state.get("error_count", 0) < MAX_RETRIES:

        return {
            "confidence": confidence,
            "needs_human_review": True,
            "error_count": state.get("error_count", 0) + 1,
            "route": "retry",
            "last_error": result.get("reason", ""),
        }

    # Don't replace the answer with fallback.
    return {
        "confidence": confidence,
        "needs_human_review": True,
        "final_answer": state["draft_answer"],
        "last_error": result.get("reason", ""),
        "route": "done",
    }


def retry_node(state: CopilotState):

    reformulated = (
        state["user_query"]
        + " definition explanation summary key points"
    )

    context = retrieve_docs(reformulated, k=10)

    if not context:
        return {
            "route": "fallback",
            "last_error": "Retry retrieval failed.",
        }

    answer = tutor_answer(
        state["user_query"],
        context,
    )

    return {
        "retrieved_context": context,
        "draft_answer": answer,
        "route": "validate",
    }


def fallback_node(state: CopilotState):

    return {
        "final_answer": fallback_answer(state["user_query"]),
        "confidence": 0.30,
        "needs_human_review": True,
        "route": "human_review",
    }


def human_review_node(state: CopilotState):

    return {
        "final_answer": state.get(
            "final_answer",
            state.get("draft_answer", ""),
        ),
        "needs_human_review": True,
        "route": "done",
    }


graph = StateGraph(CopilotState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("planner", planner_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("tutor", tutor_node)
graph.add_node("validate", validate_node)
graph.add_node("retry", retry_node)
graph.add_node("fallback", fallback_node)
graph.add_node("human_review", human_review_node)

graph.set_entry_point("supervisor")

graph.add_conditional_edges(
    "supervisor",
    route_from_state,
    {
        "planner": "planner",
        "retrieve": "retrieve",
    },
)

graph.add_conditional_edges(
    "retrieve",
    route_from_state,
    {
        "tutor": "tutor",
        "fallback": "fallback",
    },
)

graph.add_edge("planner", END)
graph.add_edge("tutor", "validate")

graph.add_conditional_edges(
    "validate",
    route_from_state,
    {
        "retry": "retry",
        "done": END,
    },
)

graph.add_conditional_edges(
    "retry",
    route_from_state,
    {
        "validate": "validate",
        "fallback": "fallback",
    },
)

graph.add_conditional_edges(
    "fallback",
    route_from_state,
    {
        "human_review": "human_review",
    },
)

graph.add_conditional_edges(
    "human_review",
    route_from_state,
    {
        "done": END,
    },
)

graph_app = graph.compile()