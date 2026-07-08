from typing import TypedDict

class CopilotState(TypedDict, total=False):
    user_query: str
    retrieved_context: str
    draft_answer: str
    final_answer: str
    confidence: float
    needs_human_review: bool
    error_count: int
    route: str
    last_error: str