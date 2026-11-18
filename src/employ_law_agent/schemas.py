import json
from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field, field_validator


class ScopeSchema(BaseModel):
    category: Literal["in_scope", "sabah_sarawak", "out_of_scope"] = Field(
        ..., description="Is the question in scope for the Employment Act 1955?"
    )


class GradeSchema(BaseModel):
    relevant_ids: List[int] = Field(
        ..., description="ids of the sections that are useful for answering the question; empty list if none"
    )

    @field_validator("relevant_ids", mode="before")
    @classmethod
    def _parse_ids(cls, v):
        return json.loads(v) if isinstance(v, str) else v


class GroundingSchema(BaseModel):
    supported: bool = Field(
        ..., description="True if every factual claim in the answer is supported or directly entailed by the context"
    )
    unsupported_claims: List[str] = Field(
        ..., description="the claims that are not supported; empty list if none"
    )

    @field_validator("unsupported_claims", mode="before")
    @classmethod
    def _parse_claims(cls, v):
        return json.loads(v) if isinstance(v, str) else v


class GraphState(TypedDict):
    question: str
    search_query: str
    documents: List[dict]
    rewrite_count: int
    regen_count: int
    answer: str
    feedback: str
    status: str
    abstain_reason: str
    path: List[str]


def initial_state(question: str) -> GraphState:
    return {
        "question": question,
        "search_query": question,
        "documents": [],
        "rewrite_count": 0,
        "regen_count": 0,
        "answer": "",
        "feedback": "",
        "status": "",
        "abstain_reason": "",
        "path": [],
    }