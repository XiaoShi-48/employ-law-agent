import re
from functools import lru_cache
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from employ_law_agent.config import (
    CHECKER_MODEL,
    GENERATOR_MODEL,
    INSUFFICIENT,
    MAX_REGENS,
    MAX_REWRITES,
)
from employ_law_agent.schemas import (
    GradeSchema,
    GraphState,
    GroundingSchema,
    ScopeSchema,
    initial_state,
)
from employ_law_agent.vectorstore import retrieve_docs

ABSTAIN_MESSAGES = {
    "sabah_sarawak": "Sabah and Sarawak have their own labour ordinances, which this system does not cover.",
    "out_of_scope": "This question is outside the scope of the Employment Act 1955 knowledge base.",
    "no_evidence": "I could not find enough reliable evidence in the Employment Act 1955 to answer this.",
    "unsupported": "I could not produce an answer that was fully supported by the source text.",
}

SCOPE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You classify questions for a Malaysian employment-law assistant. "
               "The knowledge base covers ONLY the Employment Act 1955 (Peninsular Malaysia and Labuan). "
               "in_scope: about employment rights or duties that this Act could govern. "
               "sabah_sarawak: asks specifically about Sabah or Sarawak (they have separate labour ordinances). "
               "out_of_scope: anything else, such as tax, immigration, other countries or unrelated topics."),
    ("human", "{question}"),
])

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You grade retrieved legal sections for relevance to a question. "
               "Return the ids of the sections that contain information useful for answering it. "
               "Return an empty list if none are useful."),
    ("human", "Question: {question}\n\nSections:\n{numbered_docs}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's question as a short search query using the wording of statutes, "
               "for example: hours of work, termination notice, maternity leave entitlement. "
               "Keep it under 20 words. Return ONLY the query."),
    ("human", "{question}"),
])

GROUNDING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You check an answer against its context. supported is true if EVERY factual claim in the answer "
               "is supported by or directly entailed by the context, and every citation label points to a context item "
               "that supports the claim. Accurate paraphrasing of statutory rules is allowed. "
               "Otherwise list the unsupported claims."),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
])

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You explain the Malaysian Employment Act 1955. Answer ONLY using the context. "
               "Cite every claim with its label in square brackets, exactly as given, for example [s.60A]. "
               "Do not cite anything that is not in the context. "
               "If the context does not contain enough information, reply exactly: INSUFFICIENT_EVIDENCE. "
               "This is general information, not legal advice."),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n{feedback}"),
])


def format_context(docs: list[dict[str, str]]) -> str:
    return "\n\n".join(f"[s.{d['section_no']}] {d['section_title']}\n{d['content']}" for d in docs)


def get_citations(answer: str) -> list[str]:
    matches = re.findall(r"\[s\.([0-9]+[A-Za-z]*)", answer)
    return sorted({f"s.{m}" for m in matches})


@lru_cache(maxsize=1)
def _get_chains():
    llm = ChatOpenRouter(model=GENERATOR_MODEL)
    checker_llm = ChatOpenRouter(model=CHECKER_MODEL)
    return {
        "scope": SCOPE_PROMPT | checker_llm.with_structured_output(ScopeSchema),
        "grade": GRADE_PROMPT | checker_llm.with_structured_output(GradeSchema),
        "rewrite": REWRITE_PROMPT | checker_llm,
        "grounding": GROUNDING_PROMPT | checker_llm.with_structured_output(GroundingSchema),
        "generate": GENERATE_PROMPT | llm,
    }


def check_scope(state: GraphState) -> dict:
    result = _get_chains()["scope"].invoke({"question": state["question"]})
    if result.category != "in_scope":
        return {
            "status": "abstained",
            "abstain_reason": result.category,
            "path": state["path"] + ["check_scope(out)"],
        }
    return {"path": state["path"] + ["check_scope"]}


def retrieve(state: GraphState) -> dict:
    docs = retrieve_docs(state["search_query"])
    return {"documents": docs, "path": state["path"] + ["retrieve"]}


def grade_documents(state: GraphState) -> dict:
    docs = state["documents"]
    numbered_docs = "\n\n".join(
        f"id={i}\n[s.{d['section_no']}] {d['section_title']}\n{d['content'][:800]}"
        for i, d in enumerate(docs)
    )
    result = _get_chains()["grade"].invoke({"question": state["question"], "numbered_docs": numbered_docs})
    kept = [docs[i] for i in result.relevant_ids if 0 <= i < len(docs)]
    return {"documents": kept, "path": state["path"] + [f"grade({len(kept)}/{len(docs)})"]}


def rewrite_query(state: GraphState) -> dict:
    new_query = _get_chains()["rewrite"].invoke({"question": state["question"]}).content.strip()
    return {
        "search_query": new_query,
        "rewrite_count": state["rewrite_count"] + 1,
        "path": state["path"] + ["rewrite_query"],
    }


def generate(state: GraphState) -> dict:
    feedback_text = ""
    if state["feedback"]:
        feedback_text = f"A previous draft had unsupported claims: {state['feedback']}. Fix this."

    answer = _get_chains()["generate"].invoke({
        "context": format_context(state["documents"]),
        "question": state["question"],
        "feedback": feedback_text,
    }).content.strip()

    if INSUFFICIENT in answer:
        return {
            "answer": answer,
            "status": "abstained",
            "abstain_reason": "no_evidence",
            "path": state["path"] + ["generate(insufficient)"],
        }
    return {"answer": answer, "path": state["path"] + ["generate"]}


def check_grounding(state: GraphState) -> dict:
    result = _get_chains()["grounding"].invoke({
        "context": format_context(state["documents"]),
        "answer": state["answer"],
    })
    if result.supported:
        return {"status": "answered", "path": state["path"] + ["grounding(ok)"]}

    new_count = state["regen_count"] + 1
    if new_count > MAX_REGENS:
        return {
            "status": "abstained",
            "abstain_reason": "unsupported",
            "regen_count": new_count,
            "path": state["path"] + ["grounding(fail)"],
        }
    return {
        "feedback": "; ".join(result.unsupported_claims),
        "regen_count": new_count,
        "path": state["path"] + ["grounding(fail)"],
    }


def abstain(state: GraphState) -> dict:
    message = ABSTAIN_MESSAGES.get(state["abstain_reason"], ABSTAIN_MESSAGES["no_evidence"])
    message += " Please check with the Labour Department (JTKSM) or a qualified professional."
    return {"answer": message, "status": "abstained", "path": state["path"] + ["abstain"]}


def route_after_scope(state: GraphState) -> str:
    return "abstain" if state["status"] == "abstained" else "retrieve"


def route_after_grade(state: GraphState) -> str:
    if state["documents"]:
        return "generate"
    return "rewrite_query" if state["rewrite_count"] < MAX_REWRITES else "abstain"


def route_after_generate(state: GraphState) -> str:
    return "abstain" if state["status"] == "abstained" else "check_grounding"


def route_after_grounding(state: GraphState) -> str:
    if state["status"] == "answered":
        return "end"
    if state["status"] == "abstained":
        return "abstain"
    return "generate"


def build_agent():
    graph = StateGraph(GraphState)
    graph.add_node("check_scope", check_scope)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("check_grounding", check_grounding)
    graph.add_node("abstain", abstain)

    graph.add_edge(START, "check_scope")
    graph.add_conditional_edges("check_scope", route_after_scope, {"retrieve": "retrieve", "abstain": "abstain"})
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate": "generate", "rewrite_query": "rewrite_query", "abstain": "abstain"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {"check_grounding": "check_grounding", "abstain": "abstain"},
    )
    graph.add_conditional_edges(
        "check_grounding",
        route_after_grounding,
        {"end": END, "abstain": "abstain", "generate": "generate"},
    )
    graph.add_edge("abstain", END)

    return graph.compile(checkpointer=InMemorySaver())


__all__ = ["build_agent", "initial_state", "format_context", "get_citations", "GENERATE_PROMPT", "GROUNDING_PROMPT"]