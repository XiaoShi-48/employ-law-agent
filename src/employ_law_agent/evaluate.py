import time
import pandas as pd
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from employ_law_agent.config import GENERATOR_MODEL, INSUFFICIENT, JUDGE_MODEL, QUESTIONS_PATH, RESULTS_PATH
from employ_law_agent.graph import (
    GENERATE_PROMPT,
    GROUNDING_PROMPT,
    build_agent,
    format_context,
    get_citations,
    initial_state,
)
from employ_law_agent.schemas import GroundingSchema
from employ_law_agent.vectorstore import retrieve_docs


def _parse_sections(x) -> set[str]:
    return {s.strip() for s in str(x).split(";") if s.strip() and s.strip() != "nan"}


def run_evaluation() -> pd.DataFrame:
    load_dotenv()
    questions = pd.read_csv(QUESTIONS_PATH)

    baseline_chain = GENERATE_PROMPT | ChatOpenRouter(model=GENERATOR_MODEL)
    judge_chain = GROUNDING_PROMPT | ChatOpenRouter(model=JUDGE_MODEL).with_structured_output(GroundingSchema)
    agent = build_agent()

    def baseline_answer(question: str) -> dict:
        start = time.time()
        docs = retrieve_docs(question)
        answer = baseline_chain.invoke({"context": format_context(docs), "question": question, "feedback": ""}).content.strip()
        return {
            "answer": answer,
            "documents": docs,
            "abstained": INSUFFICIENT in answer,
            "latency": time.time() - start,
        }

    def score(row, docs, answer: str, abstained: bool) -> dict:
        expected = _parse_sections(row["expected_sections"])
        retrieved = {d["section_no"] for d in docs}
        cited = {c[2:] for c in get_citations(answer)}
        should_abstain = row["expected_behaviour"] == "abstain"

        out = {
            "behaviour_ok": abstained == should_abstain,
            "retrieval_hit": None,
            "citation_ok": None,
            "faithful": None,
        }
        if not should_abstain and expected:
            out["retrieval_hit"] = bool(expected & retrieved)
            out["citation_ok"] = bool(expected & cited) and cited <= (expected | retrieved)
        if not abstained and docs:
            verdict = judge_chain.invoke({"context": format_context(docs), "answer": answer})
            out["faithful"] = verdict.supported
        return out

    rows = []
    for _, row in questions.iterrows():
        q = row["question"]
        for attempt in range(3):
            try:
                b = baseline_answer(q)
                start = time.time()
                a = agent.invoke(initial_state(q), {"configurable": {"thread_id": f"eval-{row['id']}"}})
                a_latency = time.time() - start
                a_abstained = a["status"] == "abstained"

                for name, docs, answer, abstained, latency in [
                    ("baseline", b["documents"], b["answer"], b["abstained"], b["latency"]),
                    ("agentic", a["documents"], a["answer"], a_abstained, a_latency),
                ]:
                    s = score(row, docs, answer, abstained)
                    rows.append({
                        "id": row["id"],
                        "category": row["category"],
                        "system": name,
                        "abstained": abstained,
                        "latency": latency,
                        "answer": answer[:300],
                        **s,
                    })
                break
            except Exception as exc:
                print(f"Retrying id={row['id']} (attempt {attempt + 1}/3): {exc}")
                time.sleep(5 * (attempt + 1))

        pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
        print(f"Completed question {row['id']}/{len(questions)}")

    return pd.DataFrame(rows)


def main() -> None:
    results = run_evaluation()

    def frac(s):
        s = s.dropna()
        return f"{int(s.sum())}/{len(s)}"

    summary = results.groupby("system").agg(
        retrieval_hit=("retrieval_hit", frac),
        citation_ok=("citation_ok", frac),
        faithful=("faithful", frac),
        behaviour_ok=("behaviour_ok", frac),
        avg_latency_s=("latency", "mean"),
    ).round(2)
    print("\n=== Evaluation Summary ===")
    print(summary)


if __name__ == "__main__":
    main()