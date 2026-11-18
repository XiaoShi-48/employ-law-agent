import sys
from dotenv import load_dotenv
from employ_law_agent.graph import build_agent, get_citations
from employ_law_agent.schemas import initial_state


def main() -> None:
    load_dotenv()
    question = " ".join(sys.argv[1:]) or input("Ask an Employment Act 1955 question: ")
    agent = build_agent()
    result = agent.invoke(initial_state(question), {"configurable": {"thread_id": "cli"}})

    print(f"\nStatus: {result['status']} | Path: {' -> '.join(result['path'])}")
    print(f"Citations: {get_citations(result['answer'])}\n")
    print(result["answer"])