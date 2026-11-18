import json
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from employ_law_agent.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_CHUNK_CHARS,
    SECTIONS_PATH,
    SOURCE_LABEL,
    TOP_K,
    VERSION_DATE,
)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def build_chunks(sections: list[dict[str, str]]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=150,
        separators=["\n", ". ", "; ", " "],
    )
    documents: list[Document] = []
    for s in sections:
        header = f"Section {s['no']}. {s['title']}"
        parts = splitter.split_text(s["text"]) if len(s["text"]) > MAX_CHUNK_CHARS else [s["text"]]
        for i, part in enumerate(parts):
            tag = f" (part {i + 1}/{len(parts)})" if len(parts) > 1 else ""
            documents.append(
                Document(
                    page_content=f"{header}{tag}\n{part}",
                    metadata={
                        "section_no": s["no"],
                        "section_title": s["title"],
                        "source": SOURCE_LABEL,
                        "version_date": VERSION_DATE,
                        "part": i,
                    },
                )
            )
    return documents


def rebuild_vectorstore() -> int:
    with open(SECTIONS_PATH, encoding="utf-8") as f:
        sections = json.load(f)

    documents = build_chunks(sections)
    store = get_vectorstore()
    store.delete_collection()
    get_vectorstore.cache_clear()

    fresh_store = get_vectorstore()
    fresh_store.add_documents(documents)
    return fresh_store._collection.count()


def retrieve_docs(query: str, k: int = TOP_K) -> list[dict[str, str]]:
    results = get_vectorstore().similarity_search(query, k=k)
    return [
        {
            "content": r.page_content,
            "section_no": r.metadata["section_no"],
            "section_title": r.metadata["section_title"],
        }
        for r in results
    ]


def main() -> None:
    count = rebuild_vectorstore()
    print(f"Indexed {count} chunks into {CHROMA_DIR}")


if __name__ == "__main__":
    main()