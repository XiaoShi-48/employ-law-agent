import json
import re
from pathlib import Path
from pypdf import PdfReader
from employ_law_agent.config import PDF_PATH, SECTIONS_PATH


def is_header_line(line: str) -> bool:
    parts = line.split()
    return (
        line.isdigit()
        or line in ("Employment", "Laws of Malaysia", "Act 265")
        or (len(parts) == 2 and parts[0] == "Employment" and parts[1].isdigit())
        or line.endswith("Laws of Malaysia Act 265")
    )


def clean_page(text: str) -> list[str]:
    raw = [l.strip() for l in text.splitlines() if l.strip()]
    for _ in range(4):
        if raw and is_header_line(raw[0]):
            raw.pop(0)

    kept = []
    for line in raw:
        if line.startswith("*NOTE"):
            break
        if line.startswith("WJW"):
            continue
        if line[0].isdigit() and ("(Deleted" in line or "(Omitted" in line):
            continue
        kept.append(line)
    return kept


def parse_toc(lines: list[str]) -> tuple[dict[str, str], list[str], int]:
    start = lines.index("ARRANGEMENT OF SECTIONS")
    end = next(i for i, l in enumerate(lines) if l.startswith("An Act relating to employment"))
    toc_lines = lines[start + 1:end]

    toc: dict[str, str] = {}
    order: list[str] = []
    last: str | None = None

    for line in toc_lines:
        skip = (
            line in ("Section", "Act 265", "First Schedule", "Second Schedule")
            or (line.startswith("Part ") and len(line) < 12)
            or (line.isupper() and len(line) > 3)
        )
        if skip:
            continue

        cleaned = re.sub(r"\b(\d+)\s+([a-zA-Z]{1,3})\.", r"\1\2.", line)
        parts = re.split(r"(?=\b\d+[a-zA-Z]{0,3}\.\s+)", cleaned)

        for part in parts:
            part = part.strip()
            if not part:
                continue
            number, dot, title = part.partition(". ")
            if dot and number[:1].isdigit() and number.isalnum() and len(number) <= 5:
                last = number.lower()
                toc[last] = title.strip()
                order.append(last)
            elif last:
                toc[last] += " " + part

    return toc, order, end


def join_lines(lines: list[str]) -> str:
    out: list[str] = []
    for line in lines:
        if not out or (line.startswith("(") and line[1:2].isdigit()):
            out.append(line)
        else:
            out[-1] += " " + line
    return "\n".join(out)


def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def extract_sections(pdf_path: Path = PDF_PATH) -> list[dict[str, str]]:
    pages = [p.extract_text() or "" for p in PdfReader(str(pdf_path)).pages]
    lines: list[str] = []
    for page in pages:
        lines.extend(clean_page(page))

    toc, order, end = parse_toc(lines)
    body = lines[end + 1:]

    sections: list[dict] = []
    current: dict | None = None
    pos = -1

    for line in body:
        if line.strip().upper() in ("FIRST SCHEDULE", "SECOND SCHEDULE"):
            break

        norm_line = re.sub(r"^(\d+)\s+([a-zA-Z]{1,3})\.", r"\1\2.", line)
        number, dot, rest = norm_line.partition(". ")

        if dot and number.lower() in toc:
            num_key = number.lower()
            idx = order.index(num_key)
            if 0 < idx - pos <= 5:
                if current:
                    sections.append(current)
                current = {
                    "no": num_key.upper(),
                    "title": toc[num_key],
                    "lines": [rest],
                }
                pos = idx
                continue

        if current:
            current["lines"].append(line)

    if current:
        sections.append(current)

    for i in range(len(sections) - 1):
        ls = sections[i]["lines"]
        next_title = _norm(sections[i + 1]["title"])
        for k in range(1, 5):
            if _norm(" ".join(ls[-k:])) == next_title:
                del ls[-k:]
                break
        while ls and (
            (ls[-1].startswith("Part ") and len(ls[-1]) < 12)
            or (ls[-1].isupper() and len(ls[-1]) > 3)
        ):
            ls.pop()

    return [
        {"no": s["no"], "title": s["title"], "text": join_lines(s["lines"])}
        for s in sections
    ]


def main() -> None:
    sections = extract_sections()
    SECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(sections)} sections to {SECTIONS_PATH}")


if __name__ == "__main__":
    main()