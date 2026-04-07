import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader
from sqlalchemy.orm import Session

DATA_PATH = Path(__file__).resolve().parents[1] / "data"
PYQ_INDEX_PATH = DATA_PATH / "pyq_index.json"
MANIFEST_PATH = DATA_PATH / "pdf_manifest.json"

STOPWORDS = {
    "about",
    "also",
    "and",
    "any",
    "are",
    "briefly",
    "can",
    "define",
    "describe",
    "discuss",
    "each",
    "explain",
    "for",
    "from",
    "give",
    "how",
    "into",
    "mark",
    "marks",
    "module",
    "note",
    "question",
    "short",
    "the",
    "this",
    "two",
    "what",
    "when",
    "where",
    "which",
    "with",
    "write",
}


def _load_index() -> dict:
    if not PYQ_INDEX_PATH.exists():
        return {"questions": []}
    try:
        payload = json.loads(PYQ_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"questions": []}
    questions = payload.get("questions", [])
    return {"questions": questions if isinstance(questions, list) else []}


def _save_index(payload: dict) -> None:
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    PYQ_INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _load_manifest_pyqs() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [
        item for item in payload.get("files", [])
        if isinstance(item, dict) and item.get("type") == "pyq" and item.get("stored_relative_path")
    ]


def _clean_ocr_text(text: str) -> str:
    cleaned = (text or "").replace("\uf0b7", " ").replace("\uf0a7", " ")
    cleaned = cleaned.replace("â€¢", " ").replace("Â", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\b([A-Za-z])\s+-\s+([A-Za-z])\b", r"\1-\2", cleaned)
    return cleaned.strip()


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return _clean_ocr_text("\n".join((page.extract_text() or "") for page in reader.pages[:25]))


def _normalize_question(question: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    return re.sub(r"\s+", " ", compact)


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9]{2,}", text.lower())
        if token not in STOPWORDS and not token.isdigit()
    ]


def _keywords(question: str, limit: int = 8) -> list[str]:
    counter = Counter(_tokenize(question))
    return [token for token, _ in counter.most_common(limit)]


def _infer_topic(question: str, subject: str, keywords: list[str]) -> str:
    if not keywords:
        return subject
    return " ".join(keywords[:3]).title()


def _extract_questions(text: str) -> list[str]:
    cleaned = _clean_ocr_text(text)
    cleaned = re.sub(
        r"(?im)^\s*(?:q(?:uestion)?\.?\s*)?(\d{1,2})\s*[\).:-]\s+",
        r"\n@@QUESTION@@ \1. ",
        cleaned,
    )
    cleaned = re.sub(r"(?i)\b(part\s+[abc])\b", r"\n\1", cleaned)
    parts = [part.strip() for part in cleaned.split("@@QUESTION@@") if part.strip()]

    questions: list[str] = []
    seen: set[str] = set()
    for part in parts:
        part = re.split(r"(?im)^\s*(?:answer\s+any|time\s*:|duration\s*:|max(?:imum)?\s+marks\s*:)", part)[0]
        question = re.sub(r"\s+", " ", part).strip(" -;:")
        if len(question) < 25:
            continue
        if "?" not in question and not re.search(r"\b(define|explain|describe|derive|write|compare|differentiate|prove|list|draw|discuss|find)\b", question, re.IGNORECASE):
            continue
        normalized = _normalize_question(question)
        if normalized in seen:
            continue
        seen.add(normalized)
        questions.append(question)

    if questions:
        return questions

    fallback = []
    for match in re.finditer(r"([^.?]*(?:\?|(?:explain|define|describe|derive|write|compare|differentiate|prove|draw|discuss)[^.]{20,}\.))", cleaned, re.IGNORECASE):
        question = re.sub(r"\s+", " ", match.group(1)).strip(" -;:")
        if len(question) >= 25:
            fallback.append(question)
    return fallback


def index_pyq_pdf(pdf_path: str | Path, metadata: dict) -> dict:
    path = Path(pdf_path)
    relative_path = str(path.relative_to(DATA_PATH)).replace("\\", "/")
    subject = metadata.get("subject") or "Unknown Subject"
    text = _extract_pdf_text(path)
    extracted_questions = _extract_questions(text)

    index = _load_index()
    existing = [
        item for item in index.get("questions", [])
        if item.get("stored_relative_path") != relative_path
    ]

    now = datetime.now(timezone.utc).isoformat()
    records = []
    for question in extracted_questions:
        normalized = _normalize_question(question)
        words = _keywords(question)
        question_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        records.append(
            {
                "id": question_hash,
                "question": question,
                "normalized_question": normalized,
                "subject": subject,
                "type": "pyq",
                "source": "PYQ",
                "inferred_topic": _infer_topic(question, subject, words),
                "keywords": words,
                "frequency_score": 1,
                "module_number": metadata.get("module_number"),
                "subject_code": metadata.get("subject_code"),
                "year": metadata.get("year"),
                "filename": metadata.get("filename") or metadata.get("source_filename") or path.name,
                "stored_relative_path": relative_path,
                "indexed_at": now,
            }
        )

    all_records = existing + records
    frequencies = Counter(item.get("normalized_question", "") for item in all_records)
    for item in all_records:
        item["frequency_score"] = frequencies.get(item.get("normalized_question", ""), 1)

    index["questions"] = sorted(
        all_records,
        key=lambda item: (
            str(item.get("subject", "")).lower(),
            -int(item.get("frequency_score") or 1),
            str(item.get("question", "")).lower(),
        ),
    )
    _save_index(index)
    return {"question_count": len(records), "stored_relative_path": relative_path}


def get_pyq_catalog(db: Session | None = None) -> list[dict]:
    index = _load_index()
    grouped: dict[str, dict] = {}
    if db is not None:
        from . import models

        papers = (
            db.query(models.PreviousQuestionPaper)
            .order_by(
                models.PreviousQuestionPaper.subject.asc(),
                models.PreviousQuestionPaper.year.desc(),
                models.PreviousQuestionPaper.filename.asc(),
            )
            .all()
        )
        for db_paper in papers:
            subject_entry = grouped.setdefault(
                db_paper.subject,
                {"name": db_paper.subject, "subject_code": db_paper.subject_code, "papers": {}, "question_count": 0},
            )
            if db_paper.subject_code and not subject_entry.get("subject_code"):
                subject_entry["subject_code"] = db_paper.subject_code
            subject_entry["question_count"] += int(db_paper.question_count or 0)
            subject_entry["papers"][db_paper.file_path] = {
                "name": Path(db_paper.filename).stem,
                "filename": db_paper.filename,
                "path": db_paper.file_path,
                "subject_code": db_paper.subject_code,
                "year": db_paper.year,
                "question_count": int(db_paper.question_count or 0),
                "top_topics": Counter(),
            }

    for manifest_item in _load_manifest_pyqs():
        subject = manifest_item.get("subject") or "Unknown Subject"
        subject_entry = grouped.setdefault(
            subject,
            {
                "name": subject,
                "subject_code": manifest_item.get("subject_code"),
                "papers": {},
                "question_count": 0,
            },
        )
        if manifest_item.get("subject_code") and not subject_entry.get("subject_code"):
            subject_entry["subject_code"] = manifest_item["subject_code"]
        path = manifest_item.get("stored_relative_path")
        subject_entry["papers"].setdefault(
            path,
            {
                "name": Path(path).stem,
                "filename": manifest_item.get("filename") or manifest_item.get("source_filename") or Path(path).name,
                "path": path,
                "subject_code": manifest_item.get("subject_code"),
                "year": manifest_item.get("year"),
                "question_count": 0,
                "top_topics": Counter(),
            },
        )

    for item in index.get("questions", []):
        if item.get("type") != "pyq":
            continue
        subject = item.get("subject") or "Unknown Subject"
        subject_entry = grouped.setdefault(subject, {"name": subject, "subject_code": None, "papers": {}, "question_count": 0})
        path = item.get("stored_relative_path")
        if not path:
            continue
        paper = subject_entry["papers"].setdefault(
            path,
            {
                "name": Path(path).stem,
                "filename": item.get("filename") or Path(path).name,
                "path": path,
                "subject_code": item.get("subject_code"),
                "year": item.get("year"),
                "question_count": 0,
                "top_topics": Counter(),
            },
        )
        paper["_indexed_question_count"] = int(paper.get("_indexed_question_count") or 0) + 1
        if item.get("inferred_topic"):
            paper["top_topics"][item["inferred_topic"]] += 1

    subjects = []
    for subject in sorted(grouped.values(), key=lambda item: item["name"].lower()):
        papers = []
        for paper in subject["papers"].values():
            topic_counter = paper.pop("top_topics")
            indexed_count = int(paper.pop("_indexed_question_count", 0) or 0)
            if indexed_count and not int(paper.get("question_count") or 0):
                paper["question_count"] = indexed_count
            paper["top_topics"] = [topic for topic, _ in topic_counter.most_common(5)]
            papers.append(paper)
        subject["question_count"] = sum(int(paper.get("question_count") or 0) for paper in papers)
        subject["papers"] = sorted(
            papers,
            key=lambda item: (
                -(int(item["year"]) if item.get("year") else 0),
                item["name"].lower(),
            ),
        )
        subjects.append(subject)
    return subjects


def _manifest_pyq_paths_for_subject(subject: str) -> list[Path]:
    subject_key = re.sub(r"[^a-z0-9]+", " ", subject.lower()).strip()
    paths = []
    for item in _load_manifest_pyqs():
        item_subject = re.sub(r"[^a-z0-9]+", " ", str(item.get("subject", "")).lower()).strip()
        if item_subject != subject_key:
            continue
        relative_path = item.get("stored_relative_path")
        if not relative_path:
            continue
        pdf_path = DATA_PATH / relative_path
        if pdf_path.exists() and pdf_path.suffix.lower() == ".pdf":
            paths.append(pdf_path)
    return paths


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 160) -> list[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []

    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _query_pyq_pdf_chunks(subject: str, query: str = "", k: int = 8) -> str:
    query_terms = Counter(_tokenize(f"{subject} {query} previous question paper exam"))
    scored = []
    for pdf_path in _manifest_pyq_paths_for_subject(subject):
        try:
            chunks = _chunk_text(_extract_pdf_text(pdf_path))
        except Exception:
            continue
        for chunk in chunks:
            counter = Counter(_tokenize(chunk))
            score = sum(min(counter[token], count) for token, count in query_terms.items())
            if score > 0:
                scored.append((score, pdf_path, chunk))

    if not scored:
        return f"No PYQs were found for {subject}."

    scored.sort(key=lambda item: item[0], reverse=True)
    return "\n\n".join(
        f"Source: PYQ\nSubject: {subject}\nPaper: {pdf_path.stem}\nContent: {chunk}"
        for _, pdf_path, chunk in scored[:k]
    )


def query_pyq_context(subject: str, query: str = "", k: int = 12) -> str:
    query_terms = Counter(_tokenize(f"{subject} {query}"))
    subject_key = re.sub(r"[^a-z0-9]+", " ", subject.lower()).strip()
    records = [
        item for item in _load_index().get("questions", [])
        if item.get("type") == "pyq"
        and re.sub(r"[^a-z0-9]+", " ", str(item.get("subject", "")).lower()).strip() == subject_key
    ]
    if not records:
        return _query_pyq_pdf_chunks(subject, query=query, k=k)

    scored = []
    for item in records:
        text = " ".join(
            [
                item.get("question", ""),
                item.get("inferred_topic", ""),
                " ".join(item.get("keywords", [])),
            ]
        )
        counter = Counter(_tokenize(text))
        score = sum(min(counter[token], count) for token, count in query_terms.items())
        score += int(item.get("frequency_score") or 1) * 3
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    selected = [item for _, item in scored[:k]]
    return "\n\n".join(
        "Source: PYQ"
        f"\nSubject: {item.get('subject')}"
        f"\nTopic: {item.get('inferred_topic')}"
        f"\nKeywords: {', '.join(item.get('keywords', []))}"
        f"\nFrequency score: {item.get('frequency_score', 1)}"
        f"\nQuestion: {item.get('question')}"
        for item in selected
    )
