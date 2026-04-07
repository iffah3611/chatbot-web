import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "chroma_db" / "local_rag_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_STATUS_FILE = CACHE_DIR / "train_status.json"
MANIFEST_PATH = DATA_PATH / "pdf_manifest.json"
STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "ask",
    "answer",
    "can",
    "could",
    "does",
    "doing",
    "explain",
    "for",
    "from",
    "give",
    "have",
    "help",
    "how",
    "into",
    "is",
    "its",
    "mark",
    "marks",
    "me",
    "module",
    "please",
    "question",
    "tell",
    "the",
    "this",
    "today",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
    "your",
}
SUBJECT_KEYWORDS = {
    "aad": {"aad", "algorithm", "analysis", "design", "cst306"},
    "algorithm analysis and design": {"aad", "algorithm", "analysis", "design", "cst306"},
    "compiler design": {"cd", "compiler", "design", "cst302"},
    "cd": {"cd", "compiler", "design", "cst302"},
    "computer graphics and image processing": {"cgip", "computer", "graphics", "image", "processing", "cst304"},
    "computer graphics": {"cgip", "computer", "graphics", "image", "processing", "cst304"},
    "industrial economics and foreign trade": {"ieft", "industrial", "economics", "foreign", "trade", "hut300"},
    "industrial economics": {"ieft", "industrial", "economics", "foreign", "trade", "hut300"},
    "programming in python": {"programming", "python", "cst362"},
    "python": {"programming", "python", "cst362"},
}
SUBJECT_REQUIRED_KEYWORDS = {
    "aad": {"aad", "cst306", "algorithm"},
    "algorithm analysis and design": {"aad", "cst306", "algorithm"},
    "compiler design": {"cd", "cst302", "compiler"},
    "cd": {"cd", "cst302", "compiler"},
    "computer graphics and image processing": {"cgip", "cst304", "graphics"},
    "computer graphics": {"cgip", "cst304", "graphics"},
    "industrial economics and foreign trade": {"ieft", "hut300", "industrial", "economics"},
    "industrial economics": {"ieft", "hut300", "industrial", "economics"},
    "programming in python": {"python", "cst362"},
    "python": {"python", "cst362"},
}
ROMAN_MODULES = {
    1: "i",
    2: "ii",
    3: "iii",
    4: "iv",
    5: "v",
}
EXAM_RELEVANCE_TERMS = {
    "algorithm",
    "analysis",
    "application",
    "complexity",
    "definition",
    "difference",
    "formula",
    "important",
    "module",
    "problem",
    "properties",
    "theorem",
    "time",
}


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z]{3,}|[0-9]+", text.lower())
        if token not in STOPWORDS
    ]


def _normalized_words(text: str) -> set[str]:
    return set(_tokenize(text.replace("_", " ").replace("-", " ")))


def _subject_terms(subject: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", subject.lower()).strip()
    compact = normalized.replace(" ", "")
    return SUBJECT_KEYWORDS.get(normalized) or SUBJECT_KEYWORDS.get(compact) or _normalized_words(subject)


def _subject_required_terms(subject: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", subject.lower()).strip()
    compact = normalized.replace(" ", "")
    return SUBJECT_REQUIRED_KEYWORDS.get(normalized) or SUBJECT_REQUIRED_KEYWORDS.get(compact) or _subject_terms(subject)


def _module_markers(module_number: int) -> set[str]:
    roman = ROMAN_MODULES.get(module_number, str(module_number))
    return {
        f"module {module_number}",
        f"module-{module_number}",
        f"module{module_number}",
        f"module {roman}",
        f"module-{roman}",
        f"module{roman}",
        f"mod {module_number}",
        f"mod {roman}",
    }


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _has_module_marker(text: str, module_number: int) -> bool:
    compact = _compact_text(text)
    return any(marker in compact for marker in _module_markers(module_number))


def _trim_to_module_section(text: str, module_number: int) -> str:
    roman = ROMAN_MODULES.get(module_number, str(module_number))
    match = re.search(
        rf"module\s*[-–—]?\s*({module_number}|{roman})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return text

    start = max(0, match.start() - 80)
    trimmed = text[start:start + 1400]
    return trimmed.strip() or text


def _matches_subject(pdf_path: Path, subject: str) -> bool:
    path_text = _compact_text(_relative_pdf_path(pdf_path))
    path_words = _normalized_words(path_text)
    required_terms = _subject_required_terms(subject)
    terms = _subject_terms(subject)
    return bool(required_terms & path_words) or all(term in path_words for term in terms if len(term) > 3)


def _module_context_query(subject: str, module_number: int) -> Counter:
    return Counter(
        _tokenize(
            f"{subject} module {module_number} important definition formula application "
            "algorithm complexity properties difference problem theorem"
        )
    )


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    cleaned = " ".join(text.split())
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


def _all_pdf_files(doc_type: str | None = None) -> list[Path]:
    if not DATA_PATH.exists():
        return []
    pdf_files = sorted(DATA_PATH.rglob("*.pdf"), key=lambda item: str(item).lower())
    if doc_type is None:
        return pdf_files
    return [pdf_path for pdf_path in pdf_files if _pdf_type(pdf_path) == doc_type]


def _cache_path_for(pdf_path: Path) -> Path:
    relative = _relative_pdf_path(pdf_path)
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _relative_pdf_path(pdf_path: Path) -> str:
    return str(pdf_path.relative_to(DATA_PATH)).replace("\\", "/")


def _data_url_for(pdf_path: Path) -> str:
    relative_path = _relative_pdf_path(pdf_path)
    return "/data/" + "/".join(quote(segment) for segment in relative_path.split("/"))


def _source_link_for(pdf_path: Path) -> str:
    title = pdf_path.stem.replace("[", "(").replace("]", ")")
    return f"[{title}]({_data_url_for(pdf_path)})"


def _load_pdf_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    files = payload.get("files", [])
    return files if isinstance(files, list) else []


def _manifest_item_for_path(pdf_path: Path) -> dict | None:
    relative_path = _relative_pdf_path(pdf_path)
    for item in _load_pdf_manifest():
        if item.get("stored_relative_path") == relative_path:
            return item
    return None


def _pdf_type(pdf_path: Path) -> str:
    item = _manifest_item_for_path(pdf_path)
    if item and item.get("type") in {"notes", "pyq"}:
        return item["type"]
    relative = _relative_pdf_path(pdf_path).lower().replace("\\", "/")
    if "/pyq/" in relative or relative.startswith("pyq/"):
        return "pyq"
    return "notes"


def _manifest_records_for(subject: str, module_number: int) -> list[tuple[Path, dict]]:
    subject_key = _compact_text(subject)
    records = []
    for item in _load_pdf_manifest():
        if item.get("type", "notes") != "notes":
            continue
        if int(item.get("module_number") or 0) != module_number:
            continue
        if _compact_text(item.get("subject", "")) != subject_key:
            continue
        relative_path = item.get("stored_relative_path")
        if not relative_path:
            continue
        pdf_path = DATA_PATH / relative_path
        if pdf_path.exists() and pdf_path.suffix.lower() == ".pdf":
            records.append((pdf_path, item))
    return records


def _manifest_records_for_type(subject: str, doc_type: str) -> list[tuple[Path, dict]]:
    subject_key = _compact_text(subject)
    records = []
    for item in _load_pdf_manifest():
        if item.get("type", "notes") != doc_type:
            continue
        if _compact_text(item.get("subject", "")) != subject_key:
            continue
        relative_path = item.get("stored_relative_path")
        if not relative_path:
            continue
        pdf_path = DATA_PATH / relative_path
        if pdf_path.exists() and pdf_path.suffix.lower() == ".pdf":
            records.append((pdf_path, item))
    return records


def _load_cached_chunks(pdf_path: Path) -> list[str] | None:
    cache_path = _cache_path_for(pdf_path)
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if payload.get("mtime") != pdf_path.stat().st_mtime:
        return None

    return payload.get("chunks", [])


def _save_cached_chunks(pdf_path: Path, chunks: list[str]) -> None:
    cache_path = _cache_path_for(pdf_path)
    payload = {
        "mtime": pdf_path.stat().st_mtime,
        "chunks": chunks,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages[:20]:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _get_pdf_chunks(pdf_path: Path) -> list[str]:
    cached = _load_cached_chunks(pdf_path)
    if cached is not None:
        return cached

    try:
        text = _extract_pdf_text(pdf_path)
    except Exception as exc:
        print(f"Failed to read PDF for local RAG: {pdf_path} ({exc})")
        return []

    chunks = _chunk_text(text)
    _save_cached_chunks(pdf_path, chunks)
    return chunks


def _select_candidate_files(question: str, max_files: int = 12, doc_type: str = "notes") -> list[Path]:
    pdf_files = _all_pdf_files(doc_type=doc_type)
    if not pdf_files:
        return []

    query_counter = Counter(_tokenize(question))
    path_query_counter = Counter(
        {token: count for token, count in query_counter.items() if not token.isdigit()}
    )
    if not path_query_counter:
        path_query_counter = query_counter
    scored = []

    for pdf_path in pdf_files:
        relative_path = _relative_pdf_path(pdf_path)
        path_counter = Counter(_tokenize(relative_path))
        score = sum(min(path_counter[token], count) for token, count in path_query_counter.items())
        scored.append((score, pdf_path))

    scored.sort(key=lambda item: (item[0], str(item[1]).lower()), reverse=True)
    selected = [pdf_path for score, pdf_path in scored if score > 0][:max_files]
    if selected:
        return selected
    return pdf_files


def ingest_pdf(file_path: str):
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        return
    _get_pdf_chunks(pdf_path)


def clear_rag_cache() -> int:
    removed = 0
    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            cache_file.unlink()
            removed += 1
        except Exception:
            continue
    return removed


def reindex_all_pdfs(force: bool = True) -> dict:
    pdf_files = _all_pdf_files()
    if force:
        clear_rag_cache()

    indexed = 0
    failed = 0

    for pdf_path in pdf_files:
        try:
            chunks = _get_pdf_chunks(pdf_path)
            if chunks:
                indexed += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    result = {
        "total_files": len(pdf_files),
        "indexed_files": indexed,
        "failed_files": failed,
        "cache_dir": str(CACHE_DIR),
    }
    payload = {
        "last_trained_at": datetime.now(timezone.utc).isoformat(),
        "stats": result,
    }
    TRAIN_STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return result


def get_training_status() -> dict:
    total_files = len(_all_pdf_files())
    cached_files = len([item for item in CACHE_DIR.glob("*.json") if item.name != TRAIN_STATUS_FILE.name])

    status_payload = {}
    if TRAIN_STATUS_FILE.exists():
        try:
            status_payload = json.loads(TRAIN_STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            status_payload = {}

    return {
        "trained": bool(status_payload),
        "last_trained_at": status_payload.get("last_trained_at"),
        "last_stats": status_payload.get("stats", {}),
        "current_total_files": total_files,
        "current_cached_files": cached_files,
    }


def query_rag(question: str, k: int = 4, doc_type: str = "notes"):
    if os.getenv("ENABLE_RAG", "true").lower() != "true":
        return "Document search is disabled in the current environment."

    candidate_files = _select_candidate_files(question, doc_type=doc_type)
    if not candidate_files:
        return f"No {doc_type} documents have been uploaded yet."

    query_counter = Counter(_tokenize(question))
    if not query_counter:
        return "Ask a more specific question so I can search your uploaded PDFs."

    scored_chunks = []
    for pdf_path in candidate_files:
        for chunk in _get_pdf_chunks(pdf_path):
            chunk_counter = Counter(_tokenize(chunk))
            score = sum(min(chunk_counter[token], count) for token, count in query_counter.items())
            if score > 0:
                scored_chunks.append((score, pdf_path, chunk))

    if not scored_chunks:
        file_list = "\n".join(f"- {_source_link_for(pdf_path)}" for pdf_path in candidate_files)
        return (
            "I found related PDF files but could not extract matching text yet. "
            "You can open them here:\n"
            f"{file_list}"
        )

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    top_chunks = scored_chunks[:k]
    return "\n\n".join(
        f"Source: {_source_link_for(pdf_path)}\n{chunk}"
        for _, pdf_path, chunk in top_chunks
    )


def query_module_context(subject: str, module_number: int, k: int = 10) -> str:
    if os.getenv("ENABLE_RAG", "true").lower() != "true":
        return "Document search is disabled in the current environment."

    manifest_records = _manifest_records_for(subject, module_number)
    pdf_files = [pdf_path for pdf_path, _ in manifest_records]
    if not pdf_files:
        pdf_files = [pdf_path for pdf_path in _all_pdf_files(doc_type="notes") if _matches_subject(pdf_path, subject)]
    if not pdf_files:
        return f"No study material was found for {subject}."

    module_files = pdf_files if manifest_records else [
            pdf_path
            for pdf_path in pdf_files
            if _has_module_marker(_relative_pdf_path(pdf_path), module_number)
        ]
    candidate_files = module_files or pdf_files
    query_counter = _module_context_query(subject, module_number)
    scored_chunks = []

    for pdf_path in candidate_files:
        path_has_module = _has_module_marker(_relative_pdf_path(pdf_path), module_number)
        for chunk in _get_pdf_chunks(pdf_path):
            chunk_has_module = _has_module_marker(chunk, module_number)
            if module_files and not path_has_module:
                continue
            if not module_files and not path_has_module and not chunk_has_module:
                continue

            chunk = chunk if path_has_module else _trim_to_module_section(chunk, module_number)
            chunk_counter = Counter(_tokenize(chunk))
            score = sum(min(chunk_counter[token], count) for token, count in query_counter.items())
            score += sum(chunk_counter[term] for term in EXAM_RELEVANCE_TERMS)
            if path_has_module:
                score += 5
            if chunk_has_module:
                score += 8
            if score > 0:
                scored_chunks.append((score, pdf_path, chunk))

    if not scored_chunks and not module_files:
        for pdf_path in candidate_files:
            for chunk in _get_pdf_chunks(pdf_path):
                chunk_counter = Counter(_tokenize(chunk))
                score = sum(min(chunk_counter[token], count) for token, count in query_counter.items())
                if score > 0:
                    scored_chunks.append((score, pdf_path, chunk))

    if not scored_chunks:
        return f"No module {module_number} material was found for {subject}."

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    top_chunks = scored_chunks[:k]
    return "\n\n".join(
        f"Source: {_source_link_for(pdf_path)}\n{chunk}"
        for _, pdf_path, chunk in top_chunks
    )


def query_subject_notes_context(subject: str, query: str = "", k: int = 8) -> str:
    if os.getenv("ENABLE_RAG", "true").lower() != "true":
        return "Document search is disabled in the current environment."

    manifest_records = _manifest_records_for_type(subject, "notes")
    pdf_files = [pdf_path for pdf_path, _ in manifest_records]
    if not pdf_files:
        pdf_files = [pdf_path for pdf_path in _all_pdf_files(doc_type="notes") if _matches_subject(pdf_path, subject)]
    if not pdf_files:
        return f"No notes were found for {subject}."

    query_counter = Counter(_tokenize(f"{subject} {query} important definition formula application"))
    scored_chunks = []
    for pdf_path in pdf_files:
        for chunk in _get_pdf_chunks(pdf_path):
            chunk_counter = Counter(_tokenize(chunk))
            score = sum(min(chunk_counter[token], count) for token, count in query_counter.items())
            score += sum(chunk_counter[term] for term in EXAM_RELEVANCE_TERMS)
            if score > 0:
                scored_chunks.append((score, pdf_path, chunk))

    if not scored_chunks:
        return f"No relevant notes were found for {subject}."

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return "\n\n".join(
        f"Source: {_source_link_for(pdf_path)}\n{chunk}"
        for _, pdf_path, chunk in scored_chunks[:k]
    )


def query_syllabus_context(subject: str, module_number: int, k: int = 4) -> str:
    if os.getenv("ENABLE_RAG", "true").lower() != "true":
        return "Document search is disabled in the current environment."

    syllabus_files = [
        pdf_path for pdf_path in _all_pdf_files()
        if "/syllabus/" in _relative_pdf_path(pdf_path).lower()
        or _relative_pdf_path(pdf_path).lower().startswith("dataset/syllabus/")
    ]
    candidate_files = [pdf_path for pdf_path in syllabus_files if _matches_subject(pdf_path, subject)]
    if not candidate_files:
        candidate_files = syllabus_files
    if not candidate_files:
        return f"No syllabus document was found for {subject}."

    query_counter = Counter(_tokenize(f"{subject} module {module_number} syllabus topics outcomes concepts"))
    scored_chunks = []
    for pdf_path in candidate_files:
        for chunk in _get_pdf_chunks(pdf_path):
            chunk_counter = Counter(_tokenize(chunk))
            score = sum(min(chunk_counter[token], count) for token, count in query_counter.items())
            if _has_module_marker(chunk, module_number):
                score += 10
            if score > 0:
                scored_chunks.append((score, pdf_path, chunk))

    if not scored_chunks:
        return f"No relevant syllabus topics were found for {subject} Module {module_number}."

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return "\n\n".join(
        f"Source: {_source_link_for(pdf_path)}\n{chunk}"
        for _, pdf_path, chunk in scored_chunks[:k]
    )
