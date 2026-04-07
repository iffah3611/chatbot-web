import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from .catalog import _canonical_subject_name

DATA_PATH = Path(__file__).resolve().parents[1] / "data"
DATASET_PATH = DATA_PATH / "dataset"
MANIFEST_PATH = DATA_PATH / "pdf_manifest.json"
REVIEW_LOG_PATH = DATA_PATH / "pdf_review_log.jsonl"

MODULE_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:module|mod|m)[\s_\-]*([1-5]|i{1,3}|iv|v)(?![a-z0-9])",
    re.IGNORECASE,
)
ROMAN_TO_MODULE = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
}
SUBJECT_ALIASES = {
    "aad": "Algorithm Analysis and Design",
    "algorithm analysis": "Algorithm Analysis and Design",
    "algorithm analysis design": "Algorithm Analysis and Design",
    "algorithm analysis and design": "Algorithm Analysis and Design",
    "cd": "Compiler Design",
    "compiler design": "Compiler Design",
    "cgip": "Computer Graphics and Image Processing",
    "computer graphics": "Computer Graphics and Image Processing",
    "computer graphics image processing": "Computer Graphics and Image Processing",
    "computer graphics and image processing": "Computer Graphics and Image Processing",
    "ieft": "Industrial Economics and Foreign Trade",
    "industrial economics": "Industrial Economics and Foreign Trade",
    "industrial economics foreign trade": "Industrial Economics and Foreign Trade",
    "industrial economics and foreign trade": "Industrial Economics and Foreign Trade",
    "python": "Programming in Python",
    "programming in python": "Programming in Python",
}
SUBJECT_CODE_ALIASES = {
    "cst306": "Algorithm Analysis and Design",
    "cst302": "Compiler Design",
    "cst304": "Computer Graphics and Image Processing",
    "hut300": "Industrial Economics and Foreign Trade",
    "cst362": "Programming in Python",
}
SUBJECT_CODE_BY_NAME = {subject: code.upper() for code, subject in SUBJECT_CODE_ALIASES.items()}
FILENAME_NOISE = re.compile(
    r"\b(module|mod|m|notes?|ppt|slides?|pdf|ktunotes|kerala|lectures?|complete|material|scheme|s\d+|20\d{2})\b",
    re.IGNORECASE,
)
QUESTION_PAPER_TERMS = re.compile(
    r"\b(previous\s+year|pyq|question\s+paper|answer\s+any|part\s+[abc]|"
    r"duration|maximum\s+marks|semester\s+examination|reg\s*no|questions?)\b",
    re.IGNORECASE,
)
QUESTION_START_PATTERN = re.compile(
    r"(?im)^\s*(?:q(?:uestion)?\.?\s*)?\d{1,2}\s*[\).:-]\s+\S"
)
SUBJECT_CODE_PATTERN = re.compile(r"\b([a-z]{2,}\d{3})\b", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(?<!\d)(20[0-3]\d)(?!\d)")


def extract_module_number(value: str) -> tuple[int | None, str | None]:
    matches = MODULE_PATTERN.findall(value or "")
    normalized = []
    for match in matches:
        token = match.lower()
        number = ROMAN_TO_MODULE.get(token) if token.isalpha() else int(token)
        if number not in normalized:
            normalized.append(number)

    if len(normalized) == 1:
        return normalized[0], None
    if len(normalized) > 1:
        return None, f"Conflicting module tags found: {', '.join(f'Module {item}' for item in normalized)}"
    return None, "No module tag found in filename."


def _extract_module_from_pdf_content(file_path: Path) -> int | None:
    try:
        reader = PdfReader(str(file_path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:2])
    except Exception:
        return None

    module_number, error = extract_module_number(text[:2500])
    if error:
        return None
    return module_number


def _extract_pdf_preview(file_path: Path, max_pages: int = 6) -> str:
    try:
        reader = PdfReader(str(file_path))
        return "\n".join((page.extract_text() or "") for page in reader.pages[:max_pages])
    except Exception:
        return ""


def classify_pdf_type(filename: str, relative_path: str = "", temp_file_path: Path | None = None) -> str | None:
    searchable_name = f"{relative_path} {filename}".lower()
    preview = _extract_pdf_preview(temp_file_path) if temp_file_path else ""
    text = f"{searchable_name}\n{preview}"
    compact = re.sub(r"\s+", " ", text.lower())

    filename_pyq_hint = bool(re.search(r"\b(pyq|previous|question\s*paper|qp|exam\s*paper)\b", searchable_name))
    filename_notes_hint = bool(re.search(r"\b(notes?|module|mod|slides?|ppt|textbook|material)\b", searchable_name))
    question_mark_count = compact.count("?")
    numbered_question_count = len(QUESTION_START_PATTERN.findall(preview))
    exam_term_count = len(QUESTION_PAPER_TERMS.findall(text))

    if filename_pyq_hint or exam_term_count >= 3 or numbered_question_count >= 5 or question_mark_count >= 8:
        return "pyq"
    if filename_notes_hint or extract_module_number(f"{relative_path} {filename}")[0] is not None:
        return "notes"
    if preview and len(preview.split()) > 80 and question_mark_count <= 2 and numbered_question_count <= 1:
        return "notes"
    return None


def detect_subject_code(filename: str, relative_path: str = "", subject: str | None = None) -> str | None:
    searchable = f"{relative_path} {filename}"
    code_match = SUBJECT_CODE_PATTERN.search(searchable)
    if code_match:
        return code_match.group(1).upper()
    if subject:
        return SUBJECT_CODE_BY_NAME.get(subject)
    return None


def extract_exam_year(filename: str, relative_path: str = "") -> int | None:
    years = [int(match) for match in YEAR_PATTERN.findall(f"{relative_path} {filename}")]
    if not years:
        return None
    return max(years)


def detect_subject_name(filename: str, relative_path: str = "", fallback_subject: str | None = None) -> str | None:
    searchable = f"{relative_path} {filename}".lower()
    code_match = re.search(r"\b([a-z]{2,}\d{3})\b", searchable, flags=re.IGNORECASE)
    if code_match:
        subject = SUBJECT_CODE_ALIASES.get(code_match.group(1).lower())
        if subject:
            return subject

    normalized = re.sub(r"[^a-z0-9]+", " ", searchable).strip()
    for alias, subject in sorted(SUBJECT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", normalized):
            return subject

    if fallback_subject:
        return _canonical_subject_name(fallback_subject)

    parts = Path(relative_path.replace("\\", "/")).parts
    for part in parts:
        if part and part.lower() not in {"dataset", "data", "notes", "syllabus"} and not part.lower().endswith(".pdf"):
            return _canonical_subject_name(part.split("(")[0])

    stem = Path(filename).stem
    cleaned = MODULE_PATTERN.sub(" ", stem)
    cleaned = FILENAME_NOISE.sub(" ", cleaned)
    cleaned = re.sub(r"[_\-\s]+", " ", cleaned).strip()
    if cleaned:
        return _canonical_subject_name(cleaned)

    return None


def _safe_segment(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', " ", value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "Untitled"


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    index = 2
    while True:
        candidate = path.with_name(f"{path.stem} ({index}){path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"files": []}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"files": []}


def _save_manifest(manifest: dict) -> None:
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")


def _append_review_log(payload: dict) -> None:
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    if not REVIEW_LOG_PATH.exists():
        REVIEW_LOG_PATH.write_text("", encoding="utf-8")
    with REVIEW_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def build_grouped_pdf_path(
    filename: str,
    relative_path: str = "",
    fallback_subject: str | None = None,
    temp_file_path: Path | None = None,
) -> tuple[Path, dict]:
    file_type = classify_pdf_type(filename, relative_path=relative_path, temp_file_path=temp_file_path)
    module_number, module_error = extract_module_number(f"{relative_path} {filename}")
    source = "filename"
    if module_number is None and temp_file_path is not None:
        module_number = _extract_module_from_pdf_content(temp_file_path)
        source = "pdf_content" if module_number is not None else "unmatched"

    subject = detect_subject_name(filename, relative_path=relative_path, fallback_subject=fallback_subject)
    subject_code = detect_subject_code(filename, relative_path=relative_path, subject=subject)
    exam_year = extract_exam_year(filename, relative_path=relative_path)
    issues = []
    if file_type is None:
        issues.append("Unable to classify file. Please re-upload or label manually.")
    if file_type == "notes" and module_number is None:
        issues.append(module_error or "Unable to detect module number.")
    if not subject:
        issues.append("Unable to detect subject. Provide a subject override.")
    if issues:
        _append_review_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "filename": filename,
                "relative_path": relative_path,
                "issues": issues,
            }
        )
        raise ValueError(f"{filename}: {' '.join(issues)}")

    subject_segment = _safe_segment(subject)
    if file_type == "pyq":
        destination = DATASET_PATH / subject_segment / "pyq" / _safe_segment(filename)
    else:
        module_segment = f"module{module_number}"
        destination = DATASET_PATH / subject_segment / "notes" / module_segment / _safe_segment(filename)
    destination = _unique_destination(destination)
    metadata = {
        "subject": subject,
        "subject_code": subject_code,
        "year": exam_year,
        "type": file_type,
        "module_number": module_number,
        "module": module_number,
        "source_filename": filename,
        "filename": filename,
        "original_relative_path": relative_path,
        "module_detection_source": source,
        "stored_relative_path": str(destination.relative_to(DATA_PATH)).replace("\\", "/"),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    return destination, metadata


def record_pdf_metadata(metadata: dict) -> None:
    manifest = _load_manifest()
    files = manifest.setdefault("files", [])
    files = [item for item in files if item.get("stored_relative_path") != metadata.get("stored_relative_path")]
    files.append(metadata)
    manifest["files"] = sorted(files, key=lambda item: item.get("stored_relative_path", "").lower())
    _save_manifest(manifest)
