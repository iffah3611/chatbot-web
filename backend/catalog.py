import re
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data"
DATASET_PATH = DATA_PATH / "dataset"
LEGACY_NOTES_PATH = DATA_PATH / "notes"
MANIFEST_PATH = DATA_PATH / "pdf_manifest.json"
MODULE_PATH_PATTERN = re.compile(r"\bmodule\s*([1-5])\b", re.IGNORECASE)

CANONICAL_SUBJECT_NAMES = {
    "aad": "Algorithm Analysis and Design",
    "algorithmanalysis": "Algorithm Analysis and Design",
    "algorithmanalysisanddesign": "Algorithm Analysis and Design",
    "cd": "Compiler Design",
    "compilerdesign": "Compiler Design",
    "cgip": "Computer Graphics and Image Processing",
    "computergraphics": "Computer Graphics and Image Processing",
    "computergraphicsandimageprocessing": "Computer Graphics and Image Processing",
    "ieft": "Industrial Economics and Foreign Trade",
    "industrialeconomics": "Industrial Economics and Foreign Trade",
    "industrialeconomicsandforeigntrade": "Industrial Economics and Foreign Trade",
    "programminginpython": "Programming in Python",
}


def _normalize_label(value: str) -> str:
    cleaned = re.sub(r"[_\-]+", " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _canonical_subject_name(value: str) -> str:
    normalized = _normalize_label(value)
    token = re.sub(r"[^a-z0-9]+", "", normalized.lower())

    if token in CANONICAL_SUBJECT_NAMES:
        return CANONICAL_SUBJECT_NAMES[token]

    return normalized.title()


def _subject_name_from_folder(folder_name: str) -> str:
    if "(" in folder_name:
        return _canonical_subject_name(folder_name.split("(")[0])
    return _canonical_subject_name(folder_name)


def _subject_code_from_folder(folder_name: str) -> str | None:
    match = re.search(r"\(([A-Z]{2,}\d{3})\)", folder_name)
    if match:
        return match.group(1)
    return None


def _relative_pdf(path: Path) -> str:
    return str(path.relative_to(DATA_PATH)).replace("\\", "/")


def _display_title_from_pdf(path: Path) -> str:
    return _normalize_label(path.stem)


def _load_manifest_by_path() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    records = payload.get("files", [])
    if not isinstance(records, list):
        return {}
    return {
        item["stored_relative_path"]: item
        for item in records
        if isinstance(item, dict) and item.get("stored_relative_path")
    }


def _module_number_for_pdf(path: Path, manifest_by_path: dict[str, dict]) -> int | None:
    relative = _relative_pdf(path)
    manifest_item = manifest_by_path.get(relative)
    if manifest_item and manifest_item.get("module_number"):
        return int(manifest_item["module_number"])

    match = MODULE_PATH_PATTERN.search(relative.replace("\\", "/"))
    if match:
        return int(match.group(1))
    return None


def _type_for_pdf(path: Path, manifest_by_path: dict[str, dict]) -> str:
    relative = _relative_pdf(path)
    manifest_item = manifest_by_path.get(relative)
    if manifest_item and manifest_item.get("type") in {"notes", "pyq"}:
        return manifest_item["type"]
    lowered = relative.lower().replace("\\", "/")
    if "/pyq/" in lowered:
        return "pyq"
    return "notes"


def _syllabus_subject_name(path: Path) -> str:
    stem = _normalize_label(path.stem)
    stem = re.sub(r"\s*\(\d+\)$", "", stem).strip()
    stem = re.sub(r"(?i)\bktunotes(\.in)?\b", "", stem)
    stem = _normalize_label(stem)
    return _canonical_subject_name(stem)


def get_subject_catalog() -> list[dict]:
    subjects: list[dict] = []
    manifest_by_path = _load_manifest_by_path()

    if DATASET_PATH.exists():
        for entry in sorted(DATASET_PATH.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir() or entry.name.lower() == "syllabus":
                continue

            pdfs = [
                pdf for pdf in sorted(entry.rglob("*.pdf"), key=lambda item: item.name.lower())
                if _type_for_pdf(pdf, manifest_by_path) == "notes"
            ]
            if not pdfs:
                continue

            files = [
                {
                    "name": _display_title_from_pdf(pdf),
                    "path": _relative_pdf(pdf),
                    "module_number": _module_number_for_pdf(pdf, manifest_by_path),
                    "type": "notes",
                }
                for pdf in pdfs
            ]
            modules = sorted({item["module_number"] for item in files if item.get("module_number")})

            subjects.append(
                {
                    "name": _subject_name_from_folder(entry.name),
                    "code": _subject_code_from_folder(entry.name),
                    "folder": entry.name,
                    "modules": modules,
                    "files": files,
                }
            )

    legacy_pdfs = []
    if LEGACY_NOTES_PATH.exists():
        for pdf in sorted(LEGACY_NOTES_PATH.glob("*.pdf"), key=lambda item: item.name.lower()):
            legacy_pdfs.append(
                {
                    "name": _display_title_from_pdf(pdf),
                    "path": _relative_pdf(pdf),
                    "type": "notes",
                }
            )

    if legacy_pdfs:
        ieft_subject = next((subject for subject in subjects if subject.get("name") == "Industrial Economics and Foreign Trade"), None)
        if ieft_subject:
            existing_paths = {item["path"] for item in ieft_subject.get("files", [])}
            ieft_subject["files"].extend([item for item in legacy_pdfs if item["path"] not in existing_paths])
        else:
            subjects.append(
                {
                    "name": "Industrial Economics and Foreign Trade",
                    "code": "HUT300",
                    "folder": "IEFT",
                    "files": legacy_pdfs,
                }
            )

    return subjects


def get_syllabus_catalog() -> list[dict]:
    syllabus_docs: list[dict] = []
    seen_subject_names: set[str] = set()
    dataset_syllabus = DATASET_PATH / "Syllabus"

    if dataset_syllabus.exists():
        for pdf in sorted(dataset_syllabus.glob("*.pdf"), key=lambda item: item.name.lower()):
            subject_name = _syllabus_subject_name(pdf)
            subject_key = subject_name.lower()
            if subject_key in seen_subject_names:
                continue
            seen_subject_names.add(subject_key)
            syllabus_docs.append(
                {
                    "name": subject_name,
                    "path": _relative_pdf(pdf),
                }
            )

    legacy_syllabus = LEGACY_NOTES_PATH / "syllabus"
    if legacy_syllabus.exists():
        for pdf in sorted(legacy_syllabus.glob("*.pdf"), key=lambda item: item.name.lower()):
            relative_path = _relative_pdf(pdf)
            subject_name = _syllabus_subject_name(pdf)
            subject_key = subject_name.lower()
            if any(doc["path"] == relative_path for doc in syllabus_docs):
                continue
            if subject_key in seen_subject_names:
                continue
            seen_subject_names.add(subject_key)
            syllabus_docs.append(
                {
                    "name": subject_name,
                    "path": relative_path,
                }
            )

    return syllabus_docs
