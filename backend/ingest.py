from pathlib import Path, PurePosixPath
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.orm import Session

DATA_PATH = Path(__file__).resolve().parents[1] / "data"
DATA_PATH.mkdir(parents=True, exist_ok=True)


def sanitize_relative_path(relative_path: str, fallback_filename: str) -> Path:
    raw_path = (relative_path or fallback_filename or "").strip().replace(
        "\\", "/")
    if not raw_path:
        raise ValueError("Missing file path.")

    path_parts = []
    for part in PurePosixPath(raw_path).parts:
        if part in ("", ".", ".."):
            continue
        path_parts.append(part)

    if not path_parts:
        raise ValueError("Invalid file path.")

    safe_path = Path(*path_parts)
    if safe_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are allowed.")

    return safe_path


def _reserve_batch_destination(file_path: Path, reserved_paths: set[Path]) -> Path:
    if file_path not in reserved_paths:
        reserved_paths.add(file_path)
        return file_path

    index = 2
    while True:
        candidate = file_path.with_name(f"{file_path.stem} ({index}){file_path.suffix}")
        if candidate not in reserved_paths and not candidate.exists():
            reserved_paths.add(candidate)
            return candidate
        index += 1


async def handle_uploads(
    files: list[UploadFile],
    relative_paths: list[str] | None = None,
    subject_override: str | None = None,
    db: Session | None = None,
):
    if not files:
        raise ValueError("At least one PDF file is required.")

    if relative_paths and len(relative_paths) != len(files):
        raise ValueError("Upload paths do not match the selected files.")

    from .preprocess import build_grouped_pdf_path, record_pdf_metadata
    from .pyq_engine import index_pyq_pdf
    from .rag_engine import ingest_pdf
    from . import models

    planned_uploads: list[tuple[Path, Path, dict]] = []
    reserved_paths: set[Path] = set()
    temp_dir = DATA_PATH / ".incoming"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for index, file in enumerate(files):
            if not file.filename:
                raise ValueError("Each uploaded file must have a name.")

            relative_path = relative_paths[index] if relative_paths else file.filename
            safe_relative_path = sanitize_relative_path(
                relative_path, file.filename)
            content = await file.read()
            temp_file_path = temp_dir / f"{uuid4().hex}.pdf"
            temp_file_path.write_bytes(content)

            file_path, metadata = build_grouped_pdf_path(
                file.filename,
                relative_path=str(safe_relative_path).replace("\\", "/"),
                fallback_subject=subject_override,
                temp_file_path=temp_file_path,
            )
            file_path = _reserve_batch_destination(file_path, reserved_paths)
            metadata["stored_relative_path"] = str(file_path.relative_to(DATA_PATH)).replace("\\", "/")
            planned_uploads.append((temp_file_path, file_path, metadata))
    except Exception:
        for temp_file_path, _, _ in planned_uploads:
            temp_file_path.unlink(missing_ok=True)
        for temp_file_path in temp_dir.glob("*.pdf"):
            temp_file_path.unlink(missing_ok=True)
        raise

    uploaded_files: list[str] = []
    metadata_records: list[dict] = []

    for temp_file_path, file_path, metadata in planned_uploads:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file_path.replace(file_path)

        try:
            if metadata.get("type") == "pyq":
                pyq_result = index_pyq_pdf(file_path, metadata)
                metadata["pyq_question_count"] = pyq_result.get("question_count", 0)
            ingest_pdf(str(file_path))
        except Exception as e:
            print(f"Warning: Could not index PDF {file_path}: {e}")

        record_pdf_metadata(metadata)

        if metadata.get("type") == "pyq" and db is not None:
            existing = (
                db.query(models.PreviousQuestionPaper)
                .filter(models.PreviousQuestionPaper.file_path == metadata["stored_relative_path"])
                .first()
            )
            if existing is None:
                existing = models.PreviousQuestionPaper(file_path=metadata["stored_relative_path"])
                db.add(existing)

            existing.subject = metadata["subject"]
            existing.subject_code = metadata.get("subject_code")
            existing.year = metadata.get("year")
            existing.filename = metadata.get("filename") or metadata.get("source_filename") or file_path.name
            existing.question_count = int(metadata.get("pyq_question_count") or 0)
            existing.uploaded_at = metadata["uploaded_at"]
            db.commit()

        uploaded_files.append(metadata["stored_relative_path"])
        metadata_records.append(metadata)

    return {
        "message": f"Uploaded, organized, and indexed {len(uploaded_files)} PDF file(s).",
        "files": uploaded_files,
        "metadata": metadata_records,
    }
