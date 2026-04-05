from pathlib import Path, PurePosixPath

from fastapi import UploadFile

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


async def handle_uploads(files: list[UploadFile], relative_paths: list[str] | None = None):
    if not files:
        raise ValueError("At least one PDF file is required.")

    if relative_paths and len(relative_paths) != len(files):
        raise ValueError("Upload paths do not match the selected files.")

    from .rag_engine import ingest_pdf

    uploaded_files: list[str] = []

    for index, file in enumerate(files):
        if not file.filename:
            raise ValueError("Each uploaded file must have a name.")

        relative_path = relative_paths[index] if relative_paths else file.filename
        safe_relative_path = sanitize_relative_path(
            relative_path, file.filename)
        file_path = DATA_PATH / safe_relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        content = await file.read()

        with open(file_path, "wb") as buffer:
            buffer.write(content)

        try:
            ingest_pdf(str(file_path))
        except Exception as e:
            print(f"Warning: Could not index PDF {file_path}: {e}")

        uploaded_files.append(str(safe_relative_path).replace("\\", "/"))

    return {
        "message": f"Uploaded {len(uploaded_files)} PDF file(s). Indexing may be limited due to compatibility issues.",
        "files": uploaded_files,
    }
