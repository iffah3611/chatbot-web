from pathlib import Path
import json
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import models, schemas
from .auth import create_access_token, get_current_user, hash_password, verify_password
from .database import Base, engine, get_db
from .ingest import handle_uploads
from .llm import ask_llm
from .catalog import DATA_PATH, get_subject_catalog, get_syllabus_catalog
from .pyq_engine import get_pyq_catalog
from .rag_engine import get_training_status, reindex_all_pdfs

Base.metadata.create_all(bind=engine)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Kerala Technological University Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(FRONTEND_DIR / "front.html")


@app.get("/health")
def health():
    return {"status": "Kerala Technological University Assistant Running"}


@app.post("/signup", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED)
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(models.User).filter(
        models.User.username == user.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists.")

    db_user = models.User(
        registration_number=user.registration_number,
        email=user.email,
        username=user.username,
        semester=user.semester,
        password=hash_password(user.password),
    )
    db.add(db_user)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this username, email, or registration number already exists.",
        )

    db.refresh(db_user)
    return db_user


@app.post("/login", response_model=schemas.LoginResponse)
def login(credentials: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(
        models.User.username == credentials.username).first()
    if not user or not verify_password(credentials.password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid username or password.")

    access_token = create_access_token({"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "user": user}


@app.post("/reset-password")
def reset_password(payload: schemas.PasswordResetRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(
        models.User.username == payload.username,
        models.User.email == payload.email,
    ).first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account matched that username and email.",
        )

    user.password = hash_password(payload.new_password)
    db.commit()

    return {"message": "Password reset successful. You can now log in with the new password."}


@app.post("/chat", response_model=schemas.ChatReply)
def chat(payload: schemas.ChatRequest, db: Session = Depends(get_db)):
    if not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty.")

    user = None
    if payload.user_id is not None:
        user = db.query(models.User).filter(
            models.User.id == payload.user_id).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    try:
        reply = ask_llm(payload.message, mode=payload.mode,
                        student_name=user.username if user else None,
                        session_key=f"user:{user.id}" if user else None)
    except Exception:
        reply = "I couldn't reach the assistant service right now. Please try again in a moment."

    if user is not None:
        db_chat = models.Chat(
            user_id=payload.user_id,
            message=payload.message,
            response=reply,
        )
        db.add(db_chat)
        db.commit()

    return {"response": reply}


@app.get("/chat/history", response_model=list[schemas.ChatHistoryOut])
def get_chat_history(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return (
        db.query(models.Chat)
        .filter(models.Chat.user_id == current_user.id)
        .order_by(models.Chat.id.desc())
        .all()
    )


@app.post("/upload")
async def upload_pdf(
    files: list[UploadFile] = File(...),
    relative_paths: list[str] = Form(default=[]),
    subject_override: str | None = Form(default=None),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not getattr(current_user, "is_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Admin access required.")

    try:
        return await handle_uploads(files, relative_paths, subject_override=subject_override, db=db)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload files.")


@app.get("/syllabus")
def get_syllabus():
    return {"subjects": get_syllabus_catalog()}


@app.get("/notes")
def get_notes():
    return {"subjects": get_subject_catalog()}


@app.get("/subjects")
def get_subjects():
    return {"subjects": get_subject_catalog(), "syllabus": get_syllabus_catalog()}


@app.get("/pyqs")
def get_pyqs(db: Session = Depends(get_db)):
    return {"subjects": get_pyq_catalog(db=db)}


@app.post("/train-rag")
def train_rag(current_user: models.User = Depends(get_current_user)):
    if not getattr(current_user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )

    result = reindex_all_pdfs(force=True)
    return {
        "message": "Training complete. Local study documents were reindexed for AI retrieval.",
        "stats": result,
    }


@app.get("/train-rag-status")
def train_rag_status(current_user: models.User = Depends(get_current_user)):
    if not getattr(current_user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return get_training_status()


@app.post("/generate-mocktest")
def generate_mocktest(
    request: schemas.MockTestRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from .llm import generate_structured_mock_test_from_context
    from .pyq_engine import query_pyq_context
    from .rag_engine import query_module_context, query_syllabus_context

    pyq_context = query_pyq_context(request.subject, query=f"module {request.module_number} mock test exam pattern", k=12)
    notes_context = query_module_context(request.subject, request.module_number, k=8)
    syllabus_context = query_syllabus_context(request.subject, request.module_number, k=4)
    context = (
        f"PYQ knowledge (pattern + difficulty, highest priority):\n{pyq_context}\n\n"
        f"Notes knowledge (concept clarity):\n{notes_context}\n\n"
        f"Syllabus knowledge (topic boundaries):\n{syllabus_context}"
    )
    questions = generate_structured_mock_test_from_context(
        request.subject,
        request.module_number,
        context,
        num_questions=request.num_questions,
        regenerate=request.regenerate,
    )
    if not questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not generate a grounded mock test from the available resources.",
        )

    attempts = db_attempt_summary(db, current_user.id, request.subject, request.module_number)

    return {
        "duration_seconds": 15 * 60,
        "subject": request.subject,
        "module_number": request.module_number,
        "questions": questions,
        "history": attempts,
    }


def db_attempt_summary(db: Session, user_id: int, subject: str, module_number: int) -> list[dict]:
    attempts = (
        db.query(models.MockTestAttempt)
        .filter(
            models.MockTestAttempt.user_id == user_id,
            models.MockTestAttempt.subject == subject,
            models.MockTestAttempt.module_number == module_number,
        )
        .order_by(models.MockTestAttempt.id.desc())
        .limit(10)
        .all()
    )
    return [
        {
            "id": attempt.id,
            "score": attempt.score,
            "total_questions": attempt.total_questions,
            "created_at": attempt.created_at,
        }
        for attempt in attempts
    ]


@app.post("/submit-mocktest")
def submit_mocktest(
    request: schemas.MockTestSubmitRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if len(request.answers) != len(request.questions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Answer count does not match question count.",
        )

    incorrect = []
    score = 0
    for index, question in enumerate(request.questions):
        selected = request.answers[index]
        is_correct = selected == question.correct_answer
        if is_correct:
            score += 1
            continue
        incorrect.append(
            {
                "number": index + 1,
                "question": question.question,
                "selected_answer": selected,
                "correct_answer": question.correct_answer,
                "selected_option": question.options.get(selected or "", "Not answered"),
                "correct_option": question.options.get(question.correct_answer, ""),
            }
        )

    attempt = models.MockTestAttempt(
        user_id=current_user.id,
        subject=request.subject,
        module_number=request.module_number,
        score=score,
        total_questions=len(request.questions),
        incorrect_questions=json.dumps(incorrect, ensure_ascii=True),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)

    history = (
        db.query(models.MockTestAttempt)
        .filter(
            models.MockTestAttempt.user_id == current_user.id,
            models.MockTestAttempt.subject == request.subject,
            models.MockTestAttempt.module_number == request.module_number,
        )
        .order_by(models.MockTestAttempt.id.desc())
        .limit(10)
        .all()
    )

    return {
        "score": score,
        "total_questions": len(request.questions),
        "incorrect_questions": incorrect,
        "history": [
            {
                "id": item.id,
                "score": item.score,
                "total_questions": item.total_questions,
                "created_at": item.created_at,
            }
            for item in history
        ],
    }


@app.post("/generate-flashcard")
def generate_flashcard(request: schemas.FlashCardRequest, current_user: models.User = Depends(get_current_user)):
    from .llm import generate_flashcards_from_context
    from .pyq_engine import query_pyq_context
    from .rag_engine import query_module_context, query_subject_notes_context

    pyq_context = query_pyq_context(request.subject, k=12)
    if request.module_number is not None:
        notes_context = query_module_context(request.subject, request.module_number, k=8)
    else:
        notes_context = query_subject_notes_context(request.subject, query="flashcards key terms common concepts", k=6)
    context = f"PYQ knowledge (high priority):\n{pyq_context}\n\nNotes knowledge:\n{notes_context}"
    flashcard_content = generate_flashcards_from_context(
        request.subject,
        request.module_number,
        context,
        num_cards=request.num_cards,
        regenerate=request.regenerate,
    )

    return {"flashcards": flashcard_content}


app.mount("/data", StaticFiles(directory=DATA_PATH), name="data")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
