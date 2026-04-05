from pathlib import Path

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


@app.post("/chat", response_model=schemas.ChatReply)
def chat(payload: schemas.ChatRequest, db: Session = Depends(get_db)):
    if not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty.")

    try:
        reply = ask_llm(payload.message)
    except Exception:
        reply = "I couldn't reach the assistant service right now. Please try again in a moment."

    if payload.user_id is not None:
        user = db.query(models.User).filter(
            models.User.id == payload.user_id).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

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
    current_user: models.User = Depends(get_current_user),
):
    if not getattr(current_user, "is_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Admin access required.")

    try:
        return await handle_uploads(files, relative_paths)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload files.")


@app.get("/syllabus")
def get_syllabus():
    from pathlib import Path
    data_path = Path(__file__).resolve().parent.parent / "data"
    syllabus_path = data_path / "notes" / "syllabus"

    if not syllabus_path.exists():
        return {"subjects": []}

    subjects = []
    for pdf_file in syllabus_path.glob("*.pdf"):
        subject_name = pdf_file.stem
        subjects.append({
            "name": subject_name,
            "file": str(pdf_file.relative_to(data_path))
        })

    return {"subjects": subjects}


@app.get("/notes")
def get_notes():
    from pathlib import Path
    data_path = Path(__file__).resolve().parent.parent / "data"
    notes_path = data_path / "notes"

    if not notes_path.exists():
        return {"subjects": []}

    subjects = []
    for item in notes_path.iterdir():
        if item.is_file() and item.suffix.lower() == ".pdf":
            subject_name = item.stem
            subjects.append({
                "name": subject_name,
                "files": [str(item.relative_to(data_path))]
            })
        elif item.is_dir() and item.name != "syllabus":
            subject_name = item.name
            files = [str(f.relative_to(data_path)) for f in item.glob("*.pdf")]
            if files:
                subjects.append({
                    "name": subject_name,
                    "files": files
                })

    return {"subjects": subjects}


@app.post("/generate-mocktest")
def generate_mocktest(request: schemas.MockTestRequest, current_user: models.User = Depends(get_current_user)):
    # Generate mock test based on subject
    # For now, use LLM to generate questions
    from .llm import ask_llm

    prompt = f"Generate a mock test for {request.subject} with {request.num_questions} questions. Include multiple choice and short answer questions."
    test_content = ask_llm(prompt)

    return {"test": test_content}


@app.post("/generate-flashcard")
def generate_flashcard(request: schemas.FlashCardRequest, current_user: models.User = Depends(get_current_user)):
    # Generate flashcards based on subject
    from .llm import ask_llm

    prompt = f"Generate flashcards for {request.subject}. Create {request.num_cards} flashcards with key concepts, definitions, and important points."
    flashcard_content = ask_llm(prompt)

    return {"flashcards": flashcard_content}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
