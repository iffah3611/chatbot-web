from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    registration_number = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    semester = Column(Integer, nullable=False)
    password = Column(String, nullable=False)
    is_admin = Column(Boolean, nullable=False, default=False)

    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")


class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)

    user = relationship("User", back_populates="chats")


class PreviousQuestionPaper(Base):
    __tablename__ = "previous_question_papers"

    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String, nullable=False, index=True)
    subject_code = Column(String, nullable=True, index=True)
    year = Column(Integer, nullable=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, unique=True, nullable=False, index=True)
    question_count = Column(Integer, nullable=False, default=0)
    uploaded_at = Column(String, nullable=False)


class MockTestAttempt(Base):
    __tablename__ = "mock_test_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    subject = Column(String, nullable=False, index=True)
    module_number = Column(Integer, nullable=True, index=True)
    score = Column(Integer, nullable=False)
    total_questions = Column(Integer, nullable=False)
    incorrect_questions = Column(Text, nullable=False, default="[]")
    created_at = Column(String, nullable=False)
