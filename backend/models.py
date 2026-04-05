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
