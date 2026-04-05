from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    registration_number: str = Field(min_length=1, max_length=50)
    email: str = Field(min_length=3, max_length=255)
    username: str = Field(min_length=3, max_length=50)
    semester: int = Field(ge=1, le=8)
    password: str = Field(min_length=6, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    registration_number: str
    email: str
    username: str
    semester: int
    is_admin: bool


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str


class LoginResponse(TokenOut):
    user: UserOut


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: int | None = None


class ChatReply(BaseModel):
    response: str


class ChatHistoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    message: str
    response: str


class MockTestRequest(BaseModel):
    subject: str = Field(min_length=1)
    num_questions: int = Field(ge=1, le=20, default=10)


class FlashCardRequest(BaseModel):
    subject: str = Field(min_length=1)
    num_cards: int = Field(ge=1, le=50, default=20)
