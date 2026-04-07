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


class PasswordResetRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(min_length=3, max_length=255)
    new_password: str = Field(min_length=6, max_length=128)


class TokenOut(BaseModel):
    access_token: str
    token_type: str


class LoginResponse(TokenOut):
    user: UserOut


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: int | None = None
    mode: str = Field(default="teacher")


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
    module_number: int = Field(ge=1, le=5)
    num_questions: int = Field(ge=15, le=15, default=15)
    regenerate: bool = False


class MockTestQuestion(BaseModel):
    question: str = Field(min_length=1)
    options: dict[str, str]
    correct_answer: str = Field(pattern="^[ABCD]$")


class MockTestSubmitRequest(BaseModel):
    subject: str = Field(min_length=1)
    module_number: int = Field(ge=1, le=5)
    questions: list[MockTestQuestion] = Field(min_length=1, max_length=50)
    answers: list[str | None] = Field(min_length=1, max_length=50)


class FlashCardRequest(BaseModel):
    subject: str = Field(min_length=1)
    module_number: int | None = Field(default=None, ge=1, le=5)
    num_cards: int = Field(ge=1, le=50, default=20)
    regenerate: bool = False
