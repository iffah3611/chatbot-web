from fastapi import FastAPI
from rag_engine import rag_answer

app = FastAPI()


@app.get("/")
def home():
    return {"status": "KTU Assistant Running"}


@app.get("/ask")
def ask(q: str):
    answer = rag_answer(q)
    return {"answer": answer}
