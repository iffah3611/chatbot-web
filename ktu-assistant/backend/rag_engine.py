from langchain_groq import ChatGroq
from langchain_community.embeddings
import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.2)
