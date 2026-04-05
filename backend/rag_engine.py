from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_ROOT / "chroma_db"


def _import_langchain_components():
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        return PyPDFLoader, HuggingFaceEmbeddings, RecursiveCharacterTextSplitter, Chroma
    except ImportError:
        return None, None, None, None
    except Exception:
        return None, None, None, None


def _get_embedding_model():
    _, HuggingFaceEmbeddings, _, _ = _import_langchain_components()
    if HuggingFaceEmbeddings is None:
        return None
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def ingest_pdf(file_path: str):
    PyPDFLoader, _, RecursiveCharacterTextSplitter, Chroma = _import_langchain_components()
    if PyPDFLoader is None:
        print(f"LangChain not available, skipping ingestion of {file_path}")
        return

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(documents)

    embedding_model = _get_embedding_model()
    if embedding_model is None:
        print(
            f"Embedding model unavailable, skipping ingestion of {file_path}")
        return

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=str(CHROMA_PATH),
    )
    vectordb.persist()


def get_vectorstore():
    _, _, _, Chroma = _import_langchain_components()
    if Chroma is None or not CHROMA_PATH.exists():
        return None

    embedding_model = _get_embedding_model()
    if embedding_model is None:
        return None

    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embedding_model,
    )


def query_rag(question: str, k: int = 4):
    _, _, _, Chroma = _import_langchain_components()
    if Chroma is None:
        return "Document search is currently unavailable because LangChain is not installed or compatible."

    vectordb = get_vectorstore()
    if vectordb is None:
        return "No documents have been uploaded yet."

    results = vectordb.similarity_search(question, k=k)
    if not results:
        return "No relevant information found."

    return "\n\n".join(doc.page_content for doc in results)
