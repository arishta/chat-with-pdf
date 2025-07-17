from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
import shutil


def load_pdf_and_text(file_path: str):
    """Load the document from PDF file using PyPDFLoader"""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def get_chunking_strategy(num_pages: int) -> dict:
    """Get chunking strategy based on document size"""
    if num_pages < 10:
        return {"chunk_size": 1000, "chunk_overlap": 100}
    elif num_pages < 50:
        return {"chunk_size": 600, "chunk_overlap": 100}
    else:
        return {"chunk_size": 300, "chunk_overlap": 150}


def chunk_documents(docs, chunk_size=200, overlap=50):
    """Chunk documents based on document size strategy"""
    texts = [doc.page_content for doc in docs]
    metadata = [doc.metadata for doc in docs]

    num_pages = len(docs)
    strategy = get_chunking_strategy(num_pages)

    # Ensure these are integers
    chunk_size = int(strategy["chunk_size"])
    overlap = int(strategy["chunk_overlap"])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", " ", ""]
    )

    return splitter.create_documents(texts, metadata)


def create_vectorstore(chunks):
    """Create a Chroma vector store from document chunks"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create a Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=None
    )

    return vectorstore


def query_documents(vectorstore, user_query: str, k: int = 2):
    """Query the vector store and format the response"""
    retrieved_docs = vectorstore.similarity_search_with_score(user_query, k=k)

    # Retrieved_docs is a tuple of (Document, score)
    # Document is a chunk retrieved from the vector store:
    # Score is the similarity score of the chunk to the user query
    # doc.page_content.strip() is the actual text content of the chunk
    # doc.metadata is a dictionary attached to each chunk when it was created
    # page is a key in that dictionary
    context_parts = []
    for i, (doc, score) in enumerate(retrieved_docs, 1):
        page = doc.metadata.get("page", "N/A")
        context_parts.append(f"Context {i} (Page {page}):\n{doc.page_content.strip()}")

    context = "\n\n".join(context_parts)

    return context, retrieved_docs


def generate_response(model, prompt: str) -> str:
    """Generate response using the language model with provided prompt"""
    response = model.invoke(prompt)
    return response.content
