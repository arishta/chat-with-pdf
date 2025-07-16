from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os 
import shutil


def load_pdf_and_text(file_path:str):
    # Load the document from sample.pdf using pypdfloader
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def chunk_documents(docs, chunk_size=1000, overlap=200):
    texts = [doc.page_content for doc in docs]
    metadata = [doc.metadata for doc in docs]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return splitter.create_documents(texts, metadata) 

def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create a Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=None
    )
    # All the chunks will now persist in the vector store Chroma DB
    vectorstore.persist()
    return vectorstore


# user_query = "What is the main topic of the document?"
# retrieved_docs = vectorstore.similarity_search_with_score(user_query, k = 2)

# # Retrieved_docs is a tuple of (Document, score)
# # Document is a chunk retrieved from the vector store: 
# # Score is the similarity score of the chunk to the user query
# #  doc.page_content.strip() is the actual text content of the chunk
# # doc.metadata is a dictionary atatched to each chunk when it was created
# # page is a key in that dictionary
# context_parts = []
# for i, (doc, score) in enumerate(retrieved_docs, 1):
#     page = doc.metadata.get("page", "N/A")
#     context_parts.append(f"Context {i} (Page {page}):\n{doc.page_content.strip()}")

# context = "\n\n".join(context_parts)

# prompt = f"""
# You are a helpful assistant that can answer questions about the document.
# Answer the following question: {user_query}

# Use the context below:
# {context}

# Instructions:
# 1. Be concise and accurate
# 2. If the context is insufficient, then say "not enough info"
# 3. Mention the page number(s) where the answer is found

# """

# response = model.invoke(prompt)
# print(response.content)