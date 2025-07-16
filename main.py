from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os 
import shutil

# Load the document from sample.pdf using pypdfloader
loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Extract the text and the metadata from the document
text = [d.page_content for d in docs]
metadata = [d.metadata for d in docs]

# Split the text into chunks using RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.create_documents(text, metadata)

print("Total no. of chunks: ", len(chunks))
print("Average chunk size: ", sum(len(chunk.page_content) for chunk in chunks) / len(chunks))
print("A few sample chunks: ---->")

# # Printing the three sample chunks
# for i, chunk in enumerate(chunks[:3], 1):
#     print(f"Page number: {chunk.metadata['page']}")
#     preview = " ".join(chunk.page_content.split())
#     print(f"Chunk {i}: {preview[:100]}...")


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
persist_directory = "./chroma_db"

if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

# Create a Chroma vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

# All the chunks will now persist in the vector store Chroma DB
vectorstore.persist()
print("Vector store was created successfully")
print(f"Total vectors stored: {vectorstore._collection.count()}")

user_query = "What is the main topic of the document?"
retrieved_docs = vectorstore.similarity_search_with_score(user_query, k = 2)

# Retrieved_docs is a tuple of (Document, score)
# Document is a chunk retrieved from the vector store: 
# Score is the similarity score of the chunk to the user query
#  doc.page_content.strip() is the actual text content of the chunk
# doc.metadata is a dictionary atatched to each chunk when it was created
# page is a key in that dictionary
context_parts = []
for i, (doc, score) in enumerate(retrieved_docs, 1):
    page = doc.metadata.get("page", "N/A")
    context_parts.append(f"Context {i} (Page {page}):\n{doc.page_content.strip()}")

context = "\n\n".join(context_parts)

prompt = f"""
You are a helpful assistant that can answer questions about the document.
Answer the following question: {user_query}

Use the context below:
{context}

Instructions:
1. Be concise and accurate
2. If the context is insufficient, then say "not enough info"
3. Mention the page number(s) where the answer is found

"""

response = model.invoke(prompt)
print(response.content)