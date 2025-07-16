# To allow the user to upload a PDF via the browser
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from pdf_utils import load_pdf_and_text, chunk_documents, create_vectorstore
from langchain_openai import ChatOpenAI
from string import Template
import os

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Load prompt from external file
def load_prompt_template():
    with open("prompt_template.txt", "r") as f:
        return Template(f.read())
# Setup a page 
st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("Chat with your PDF")
st.markdown("Upload a PDF and ask questions about it")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    st.success(f"Upload file: {uploaded_file.name}")
    st.write("File size:", uploaded_file.size, "bytes")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    # Load and parse the PDF
    docs = load_pdf_and_text(temp_file_path)

    chunks = chunk_documents(docs)
    # st.subheader("📄 Text Chunks")
    # for i, chunk in enumerate(chunks):
    #     preview = " ".join(chunk.page_content.split())
    #     with st.expander(f"Chunk {i+1}"):
    #         st.text(preview[:1000])

    vectorstore = create_vectorstore(chunks)
    st.subheader("📦 Vector Store")
    st.success("✅ Vector store created and stored in memory")
    st.write(f"Total vectors stored: {vectorstore._collection.count()}")

    st.subheader("Please ask a question!!")
    user_query = st.text_input("What would you like to know about the document?")

    if user_query:
        with st.spinner("🔍 Retrieving relevant chunks..."):
            retrieved_docs = vectorstore.similarity_search_with_score(user_query, k=3)

            # Build context with page numbers
            context_parts = []
            for i, (doc, score) in enumerate(retrieved_docs, 1):
                page = doc.metadata.get("page", "N/A")
                context_parts.append(f"Context {i} (Page {page}):\n{doc.page_content.strip()}")
            
            context = "\n\n".join(context_parts)

            template = load_prompt_template()

            prompt = template.substitute(question=user_query, context=context)

            with st.spinner("🧠 Thinking..."):
                response = llm.invoke(prompt)
                st.subheader("📝 Answer")
                st.markdown(response.content)