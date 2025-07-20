import streamlit as st
import tempfile
import os
from string import Template
from langchain_openai import ChatOpenAI
from pdf_utils import (
    load_pdf_and_text,
    chunk_documents,
    create_vectorstore,
    query_documents,
    generate_response,
)

# Configuration
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_RETRIEVAL_DOCS = 3

@st.cache_resource
def get_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)

def load_prompt_template(template_path: str = "prompt_template.txt") -> Template:
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return Template(f.read())
    except FileNotFoundError:
        return Template("""
You are a helpful assistant that can answer questions about the document.
Answer the following question: $question

Use the context below:
$context

Instructions:
1. Be concise and accurate
2. If the context is insufficient, then say "not enough info"
3. Mention the page number(s) where the answer is found
""".strip())

def display_document_stats(docs, chunks, vectorstore):
    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Pages", len(docs))
    col2.metric("📦 Chunks", len(chunks))
    col3.metric("🔍 Vectors", vectorstore._collection.count())

def display_retrieved_context(retrieved_docs):
    st.subheader("📚 Retrieved Context")
    for i, (doc, score) in enumerate(retrieved_docs, 1):
        page = doc.metadata.get("page", "N/A")
        with st.expander(f"Context {i} - Page {page} (Similarity: {score:.3f})"):
            st.text(doc.page_content.strip())

def cleanup_temp_file(temp_file_path: str):
    try:
        os.unlink(temp_file_path)
    except OSError:
        pass

def init_session_state():
    st.session_state.setdefault("vectorstore", None)
    st.session_state.setdefault("document_name", None)

def process_uploaded_file(uploaded_file):
    st.session_state.document_name = uploaded_file.name

    with st.spinner("📄 Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        try:
            docs = load_pdf_and_text(temp_path)
            chunks = chunk_documents(docs)
            vectorstore = create_vectorstore(chunks)

            st.session_state.vectorstore = vectorstore
            st.success(f"✅ Successfully processed: {uploaded_file.name}")
            display_document_stats(docs, chunks, vectorstore)
        except Exception as e:
            st.error(f"❌ Error processing PDF: {e}")
        finally:
            cleanup_temp_file(temp_path)

def handle_user_query():
    user_query = st.text_input(
        "What would you like to know about the document?",
        placeholder="e.g., What is the main topic of this document?",
        key="user_query"
    )

    with st.expander("💡 Example Questions"):
        st.write("• What is the main topic of this document?")
        st.write("• Can you summarize the key points?")
        st.write("• What are the conclusions mentioned?")
        st.write("• Are there any specific recommendations?")

    if user_query:
        with st.spinner("🔍 Searching for relevant information..."):
            try:
                context, retrieved_docs = query_documents(
                    st.session_state.vectorstore, user_query, k=MAX_RETRIEVAL_DOCS
                )
                template = load_prompt_template()
                prompt = template.substitute(question=user_query, context=context)
                response = generate_response(get_llm(), prompt)

                st.subheader("📝 Answer")
                st.markdown(response)

                if st.checkbox("Show retrieved context", key="show_context"):
                    display_retrieved_context(retrieved_docs)
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")

def main():
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    st.title("📄 Chat with your PDF")
    st.markdown("Upload a PDF document and ask questions about its content")

    init_session_state()

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to start chatting with it"
    )

    if uploaded_file:
        if st.session_state.document_name != uploaded_file.name:
            process_uploaded_file(uploaded_file)
        else:
            st.info(f"📄 Current document: {uploaded_file.name}")

        if st.session_state.vectorstore:
            st.divider()
            st.subheader("💬 Ask a Question")
            handle_user_query()
    else:
        st.info("👆 Please upload a PDF file to get started")

    st.divider()
    st.markdown("---")
    st.markdown(
        "💡 **Tips:** Upload a PDF, wait for processing, then ask specific questions about its content. "
        "The system will find relevant sections and provide answers with page ref."
    )

if __name__ == "__main__":
    main()
