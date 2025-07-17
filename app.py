import streamlit as st
import tempfile
import os
from string import Template
from langchain_openai import ChatOpenAI
from pdf_utils import load_pdf_and_text, chunk_documents, create_vectorstore, query_documents, generate_response


# Configuration
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_RETRIEVAL_DOCS = 3

# Initialize LLM
@st.cache_resource
def get_llm():
    """Initialize and cache the language model"""
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)


def load_prompt_template(template_path: str = "prompt_template.txt") -> Template:
    """Load prompt template from external file"""
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return Template(f.read())
    except FileNotFoundError:
        # Fallback template if file doesn't exist
        fallback_template = """
You are a helpful assistant that can answer questions about the document.
Answer the following question: $question

Use the context below:
$context

Instructions:
1. Be concise and accurate
2. If the context is insufficient, then say "not enough info"
3. Mention the page number(s) where the answer is found
"""
        return Template(fallback_template.strip())


def display_document_stats(docs, chunks, vectorstore):
    """Display document processing statistics"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📄 Pages", len(docs))
    
    with col2:
        st.metric("📦 Chunks", len(chunks))
    
    with col3:
        st.metric("🔍 Vectors", vectorstore._collection.count())


def display_retrieved_context(retrieved_docs):
    """Display retrieved context in expandable sections"""
    st.subheader("📚 Retrieved Context")
    
    for i, (doc, score) in enumerate(retrieved_docs, 1):
        page = doc.metadata.get("page", "N/A")
        similarity_score = f"{score:.3f}"
        
        with st.expander(f"Context {i} - Page {page} (Similarity: {similarity_score})"):
            st.text(doc.page_content.strip())


def cleanup_temp_file(temp_file_path: str):
    """Clean up temporary file"""
    try:
        os.unlink(temp_file_path)
    except OSError:
        pass


def main():
    """Main Streamlit application"""
    # Page setup
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    st.title("📄 Chat with your PDF")
    st.markdown("Upload a PDF document and ask questions about its content")
    
    # Initialize session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to start chatting with it"
    )
    
    if uploaded_file is not None:
        # Check if this is a new file
        if st.session_state.document_name != uploaded_file.name:
            st.session_state.document_name = uploaded_file.name
            
            with st.spinner("📄 Processing PDF..."):
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                
                try:
                    # Load and process the PDF
                    docs = load_pdf_and_text(temp_file_path)
                    chunks = chunk_documents(docs)
                    vectorstore = create_vectorstore(chunks)
                    
                    # Store in session state
                    st.session_state.vectorstore = vectorstore
                    
                    # Display success message and stats
                    st.success(f"✅ Successfully processed: {uploaded_file.name}")
                    display_document_stats(docs, chunks, vectorstore)
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    return
                
                finally:
                    # Clean up temporary file
                    cleanup_temp_file(temp_file_path)
        
        else:
            # File already processed
            st.info(f"📄 Current document: {uploaded_file.name}")
        
        # Query section
        if st.session_state.vectorstore is not None:
            st.divider()
            st.subheader("💬 Ask a Question")
            
            user_query = st.text_input(
                "What would you like to know about the document?",
                placeholder="e.g., What is the main topic of this document?",
                key="user_query"
            )
            
            # Add some example questions
            with st.expander("💡 Example Questions"):
                st.write("• What is the main topic of this document?")
                st.write("• Can you summarize the key points?")
                st.write("• What are the conclusions mentioned?")
                st.write("• Are there any specific recommendations?")
            
            if user_query:
                with st.spinner("🔍 Searching for relevant information..."):
                    try:
                        # Query the document using the utility function
                        context, retrieved_docs = query_documents(
                            st.session_state.vectorstore, 
                            user_query, 
                            k=MAX_RETRIEVAL_DOCS
                        )
                        
                        # Load template and generate response
                        template = load_prompt_template()
                        prompt = template.substitute(question=user_query, context=context)
                        
                        llm = get_llm()
                        
                        with st.spinner("🧠 Generating response..."):
                            response = generate_response(llm, prompt)
                        
                        # Display results
                        st.subheader("📝 Answer")
                        st.markdown(response)
                        
                        # Show retrieved context (optional)
                        if st.checkbox("Show retrieved context", key="show_context"):
                            display_retrieved_context(retrieved_docs)
                            
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")
    
    else:
        st.info("👆 Please upload a PDF file to get started")
    
    # Footer
    st.divider()
    st.markdown("---")
    st.markdown(
        "💡 **Tips:** Upload a PDF, wait for processing, then ask specific questions about its content. "
        "The system will find relevant sections and provide answers with page references."
    )


if __name__ == "__main__":
    main()