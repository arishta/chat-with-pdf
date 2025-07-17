# 🧠 Chat with PDF — RAG-based Question Answering App

This is a lightweight **Retrieval-Augmented Generation (RAG)** application that lets you upload a PDF and ask questions about its content.  
It provides **accurate, page-cited answers** using **LangChain**, **OpenAI**, and **Chroma**.

---

## 📸 Demo

Ask questions like:

- “What is the main topic of the document?”
- “What solution is being proposed?”
- “Which page talks about shared VPCs?”

And get answers like:

> “The document proposes refactoring AWS VPC connectivity for better scalability. See Page 2.”

---

## 🛠️ Tech Stack

| Component        | Tool/Library                         |
|------------------|--------------------------------------|
| 🔍 Vector Store  | [Chroma](https://www.trychroma.com/) |
| 🧠 Embeddings     | `text-embedding-3-small` (OpenAI)    |
| 💬 LLM            | `gpt-4o-mini` (OpenAI)               |
| 📄 PDF Loader     | `PyPDFLoader` (LangChain)            |
| ✂️ Chunking       | `RecursiveCharacterTextSplitter`     |

---

## ✅ Features

1. 📄 Load and chunk PDF documents
2. 🧠 Embed text using OpenAI embeddings
3. 🗃️ Store and persist vectors in Chroma DB
4. 🔍 Perform semantic search over document chunks
5. 💬 Answer questions using retrieved context
6. 📌 Mention the exact **page number(s)** where the answer is found

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/chat-with-pdf-rag.git
cd chat-with-pdf-rag
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Set your OpenAI API key 
export OPENAI_API_KEY="sk-..."


# Run the app 


```
streamlit run app.py
```
You’ll be able to:
	•	Upload any PDF
	•	Ask questions in natural language
	•	Get answers with exact page references
<img width="1720" height="839" alt="Screenshot 2025-07-17 at 2 10 50 PM" src="https://github.com/user-attachments/assets/cfabf03f-07fa-4ce5-976e-a54deab9a6a9" />
