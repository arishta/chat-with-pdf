🧠 Chat with PDF — RAG-based Question Answering App

This is a lightweight Retrieval-Augmented Generation (RAG) application that lets you ask questions about a PDF document and get accurate, page-cited answers — powered by LangChain, OpenAI, and Chroma.

📸 Demo

Ask questions like:

“What is the main topic of the document?”
“What solution is being proposed?”
“Which page talks about shared VPCs?”

And get answers like:

“The document proposes refactoring AWS VPC connectivity for better scalability. See Page 2.”

🛠️ Tech Stack
1. Chroma DB -> Chroma Vector DB
2. Embedding -> OpenAI text-embedding-3-small
3. LLM -> OpenAI gpt-4o-mini
4. PDF loader -> Langchain PyPDFLoader
5. Chunking -> RecursiveCharacterTextSplitter


## Features: 
1. Load and chunk PDF documents 
2. Embed using OpenAI embeddings
3. Store and persisy vector in the Chroma DB 
4. Perform semantic seach over chunks 
5. Answer questions using the retrieved context 
6. Mention the exact page where the answer was found


## Installation: 
1. Clone the repo 
2. Install dependencies
3. Set openapi key 

## Folder structure:

.
├── data/
│   └── sample.pdf       # Your input PDF
├── chroma_db/           # Vector store will be saved here
├── main.py              # Main app script
└── README.md