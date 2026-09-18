# Document Q&A ChatBot

A lightweight Document Q&A Chatbot implementing a Retrieval-Augmented Generation (RAG) workflow using **Streamlit**, **LangChain**, and **Google Gemini**. Upload any PDF and chat with its content in real time.

## Features

- **PDF Ingestion**: Upload, extract, and chunk PDF documents.
- **Vector Search**: Embeddings powered by `gemini-embedding-2-preview` stored in an in-memory vector store.
- **Context-Aware QA**: Grounded answers generated using `gemini-3.5-flash`.
- **Session History**: Persistent conversation thread across interactions.

## Tech Stack

- **UI**: Streamlit
- **LLM & Embeddings**: Google Gemini via `langchain-google-genai`
- **Orchestration**: LangChain
- **PDF Loader**: PyPDF

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Doc_QA_Chatbot.git
cd Doc_QA_Chatbot
```

### 2. Create & activate a virtual environment

```bash
# Windows
python -m venv envrionment
.\envrionment\Scripts\activate

# macOS / Linux
python3 -m venv envrionment
source envrionment/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

## Running the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.
