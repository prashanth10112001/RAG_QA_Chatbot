from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
import streamlit as st
from time import sleep

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def document_process(path):

    # Document loading
    loader = PyPDFLoader(path)
    docs = loader.load()
    print("loaded")

    # Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)
    print("splitted")



    # Embeddings and Vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )
    print("embedded")

    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True




st.subheader("Document Q&A ChatBot - Ask Anything")

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if not st.session_state.document_uploaded:
    file = st.file_uploader(label="Select your PDF file", type="pdf")
    if file:
        with open("uploaded-document.pdf","wb") as f:
            f.write(file.getvalue())

        with st.spinner("Processing ..."):
            document_process("./uploaded-document.pdf")

        st.markdown("Document uploaded Successfully")
        sleep(2)
        st.rerun()


if st.session_state.document_uploaded and st.session_state.vector_db:

    for oneMessage in st.session_state.messages:
        role = oneMessage["role"]
        content = oneMessage["content"]

        st.chat_message(role).markdown(content)



    query = st.chat_input("ASk Anything...")
    if query:

        st.session_state.messages.append({"role":"user","content":query})


        st.chat_message("user").markdown(query)

        documents = st.session_state.vector_db.similarity_search(query)
        context = ""

        for doc in documents:
            context += doc.page_content + "\n\n"

        prompt = f"""
        You are a helpful assistant and provide answer based on the provided context.
        Context: {context},
        Question: {query}
        """

        result = llm.invoke(prompt)

        print(result.content)
        if isinstance(result.content, list):
            clean_answer = "".join([item.get("text", "") for item in result.content if isinstance(item, dict) and item.get("type") == "text"])
        else:
            clean_answer = result.content

        st.session_state.messages.append({"role":"ai","content":clean_answer})
        st.chat_message("ai").markdown(clean_answer)

