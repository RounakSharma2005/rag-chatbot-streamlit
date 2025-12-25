#Phase 1 imports
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
#Phase 2 imports
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
#Phase 3 imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
# ---------------- AUTH STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
# ---------------- LOGIN UI ----------------
if not st.session_state.logged_in:
    st.title("Login to RAG Chatbot")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()   # ⛔ Stop app here until login
st.title("Upload PDFs for RAG")
def get_vectorstore(uploaded_files):
    """
    Creates a FAISS vectorstore from uploaded PDF files
    """
    documents = []

    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file.name)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# Stop if no files uploaded
if not uploaded_files:
    st.warning("Please upload at least one PDF to continue")
    st.stop()

# Create vectorstore once
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_vectorstore(uploaded_files)

vectorstore = st.session_state.vectorstore







st.title("RAG Chatbot!")

#Setup a session state variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
    #Display all the historical messages
for message in st.session_state.messages:
     st.chat_message(message['role']).markdown(message['content'])


def get_vectorstore(uploaded_files):
    all_documents = []

    for uploaded_file in uploaded_files:
        temp_file_path = f"temp_{uploaded_file.name}"

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore





prompt = st.chat_input("Pass your prompt")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role':'user','content':prompt})
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at Everything, you always gives the best,
                                                       the most accurate and most precise answer.
                                                       Answer the following Question:{user_prompt}.
                                                       Start the answer directly.No small talk please""")


    model = "llama-3.1-8b-instant"
    groq_chat = ChatGroq(
        
      groq_api_key = os.getenv("GROQ_API_KEY"),

      model_name = model 
    )


if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    try:
        vectorstore = get_vectorstore(uploaded_files)

        if vectorstore is None:
            st.error("Vectorstore not created")
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            prompt_template = ChatPromptTemplate.from_template(
                """Use the context below to answer the question.

                Context:
                {context}

                Question:
                {question}
                """
            )

            rag_chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough()
                }
                | prompt_template
                | groq_chat
            )

            response = rag_chain.invoke(prompt)

            st.chat_message("assistant").markdown(response.content)
            st.session_state.messages.append(
                {"role": "assistant", "content": response.content}
            )

    except Exception as e:
        st.error(str(e))



