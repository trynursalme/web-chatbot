import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import init_chat_model
from langchain.prompts import ChatPromptTemplate

st.set_page_config(page_title="RAG Web Chatbot Demo", page_icon="🤖", layout="wide")
st.title("RAG Web Q&A Chatbot Demo")

# Load environment variables
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY is not set in the environment variables")
    st.stop()
    
if 'vectorstore' not in st.session_state:
    st.session_state.vectorestore = None

url = st.text_input("Enter a URL to load the documents from:", 
                    value="https://www.govinfo.gov/content/pkg/CDOC-110hdoc50/html/CDOC-110hdoc50.htm")

if st.button("Initialize RAG System"):
    with st.spinner("Loading and processing documents..."):
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(documents=documents)
        
        embeddings_model = HuggingFaceEmbeddings(model_name="NovaSearch/stella_en_1.5B_v5")
        st.session_state.vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings_model)
        st.success("RAG System initialized successfully!")
        
if st.session_state.vectorstore is not None:
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Your a helpful assistant that can answer questions about the USA Constitution. Use the provided context to answer the question."),
    ("user", "Question: {question}\nContext: {context}")
])

chain = prompt | llm

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ask a question about the documents")
    question = st.text_area("Enter your question here:")
    
    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                retriever = st.session_state.vectorstore.as_retriever()
                docs = retriever.invoke(question)
                docs_content = "\n\n".join(doc.page_content for doc in docs)
                
                response = chain.invoke({
                    "question": question,
                    "context": docs_content
                })