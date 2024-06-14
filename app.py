import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
import tempfile
from dotenv import load_dotenv
import os
load_dotenv()
from llama_index.embeddings.mistralai import MistralAIEmbedding

MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Function to save uploaded files to a temporary directory
def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return file_paths

# Function to generate the model based on uploaded files
def generate_model(file_paths):
    try:
        if not file_paths:
            st.error("Please upload files to proceed.")
            return None
        st.info("Loading and processing documents...")
        reader = SimpleDirectoryReader(input_files=file_paths)
        documents = reader.load_data()
        st.info("Splitting text into nodes...")
        text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
        nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)
        st.info("Initializing embedding model and language model...")
        embed_model = MistralAIEmbedding(api_key=MISTRAL_API_KEY)
        llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
        st.info("Creating service context...")
        service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
        st.info("Creating vector index from documents...")
        vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, service_context=service_context, node_parser=nodes)
        vector_index.storage_context.persist(persist_dir="./storage_mini")
        st.info("Loading index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir="./storage_mini")
        index = load_index_from_storage(storage_context, service_context=service_context)
        st.success("PDF loaded successfully!")
        chat_engine = index.as_query_engine(service_context=service_context, similarity_top_k=20)
        return chat_engine
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Main Streamlit application
def main():
    # Set page configuration
    st.set_page_config(
        page_title="RAG Based Chatbot",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Load CSS styles
    with open("streamlit.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Add logo and description
    st.sidebar.title("BRAINY BUDDY")
    st.sidebar.markdown("""
    #### Welcome to our RAG Based Chatbot! This chatbot utilizes a Retrieval-Augmented Generation (RAG) model to provide accurate and relevant responses based on the information contained in the uploaded files.

    Please upload your PDF files, and we'll handle the rest.
    """)

    if "file_paths" not in st.session_state:
        st.session_state.file_paths = None
    if "model" not in st.session_state:
        st.session_state.model = None

    # Sidebar for file upload
    with st.sidebar:
        uploaded_files = st.file_uploader(label='Upload your files', type=['pdf'], accept_multiple_files=True)
        if uploaded_files:
            file_paths = save_uploaded_files(uploaded_files)
            if file_paths != st.session_state.file_paths:
                st.session_state.file_paths = file_paths
                st.session_state.model = generate_model(file_paths)

    if st.session_state.model:
        user_input = st.text_input("Question", key="question_input", placeholder="Ask your question here...", label_visibility="collapsed")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            button = st.button(label='Enter')
        if button:
            with st.spinner():
                response = str(st.session_state.model.query(user_input).response)
                st.write(response)
    else:
        st.info("Please upload files to initiate the chatbot.")

if __name__ == "__main__":
    main()
