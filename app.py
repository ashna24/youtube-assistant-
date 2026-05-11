import sys
try:
    import pytubefix
    sys.modules["pytube"] = pytubefix
except ImportError:
    pass

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

load_dotenv()

st.set_page_config(page_title="YT Research Assistant", page_icon="🤖", layout="wide")
st.title("YouTube Assistant!")

st.caption("Chat with any video to extract insights and summaries.")

url = st.text_input("Enter YouTube video URL here:", placeholder="https://youtube.com/...")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background: radial-gradient(circle at top left, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }

    /* The Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
 
    .main-title {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        text-align: center;
        padding-bottom: 20px;
    }

    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px !important;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .stChatInputContainer {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
        border-radius: 30px !important;
        backdrop-filter: blur(20px);
    }

    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
with st.sidebar:
    st.title("Welcome to the YouTube Assistant!")
    st.markdown("---")
    
    st.info("This Youtube assistant uses Gemini 2.5 Flash to analyze video transcripts.")

    st.warning("⚠️ **Important:** Please ensure the YouTube video has closed captions or transcripts enabled. The AI cannot analyze videos without text data.")

st.subheader("💬 Ask a Question")
user_query = st.chat_input("What would you like to know about this video?")


if url:
    with st.status("Analyzing Video...", expanded=True) as status:
        try:     
            loader = YoutubeLoader.from_youtube_url(url, add_video_info = False , language = ["en", "en-US", "hi", "ur"], translation = "en")
            data= loader.load()

        except Exception as e:
            status.update(label="Transcript Error", state="error", expanded=True)
            st.error("Couldn't extract the text. The video might have captions completely disabled.")
            st.info(f"Developer Details: {e}")
            st.stop()

        if not data:
            status.update(label="Error: No Transcript Found!", state="error", expanded=True)
            st.error("couldn't extract text from this video. It might not have closed captions enabled. Please try a different link!")
            st.stop()

        #chunking 
        textSplitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        text = textSplitter.split_documents(data)

        #embeddings & storage in chroma
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001") 
        vector_db = Chroma.from_documents(text , embeddings)

        # Set up Retriever & Retrieval Chain
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        qa_chain = RetrievalQA.from_chain_type(llm =llm , chain_type = "stuff" , retriever = vector_db.as_retriever())

    st.success("video processed! please ask your questions now!")

    user_question = st.text_input("Ask a question about the video content:")

    if user_question:
        with st.chat_message("user"):
            st.write(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"query": user_query})
                st.write(response["result"])









    

