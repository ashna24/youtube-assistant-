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

st.title("YouTube Research Assistant!")
st.write("Environment is set up and ready!")

url = st.text_input("Enter YouTube video URL here:")

if url:
    with st.spinner("Fetching video and extracting transcript..."):
        #fetch and extract transcript 
        loader = YoutubeLoader.from_youtube_url(url, add_video_info = False)
        data= loader.load()

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
        with st.spinner("thinking..."):
            response = qa_chain.invoke(user_question)
            st.write("Answer:")
            st.write(response)









    

