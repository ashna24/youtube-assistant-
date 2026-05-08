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

#the system will fetch the url 
# if the url exists then , the system will fetch the video and extract the transcript
# then the system will split the transcript into chunks
# after chunking, The system will generate embeddings for each chunk: Using the GoogleGenerativeAIEmbeddings model
#The system will store the vectors in a Vector Database (ChromaDB)
#The system will initialize a Retriever
#the system will set up a Retrieval Chain
#The system will display a Chat Interface in Streamlit
#The system will output the AI's response
# The final answer is displayed in the app, providing a concise summary or a direct answer based purely on the video's content.

if url:
    with st.spinner("Fetching video and extracting transcript..."):
        #fetch and extract transcript 
        loader = YoutubeLoader.from_youtube_url(url, add_video_info = True)
        data= loader.load()

        #chunking 
        textSplitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        text = textSplitter.split_documents(data)

        #embeddings & storage in chroma
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_db = Chroma.from_documents(text , embeddings)

        # Set up Retriever & Retrieval Chain
        llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")
        qa_chain = RetrievalQA.from_chain_type(llm =llm , chain_type = "stuff" , retriever = vector_db.as_retriever())

    st.success("video processed! please ask your questions now!")

    user_question = st.text_input("Ask a question about the video content:")

    if user_question:
        with st.spinner("thinking..."):
            response = qa_chain.invoke(user_question)
            st.write("Answer:")
            st.write(response)









    

