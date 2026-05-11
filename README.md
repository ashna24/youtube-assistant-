# YouTube Assistant

A premium, AI-powered YouTube assistant built with Streamlit, LangChain, ChromaDB, and Gemini 2.5 Flash.

## What it does:
* Paste any YouTube URL in the sidebar (ensure the video has closed captions/transcripts enabled).
* The system extracts the transcript, processing even multi-lingual auto-generated captions (English, Hindi, Urdu), and translates them on the fly.
* It splits the text into chunks and embeds it into a local vector database using ChromaDB.
* You can then chat directly with the video asking the AI to summarize core concepts, find specific quotes, or explain complex topics.

## Key Features:
* **Glassmorphic UI:** Custom CSS featuring a sleek, cyberpunk-inspired dark mode design with animated visual states.
* **Multilingual RAG Pipeline:** Automatically detects and translates regional transcripts into English for the LLM to process.
* **Fail-Safe Logic:** Robust error handling that gracefully alerts users via the UI if a video lacks transcripts, bypassing fatal application crashes.

## How to run this on your machine:
1. **Clone the repository:** `git clone https://github.com/ashna24/youtube-assistant-.git`
2. **Navigate to the folder:** `cd youtube-assistant-`
3. **Create a virtual environment:** `python -m venv .venv`
4. **Activate it:** * Mac: `source .venv/bin/activate` 
   * Windows: `.venv\Scripts\activate`
5. **Install dependencies:** `pip install streamlit langchain langchain-google-genai pytubefix youtube-transcript-api chromadb python-dotenv`
6. **Add your API Key:** Create a `.env` file in the main project folder and add your Gemini API key: 
   ```text
   GOOGLE_API_KEY="your_key_here"
