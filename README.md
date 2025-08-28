# Chat-with-PDFs Chatbot 📄💬

## Introduction

Chat with PDFs is an interactive Python tool that allows you to query multiple PDF documents using natural language. Powered by Google’s Gemini LLM, it provides context-aware responses based only on your uploaded PDFs.

## How It Works

The application follows these steps to provide responses to your questions:

### 1. PDF Loading:
Uploads multiple PDFs and extracts their text using PyPDF2.

### 2. Text Chunking:
Divides extracted text into smaller, manageable chunks for semantic search.

### 3. Language Model:
Uses Google Generative AI embeddings (Gemini LLM) to convert text chunks into numerical representations.

### 4. Similarity Matching:
Compares your question with the text chunks to find the most relevant sections.

### 5. Response Generation:
Generates context-aware, detailed responses from the selected text chunks.

## Dependencies and Installation

1. Install the required dependencies:

 - pip install -r requirements.txt

2. Set up Google API Key:

 - Create a .env file in the project directory and add:

 - GOOGLE_API_KEY=your_gemini_api_key

## Demonstration:

Here are some examples of the PDFs Chatbot in action:

### Upload & Load PDFs:
<img width="301" height="716" alt="Screenshot 2025-08-28 044536" src="https://github.com/user-attachments/assets/f637fbf9-66d1-4937-8fdb-9ba6725846c1" />
<img width="319" height="794" alt="Screenshot 2025-08-28 044436" src="https://github.com/user-attachments/assets/20580f63-aa22-4701-9891-d7c193b4f790" />

### Chatting with PDFs:
<img width="1769" height="894" alt="Screenshot 2025-08-28 044327" src="https://github.com/user-attachments/assets/84b64969-eac4-43eb-8a7a-512b1703ed7f" />


Video to get better understanding : https://github.com/user-attachments/assets/16e9b539-260d-4f67-a546-dd79fa0975b8

## Features

- Query multiple PDFs simultaneously.

- LLM-powered natural language understanding.

- Provides precise answers based on PDF content.

- Supports conversation history for multi-turn questions.

- Interactive Streamlit-based UI with real-time responses.
  
## Notes on Model Usage

- By default, the app uses Gemini-1.5-flash via Google Generative AI.

- Offline usage is possible with locally stored embeddings.

- For faster responses, online models (OpenAI or Google API) can be used with API keys.
