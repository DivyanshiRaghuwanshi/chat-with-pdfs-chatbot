# chat-with-pdfs-chatbot

## Introduction

Chat with PDFs is a Python tool for querying multiple PDF documents. Ask questions about your PDFs in natural language, and using Large Language Models (LLMs), it can answer your questions from the content of your PDFs. Responses are limited to the PDFs you provide.

## How It Works

The application follows these steps to provide responses to your questions:

### 1. PDF Loading:
The app loads multiple PDF files and extracts their text.

### 2. Text Chunking:
Divides the text into smaller sections for better processing.

### 3. Language Model:
Uses a Large Language Model to transforms text chunks into numerical representations for better understanding.

### 4. Similarity Matching:
Compares your question to the text chunks to find the closest matches.

### 5. Response Generation:
Generates a response based on the selected text chunks.

## Dependencies and Installation

1. Install the required dependencies:

pip install -r requirements.txt

2. Obtain a Hugging Face API token from https://huggingface.co/settings/tokens

 -Create a .env file in the project directory

 - Add your Hugging Face API token to the .env file:
   
   HUGGINGFACE_API_KEY=your_huggingface_api_token

## Demo:

Here are some examples of the PDFs Chatbot in action:

### Upload & Load PDFs:
<img width="415" height="831" alt="Screenshot 2025-08-27 020517" src="https://github.com/user-attachments/assets/652f61af-2b53-44e6-9dee-567a5c7f113a" />
<img width="372" height="742" alt="Screenshot 2025-08-27 011432" src="https://github.com/user-attachments/assets/af0dcf4c-1867-4f66-a12a-57a83a716c8c" />

### Chatting with PDFs:
<img width="1768" height="819" alt="Screenshot 2025-08-27 020705" src="https://github.com/user-attachments/assets/d8ec94eb-506e-486c-a88e-9accd2497895" />
<img width="742" height="881" alt="Screenshot 2025-08-27 020934" src="https://github.com/user-attachments/assets/cec30594-a680-4dd1-9605-ae7ff4210b0e" />

## Features

- Chat with multiple PDFs simultaneously
  
- Uses LLMs for natural language understanding
  
- Provides precise answers based on document content
  
- Supports PDF uploading and processing

## Notes on Model Usage

- By default, the app uses a **free Hugging Face model** downloaded and run locally.

- This allows offline usage and avoids API costs, but **response times may be slower** compared to cloud models.

- For faster responses, you can use an **OpenAI model** by providing your API key in the `.env` file, which can **significantly increase speed and efficiency**.
