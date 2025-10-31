
import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
  
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# get pdf text → get chunks → vectorstore (embeddings → FAISS) → conversation chain 
# → User asks → retrieve top-k chunks → LLM (prompt + memory) → response & memory update

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
            continue
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks


def _ensure_event_loop():
    """Ensure an asyncio event loop exists in the current thread.
    Needed for some libs that rely on asyncio in Streamlit worker threads.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def get_vectorstore(text_chunks):
    """Create a vector store from text chunks with a robust embeddings fallback chain.
    Order: Google Gemini (best quality) -> FastEmbed (local ONNX) -> HuggingFace (if torch works)
    """
    embeddings = None
    # 1) Try Google Gemini embeddings (high quality, uses API)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        st.info("✅ Using Google Gemini embeddings (text-embedding-004)")
    except Exception as e:
        st.warning(f"Google Gemini embeddings unavailable: {e}")
        # 2) Fallback to FastEmbed (no torch, lightweight ONNX)
        try:
            _ensure_event_loop()
            embeddings = FastEmbedEmbeddings()
            st.info("Using FastEmbed embeddings (ONNX, no torch)")
        except Exception as e2:
            st.warning(f"FastEmbed unavailable: {e2}")
            # 3) Last resort: HuggingFace local embeddings
            try:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.info("Using local HuggingFace embeddings (all-MiniLM-L6-v2)")
            except Exception as e3:
                st.error(f"All embedding options failed! Last error: {e3}")

    _ensure_event_loop()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    
    prompt_template = """
    You are a helpful assistant.
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    
    # Use a model that is available for your API version. Listed by `genai.list_models()`.
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt} 
    )
    
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your PDFs first.")
        return
    
    try:
        response = st.session_state.conversation.invoke({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")


def main():
    
    st.set_page_config(page_title="Chat with PDFs",
                       page_icon="📄")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None    
    
    st.header("Chat with PDFs 📃")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)
    
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    try:
                        # get pdf text
                        raw_text = get_pdf_text(pdf_docs)
                        
                        if not raw_text.strip():
                            st.error("No text could be extracted from the PDFs. Please check your files.")
                        else:
                            # get the text chunks
                            text_chunks = get_text_chunks(raw_text)
                            st.info(f"Created {len(text_chunks)} text chunks from your documents.")
                            
                            # create vector store
                            vectorstore = get_vectorstore(text_chunks)
                            
                            # create conversation chain
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            
                            st.success("PDFs processed successfully! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")
        
        if st.button("Clear Conversation"):
            st.session_state.chat_history = None
            st.success("Conversation cleared!")
                
                
                
        
if __name__ == '__main__':
    main()        