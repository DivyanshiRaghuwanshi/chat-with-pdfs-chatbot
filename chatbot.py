
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore
    # vector_store.save_local("faiss_index")


def get_conversation_chain(vectorstore):
    
    prompt_template = """
    You are a helpful assistant.
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, transport="grpc")

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
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


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
            with st.spinner("Processing..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                
                
        
if __name__ == '__main__':
    main()        