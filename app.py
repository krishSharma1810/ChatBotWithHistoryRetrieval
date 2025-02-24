import streamlit as st
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from chromadb.config import Settings
chroma_settings = Settings(
    chroma_db_impl="duckdb+parquet",  # Or another implementation you're using
    persist_directory="./chroma_db",  # Directory to persist the database
    anonymized_telemetry=False
)


from dotenv import load_dotenv
load_dotenv()

os.environ['HUGGINGFACE_TOKEN']=os.getenv('HUGGINGFACE_TOKEN')

embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')


st.title('Conversational RAG with PDF uploads and chat history')

st.write("Upload PDF's and chat with the content")
api_key=st.text_input('Enter your GROQ API key',type='password')


if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")


    session_id=st.text_input('session_ID',value='default_session')

    if 'store' not in st.session_state:
        st.session_state.store={}

    
    uploaded_files=st.file_uploader('Choose a PDF fill',type='pdf',accept_multiple_files=False)


    if uploaded_files is not None:
        documents=[]
        # for uploaded_file in uploaded_files:
        temppdf=f'./temp.pdf'
        with open(temppdf,'wb') as file:
            file.write(uploaded_files.read())
            
        file_name=uploaded_files.name    
        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)

        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings,client_settings=chroma_settings)
        retriever=vectorstore.as_retriever()

    
        contextulize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate the chat history. do not answer the questions,"
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextulize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human",'{input}'),
            ]
        )


        history_aware_retirever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
        system_message=(
            "you are a assistant for question tasks."
            "use the following pieces of retrieved to answer"
            "the question, is you dont know the answer, say that you "
            "dont know . use three sentences maximum and keep the answer concise"
            "\n\n"
            "{content}"
        )


        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_message),
                MessagesPlaceholder('chat_history'),
                ('human',"{input}")
            ]
        )

        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retirever,question_answer_chain)

        def getSessionHistory(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        

        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history=getSessionHistory,input_messages_key='input',
            history_message_key='chathistory',
            output_message_key='answer'
        )

        user_input=st.text_input('Your question:')
        if user_input:
            session_history=getSessionHistory(session_id)
            response=conversational_rag_chain.invoke({
                "input":user_input
            },
            config={
                "configurable":{"session_id":session_id}
            },)

            st.write(st.session_state.store)
            st.write("assistant:",response['answer'])
            st.write('chat history',session_history.message)
else:
    st.warning("Please enter the groq api key")