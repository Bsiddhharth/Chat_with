from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain # combining the entire doc and send it to the context
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()


os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    session_state_defaults = {
        'vectorstore': None,
        'retriever': None,
        'conversation_chain': None,
        'chat_history': [],
        'uploaded_file_names': set()
    }
    
    for key, default_value in session_state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def setup_rag_pipeline(documents):
    """Set up the RAG pipeline with embeddings and retrieval."""
    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    
    # Create vector store and retriever
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    # Configure LLM
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    
    # Contextualization prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, "
         "formulate a standalone question which can be understood "
         "without the chat history. Do NOT answer the question, "
         "just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    # QA prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. "
         "Use the following pieces of retrieved context to answer "
         "the question. If you don't know the answer, say that you "
         "don't know. Use three sentences minimum and keep the "
         "answer concise. Can include any number of words\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    # Create chains
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Conversational RAG chain with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: ChatMessageHistory(),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    return conversational_rag_chain, vectorstore, retriever

def main():
    # Initialize Streamlit app
    st.title("RAG with PDF Uploads")
    st.write("Upload PDFs and chat with their content")
    
    # Initialize session state
    initialize_session_state()

    # Reset session state when the reset button is clicked
    # if st.button("Reset"):
    #     # Clear session state variables
    #     st.session_state.clear()
    #     # Reinitialize the session state
    #     initialize_session_state()
    #     st.success("Session reset successfully!")

    if st.button("Reset"):
        # Clear all session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Reinitialize the session state
        initialize_session_state()
        st.success("Session reset successfully!")
        # Force a rerun of the app to clear the UI
        st.rerun()
    
    # API Key check
    if not os.getenv("GROQ_API_KEY"):
        st.error("Please set the GROQ_API_KEY environment variable.")
        return
    
    # File upload
    uploaded_files = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)
    
    # Process uploaded files
    if uploaded_files:
        # Get current file names
        current_file_names = {file.name for file in uploaded_files}
        
        # Check if new files have been uploaded
        if current_file_names != st.session_state.uploaded_file_names:
            # Update the set of uploaded file names
            st.session_state.uploaded_file_names = current_file_names
            
            # Process PDF documents
            documents = []
            for uploaded_file in uploaded_files:
                # Save the uploaded file temporarily
                with open("./temp.pdf", "wb") as file:
                    file.write(uploaded_file.getvalue())
                
                # Load the PDF
                loader = PyPDFLoader("./temp.pdf")
                docs = loader.load()
                documents.extend(docs)
            
            # Setup RAG pipeline
            st.session_state.conversation_chain, st.session_state.vectorstore, st.session_state.retriever = setup_rag_pipeline(documents)
    
    # Chat interface
    user_input = st.text_input("Ask a question about your documents:")
    
    if user_input and st.session_state.conversation_chain:
        try:
            # Invoke the conversational chain
            response = st.session_state.conversation_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "default_session"}}
            )
            
            # Display the answer
            st.write("Assistant:", response['answer'])
            
            # Update chat history
            st.session_state.chat_history.append({"user": user_input, "assistant": response['answer']})
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Assistant:** {chat['assistant']}")

if __name__ == "__main__":
    main()




# st.title("RAG With PDF uplaods and chat history")
# st.write("Upload Pdf's and chat with their content")

# llm = ChatGroq(groq_api_key = groq_api_key, model_name = "Gemma2-9b-It")

# session_id=st.text_input("Session ID",value="default_session")
# # statefully manages the session history 

# if 'store' not in st.session_state:
#     st.session_state.store={}

# uploaded_files = st.file_uploader("Upload the pdf file", type='pdf', accept_multiple_files=True)

# # process uploaded files
# if uploaded_files:
#     documents = []
#     for uploaded_file in uploaded_files:
#         tempfile = f"./temp.pdf"
#         with open(tempfile,"wb") as file:
#             file.write(uploaded_file.getvalue())
#         # file.name = uploaded_file.name

#         loader= PyPDFLoader(tempfile)  # i think this works only on saved files hence tempfile was created
#         # recheck 
#         docs = loader.load()
#         documents.extend(docs)

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
#     splits = text_splitter.split_documents(documents)
#     vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
#     retriever = vectorstore.as_retriever()  

#     contextualize_q_system_prompt=(
#             "Given a chat history and the latest user question"
#             "which might reference context in the chat history, "
#             "formulate a standalone question which can be understood "
#             "without the chat history. Do NOT answer the question, "
#             "just reformulate it if needed and otherwise return it as is."
#         )

#     contextualize_q_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", contextualize_q_system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human","{input}")
            
#         ]
#     )
    
#     history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

#     ## Answer question

#         # Answer question
#     system_prompt = (
#                 "You are an assistant for question-answering tasks. "
#                 "Use the following pieces of retrieved context to answer "
#                 "the question. If you don't know the answer, say that you "
#                 "don't know. Use three sentences maximum and keep the "
#                 "answer concise."
#                 "\n\n"
#                 "{context}"
#             )
#     qa_prompt = ChatPromptTemplate.from_messages(
#                 [
#                     ("system", system_prompt),
#                     MessagesPlaceholder("chat_history"),
#                     ("human", "{input}"),
#                 ]
#             )
#     question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
#     rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

#     def get_session_history(session:str)->BaseChatMessageHistory:
#             if session_id not in st.session_state.store:
#                 st.session_state.store[session_id]=ChatMessageHistory()
#             return st.session_state.store[session_id]
        
#     conversational_rag_chain=RunnableWithMessageHistory(
#             rag_chain,get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#             output_messages_key="answer"
#         )

#     user_input = st.text_input("Question: ")

#     if user_input:
#             session_history=get_session_history(session_id)
#             response = conversational_rag_chain.invoke(
#                 {"input": user_input},
#                 config={
#                     "configurable": {"session_id":session_id}
#                 },  # constructs a key "abc123" in `store`.
#             )
#             st.write(st.session_state.store)
#             st.write("Assistant:", response['answer'])
#             st.write("Chat History:", session_history.messages)
