# ## RAG Q&A Convertional with PDF Including Chat History
# import streamlit as st
# from langchain.chains import create_history_aware_retriever,create_retriever_chain
# from langchain.chains.combine_documents import create_stuff_document_chain
# from langchain_chroma import Chroma
# from langchain_core.chat_history import BaseChatMessageHistory 
# from langchain.memory import ChatMessageHistory
# from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_core.runnables.history import RunnableWithMessageHistory 
# import os
# from dotenv import load_dotenv
# load_dotenv()
# os.environ["HK_Token"]=os.getenv["HF_token"]
# embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ## Streamlit app
# st.title("Conversational Rag with pDF")
# st.write("Upload PDF's and chat with there content")


# #Input Groq api Key
# api_key=st.text_input("Enter your Groq api key:",type="password")

# ## Chat if groq api key is provided
# if api_key:
#     llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")#llama-3.1-8b-instant

#     ## Chat interface

#     session_id=st.text_input("session_id",value="default_session")
#     #statefull manage chat history

#     if 'store' not in session_state:
#         st.session_state.store={}
#     uploded_files=st.file_uploader("Choose PDF file",type="pdf",accept_multiple_files=False)
#     ## Process upload PDF's
#     if uploded_files:
#         documents=[]
#         for uploded_file in uploded_files:
#             temppdf=f"./tem.pdf"
#             with open(temppdf,"wb") as file:
#                 file.write(uploded_file.getvalue())
#                 file_name=uploded_file.name
#             loader=PyMuPDFLoader(temppdf)
#             docs=loader.load()
#             documents.extend(docs)
#     # Split and create Embeddings
#         text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=300)
#         splitter=text_splitter.split_documents(documents)
#         vectorstore=Chroma.from_documents(documents=splitter,embedding=embeddings)
#         retriver=vectorstore.as_retriever()

#         ## Contactulize prompt

#         context_q_sys_prompt=(
#             "Give a chat History and the latest user question"
#             "which meight refrence context in the chat History"
#         )

#         context_q_prompt=ChatPromptTemplate.from_messages([

#             ("system",context_q_sys_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human","{input}"),
#         ])


#         history_aware_retriver=create_history_aware_retriever(llm,retriver,context_q_sys_prompt)

#         ## Answer question 

#         system_prompt=(

#             "you are a inteligent provide answer as per question"
#             "which meight refrence context in the chat History"
#             "answer concise"
#             "\n\n"
#             "{context}"
#         )
        
#         qa_prompt=ChatPromptTemplate.from_messages([

#             ("system",system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human","{input}"),
#         ])


#         question_answer_chain=create_stuff_document_chain(llm,qa_prompt)
#         rag_chain=create_retriever_chain(history_aware_retriver,question_answer_chain)
#         def get_session_history(session:str)->BaseChatMessageHistory:
#             if session_id not in st.session_state.store:
#                 st.session_state.store[session_id]=ChatMessageHistory()
#             return st.session_state.store[session_id]


#         convertional_rag_chain=RunnableWithMessageHistory(
#             rag_chain.get_session_history,
#             input_messages_key="input",
#             history_messages_key="chathistory",
#             output_messages_key="answer"
#         )
        
#         ## User input
#         user_nput=st.text_input("your_question:")
#         if user_nput:
#             session_history=get_session_history(session_id)
#             resposnse =convertional_rag_chain.invoke(
#                 {"input":user_nput},
#                 config={
#                 "configurable":{"session_id":session_id}
#                 },

#             )
#             st.write(st.session_state.store)
#             st.write("Assistant",resposnse['answer'])
#             st.write("Chat History:",session_history.messages)

# else:
#     st.warning("please enter the Groq API Key")










# import os
# import streamlit as st
# from dotenv import load_dotenv

# from langchain.memory import ConversationBufferMemory
# from langchain_groq import ChatGroq 
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.chains import ConversationalRetrievalChain
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma

# # Load environment variables
# load_dotenv()
# hf_token = os.getenv("HF_token")
# os.environ["HK_Token"] = hf_token

# # Initialize embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Streamlit UI
# st.title("Conversational RAG with PDF")
# st.write("Upload PDFs and chat with their content")

# # Input Groq API Key
# api_key = st.text_input("Enter your Groq API key:", type="password")

# # Only run if API key is provided
# if api_key:
#     llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")  # Use valid Groq model

#     # Session ID input
#     session_id = st.text_input("Session ID", value="default_session")

#     # Initialize session state for history
#     if 'store' not in st.session_state:
#         st.session_state.store = {}

#     # Upload PDF
#     uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)

#     if uploaded_file:
#         # Save uploaded file temporarily
#         temp_path = "./temp.pdf"
#         with open(temp_path, "wb") as f:
#             f.write(uploaded_file.getvalue())

#         # Load and split document
#         loader = PyMuPDFLoader(temp_path)
#         documents = loader.load()

#         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#         split_docs = splitter.split_documents(documents)

#         # Create vector store
#         vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)
#         retriever = vectorstore.as_retriever()

#         # Define chat memory
#         def get_session_history(session_id):
#             if session_id not in st.session_state.store:
#                 st.session_state.store[session_id] = ConversationBufferMemory(
#                     memory_key="chat_history",
#                     return_messages=True
#                 )
#             return st.session_state.store[session_id]

#         # Build Conversational RAG chain
#         memory = get_session_history(session_id)
#         # qa_chain = ConversationalRetrievalChain.from_llm(
#         #     llm=llm,
#         #     retriever=retriever,
#         #     memory=memory,
#         #     return_source_documents=True
#         # )
#         qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=llm,
#             retriever=retriever,
#             memory=memory,
#             return_source_documents=True,
#             output_key="answer"  # 👈 THIS LINE FIXES THE ERROR
#         )

#         # Chat input
#         user_input = st.text_input("Your question:")
#         if user_input:
#             # response = qa_chain.invoke({"question": user_input})
            
#             # st.write(response['answer'])
            

#             response = qa_chain.invoke({"question": user_input})
#             st.markdown("### Assistant:")
#             st.write("Answer:", response["answer"])
#             st.write("Sources:", response["source_documents"])
#                         # Show history
#             st.markdown("### Chat History:")
#             for msg in memory.chat_memory.messages:
#                 st.write(f"**{msg.type.capitalize()}**: {msg.content}")

# else:
#     st.warning("Please enter the Groq API key.")



import os
import streamlit as st
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# Load env vars
load_dotenv()
hf_token = os.getenv("HF_token")
os.environ["HK_Token"] = hf_token

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.title("Conversational RAG with PDF")
st.write("Upload PDFs and chat with their content")

# Input Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    # Set up LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")

    # Chat session ID
    session_id = st.text_input("Session ID", value="default_session")

    # Ensure memory store exists
    if "store" not in st.session_state:
        st.session_state.store = {}

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load and split documents
        loader = PyMuPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)

        # Create vector store
        vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Set up memory for current session
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        memory = st.session_state.store[session_id]

        # Create ConversationalRetrievalChain (fix output_key!)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            # return_source_documents=True
            output_key="answer"  # ✅ This fixes the ValueError
        )

        # User input
        user_input = st.text_input("Your question:")
        if user_input:
            response = qa_chain.invoke({"question": user_input})

            # Show assistant answer
            st.markdown("### Assistant:")
            st.write(response["answer"])

            # Optional: Show sources
            st.markdown("### Sources:")
            # for doc in response["source_documents"]:
            #     st.write(doc.metadata.get("source", "Unknown Source"))
            #     st.write(doc.page_content[:200] + "...")
            # Display sources, safely
            if "source_documents" in response:
                st.write("### Source Documents:")
                for doc in response["source_documents"]:
                    st.write(doc.page_content[:300] + "...")
            else:
                st.warning("No source documents returned.")
                        # Show chat history
            st.markdown("### Chat History")
            for msg in memory.chat_memory.messages:

                st.write(f"**{msg.type.capitalize()}**: {msg.content}")

else:
    st.warning("Please enter your Groq API key to begin.")
