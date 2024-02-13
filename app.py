from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
# from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time

from chromadb import Documents, EmbeddingFunction, Embeddings

from langchain_community.vectorstores import Chroma
 


from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List

import chromadb
from langchain_core.retrievers import BaseRetriever


class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        # embeddings = embedding_functions.ollama(input)
        # embedding_function_ollama = OllamaEmbeddings(model="llama2")

        embedding_function_ollama = OllamaEmbeddings(
            base_url="http://20.86.65.94:11434",
            model="llama2",
        )

        # embeddings = embedding_function_ollama.embed_documents(input)
        # embeddings = embedding_function_ollama.embed_documents(input)
        embeddings = embedding_function_ollama.embed_documents(input)
        
        return embeddings
    
    embed_documents = __call__

custom = MyEmbeddingFunction()


persistent_client = chromadb.PersistentClient()

class CustomRetriever1(BaseRetriever):
    def __init__(self, dbs):
        self.dbs = dbs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Use your existing retriever to get the documents
        documents = []
        for db in self.dbs.values():
            documents.extend(db.search(query))
        
        # Sort the documents by "source"
        documents = sorted(documents, key=lambda doc: doc.metadata.get('source'))
        
        return documents
    
    dbs: dict


class CustomRetriever(BaseRetriever):
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [Document(page_content=query)]

if not os.path.exists('files'):
    os.mkdir('files')


if 'template' not in st.session_state:
    template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""
    st.session_state.template = template


if 'prompt' not in st.session_state:
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )
    st.session_state.prompt = prompt

if 'memory' not in st.session_state:
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
        )
    st.session_state.memory = memory

if 'llm' not in st.session_state:
    llm = Ollama(
        base_url="http://20.86.65.94:11434",
        model="llama2",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
    st.session_state.llm = llm



st.title("PDF Chatbot")

langchain_chroma = Chroma(
    client=persistent_client,
    collection_name="collection_name",
    embedding_function=custom,
    )

def pdf_document_qa(
    file_name = "tesla-earnings-report.pdf",
    user_input = "What is the per share revenue for Meta during 2023?"
    ):
    loader = PyPDFLoader("files/"+file_name)
    data = loader.load()

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    all_splits = text_splitter.split_documents(data)
 
   
    # st.session_state.vector_store = Chroma.from_documents(
    #     documents=all_splits,
    #     embedding_function=OllamaEmbeddings(model="llama2")
    #     )
    docs = all_splits

    collection = persistent_client.get_or_create_collection("collection_name")
    # collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

    # langchain_chroma = Chroma(
    #     client=persistent_client,
    #     collection_name="collection_name",
    #     embedding_function=custom,
    # )
    # db2 = langchain_chroma.from_documents(docs,custom, persist_directory="./chroma_db")
    db2 = langchain_chroma.from_documents(docs,custom)
    
    print("There are", db2._collection.count(), "in the collection")
    

    # db2 = Chroma.from_documents(docs, custom, persist_directory="./chroma_db")
    # embedding_function = OllamaEmbeddings(model="llama2")
    # load it into Chroma
    # st.session_state.vector_store  = Chroma.from_documents(docs, embedding_function)
    # st.session_state.vector_store  = Chroma.from_documents(docs, custom)
    st.session_state.vector_store  = db2

    # return st.session_state.vector_store

    st.session_state.vector_store.persist()
 
    st.session_state.retriever = st.session_state.vector_store.as_retriever()

    if 'qa_chain' not in st.session_state:
        # st.session_state.qa_chain = RetrievalQA.from_chain_type(
        #     llm=st.session_state.llm,
        #     chain_type='stuff',
        #     retriever=st.session_state.retriever,
        #     verbose=True,
        #     chain_type_kwargs={
        #         "verbose": True,
        #         "prompt": st.session_state.prompt,
        #         "memory": st.session_state.memory,
        #     }
        # )

        # llm_model = Ollama(

        llm_model = st.session_state.llm
        #     base_url="http://20.86.65.94:11434",
        #     model="llama2",
        #     verbose=True,
        #     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        #     )
        retriever = CustomRetriever(db2)

        print("Creating qa_chain")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            retriever=retriever,
            chain_type_kwargs={
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory
                }
        )

        st.session_state.qa_chain = qa_chain
    
    response = st.session_state.qa_chain(user_input)

    return response

    pass


# st.write("Please upload a PDF file.")

test_response = "pdf_document_qa()"
test_response = pdf_document_qa()
st.write(test_response)