import os
import threading
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from models import llm
from utils import load_text_document, process_image

# Embeddings and splitter
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Vector store
vector_store = None
vector_store_lock = threading.Lock()

def update_vector_store():
    global vector_store
    with vector_store_lock:
        documents = []
        for dir_name in ["docs", "images"]:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            for filename in os.listdir(dir_name):
                filepath = os.path.join(dir_name, filename)
                if dir_name == "docs":
                    documents.extend(load_text_document(filepath))
                elif dir_name == "images" and filename.endswith((".png", ".jpg", ".jpeg")):
                    documents.append(process_image(filepath))
        
        if documents:
            texts = splitter.split_documents(documents)
            if vector_store is None:
                vector_store = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
            else:
                vector_store.add_documents(texts)

def query_rag(query, temp=0.7, top_p=0.9):
    global vector_store
    with vector_store_lock:
        if vector_store is None:
            update_vector_store()
        if vector_store is None:
            return "No data available yet. Please upload notes or images."
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        llm.temperature = temp
        llm.top_p = top_p
        result = qa_chain({"query": query})
        answer = result["result"]
        if not answer.strip() or "I don't know" in answer.lower():
            return "I don’t have enough information to answer that."
        return answer