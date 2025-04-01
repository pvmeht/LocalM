# from charset_normalizer import detect
# from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import FAISS
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch

# class QARAG:
#     def __init__(self, model_name="facebook/opt-350m"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16,
#             device_map="auto"
#         )
#         hf_pipeline = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             max_new_tokens=50,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9
#         )
#         self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

#         self.qa_prompt = PromptTemplate(
#             input_variables=["question"],
#             template="Question: {question}\nAnswer:"
#         )
#         self.qa_chain = self.qa_prompt | self.llm

#         self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         self.vector_store = None

#         self.rag_prompt = PromptTemplate(
#             input_variables=["question", "context"],
#             template="Context: {context}\nQuestion: {question}\nAnswer:"
#         )
#         self.rag_chain = self.rag_prompt | self.llm

#     from charset_normalizer import detect

#     def add_document(self, file_path):
#         """Add a text file's content to the RAG vector store."""
#         # Read file as bytes first
#         with open(file_path, "rb") as f:
#             byte_content = f.read()
        
#         # Detect encoding
#         detection = detect(byte_content)
#         encoding = detection["encoding"] if detection["encoding"] else "utf-8"
        
#         # Decode bytes to text
#         try:
#             text = byte_content.decode(encoding)
#         except UnicodeDecodeError:
#             raise ValueError(f"Failed to decode '{file_path}' with detected encoding '{encoding}'. Please ensure the file is a valid text file.")
        
#         documents = [text]
#         if self.vector_store is None:
#             self.vector_store = FAISS.from_texts(documents, self.embeddings)
#         else:
#             self.vector_store.add_texts(documents)
            
#     def clear_rag_data(self):
#         self.vector_store = None

#     def run_qa(self, question):
#         result = self.qa_chain.invoke({"question": question})
#         return result.strip()

#     def run_rag(self, question):
#         if self.vector_store is None:
#             return "No documents available for RAG."
#         docs = self.vector_store.similarity_search(question, k=1)
#         if not docs:
#             return "No relevant documents found."
#         context = " ".join([doc.page_content for doc in docs])
#         result = self.rag_chain.invoke({"question": question, "context": context})
#         return result.strip()


from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class QARAG:
    def __init__(self, model_name="facebook/opt-350m"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        hf_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        self.qa_prompt = PromptTemplate(
            input_variables=["question"],
            template="Question: {question}\nAnswer:"
        )
        self.qa_chain = self.qa_prompt | self.llm

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None

        self.rag_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="Context: {context}\nQuestion: {question}\nAnswer:"
        )
        self.rag_chain = self.rag_prompt | self.llm

    def add_document(self, file_path):
        """Add a text file's content to the RAG vector store."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        documents = [text]
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(documents, self.embeddings)
        else:
            self.vector_store.add_texts(documents)

    def clear_rag_data(self):
        self.vector_store = None

    def run_qa(self, question):
        result = self.qa_chain.invoke({"question": question})
        return result.strip()

    def run_rag(self, question):
        if self.vector_store is None:
            return "No documents available for RAG."
        docs = self.vector_store.similarity_search(question, k=1)
        if not docs:
            return "No relevant documents found."
        context = " ".join([doc.page_content for doc in docs])
        result = self.rag_chain.invoke({"question": question, "context": context})
        return result.strip()