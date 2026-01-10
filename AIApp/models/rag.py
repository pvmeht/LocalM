# import faiss
# import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from .qa import answer_question  

# # Placeholder for Stable Code 3B
# model_name = "stable-code-3b"  # Update with actual name
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")  # Fallback
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")  # Fallback

# # Simple document store
# documents = ["Sample doc 1", "Sample doc 2"]  # Replace with real data
# doc_embeddings = []

# # Initialize FAISS index
# dimension = 768  # Adjust based on model output
# index = faiss.IndexFlatL2(dimension)

# def initialize_rag():
#     global doc_embeddings, index
#     for doc in documents:
#         inputs = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
#         with torch.no_grad():
#             embedding = model(**inputs).logits.mean(dim=1).detach().numpy()
#         doc_embeddings.append(embedding)
#     doc_embeddings = np.vstack(doc_embeddings)
#     index.add(doc_embeddings)

# def rag_answer(question: str) -> str:
#     inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         q_embedding = model(**inputs).logits.mean(dim=1).detach().numpy()
#     distances, indices = index.search(q_embedding, k=1)
#     context = documents[indices[0][0]]
#     return answer_question(question, context)  # Reuse QA function

# initialize_rag()



# import faiss
# import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Placeholder for Stable Code 3B (using opt-350m as fallback)
# model_name = "facebook/opt-350m"  # Replace with "stable-code-3b" when available
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Simple document store
# documents = ["Sachin_Rajput_Resume.pdf", "Sample doc 2"]  # Replace with real data
# doc_embeddings = []

# # Initialize FAISS index with correct dimension (512 for opt-350m)
# dimension = 512  # Matches opt-350m hidden size
# index = faiss.IndexFlatL2(dimension)

# def initialize_rag():
#     global doc_embeddings, index
#     for doc in documents:
#         inputs = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
#         with torch.no_grad():
#             # Get the last hidden state instead of logits for better representation
#             outputs = model(**inputs, output_hidden_states=True)
#             embedding = outputs.hidden_states[-1].mean(dim=1).detach().numpy()  # Shape: [1, 512]
#         doc_embeddings.append(embedding)
#     doc_embeddings = np.vstack(doc_embeddings)  # Shape: [n_docs, 512]
#     print(f"Embedding shape: {doc_embeddings.shape}")  # Debug output
#     assert doc_embeddings.shape[1] == dimension, f"Dimension mismatch: {doc_embeddings.shape[1]} != {dimension}"
#     index.add(doc_embeddings)

# def rag_answer(question: str) -> str:
#     inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)
#         q_embedding = outputs.hidden_states[-1].mean(dim=1).detach().numpy()  # Shape: [1, 512]
#     distances, indices = index.search(q_embedding, k=1)
#     context = documents[indices[0][0]]
#     # Reuse QA function from qa.py
#     from models.qa import answer_question
#     return answer_question(question, context)

# # Initialize RAG at startup
# initialize_rag()




import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from models.qa import answer_question

# Placeholder model (OPT as fallback)
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def rag_answer(user_id: str, question: str, index, documents) -> str:
    """Compatible with main.py's in-memory store—pass index/docs as args."""
    if not documents:
        return "No documents uploaded for this user."
    
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        q_embedding = outputs.hidden_states[-1].mean(dim=1).detach().numpy()
    
    distances, indices = index.search(q_embedding, k=1)
    if indices[0][0] >= len(documents):
        return "No relevant context found."
    
    context = documents[indices[0][0]]
    return answer_question(question, context)

# No initialize_rag()—handle in main.py