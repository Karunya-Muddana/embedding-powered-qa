import os
os.environ['HF_TOKEN'] = 'YOUR TOKEN HERE'

import PyPDF2
import math
import os
import requests

pdf_path = "/content/test.pdf"
full = ""

with open(pdf_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    for page in reader.pages:
        text = page.extract_text() or ""
        full += text + " "

words = full.split()
x = math.ceil(len(words) / 50)
setofwords = [" ".join(words[i * 50:(i + 1) * 50]) for i in range(x)]

print(len(setofwords))

question = input("Enter your question: ")

import torch
from sentence_transformers import SentenceTransformer

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model_id = "google/embeddinggemma-300M"
model = SentenceTransformer(model_id)

# Encode the sentences on the chosen device
pdfembeddings = model.encode(
    setofwords,
    device=device,      # Move computation to GPU if available
    batch_size=16,      # Adjust depending on memory
    show_progress_bar=True
)

questionembedding = model.encode(question, device=device)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# pdfembeddings: (num_sentences, embedding_dim)
# questionembedding: (embedding_dim,) → reshape to (1, embedding_dim)
question_vec = questionembedding.reshape(1, -1)

# Compute cosine similarity between question and all PDF embeddings
similarities = cosine_similarity(question_vec, pdfembeddings)  # shape: (1, num_sentences)

# Flatten the result
similarities = similarities.flatten()

best_idx = np.argmax(similarities)
best_score = similarities[best_idx]
best_context = setofwords[best_idx]

context = "You are a teacher and your job is to help your student easily answer the question he asked using the context provided -> context = "
context = context + best_context
context = context + " -> question = "
context = context + question
print(context)

import os
import requests

API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

response = query({
    "messages": [
        {
            "role": "user",
            "content": context
        }
    ],
    "model": "openai/gpt-oss-120b:cerebras"
})

print(response["choices"][0]["message"])
