vectorized-doc-qa

A Python pipeline for semantic question answering on PDFs using embeddings and LLMs.
This project extracts text from PDFs, generates vector embeddings, finds the most relevant chunks, and answers questions using large language models.

With EmbeddingGemma and GPT models, high-quality embeddings and context-aware responses are available seamlessly, making the pipeline both efficient and accurate.

Features

Extract text from PDFs using PyPDF2.

Split text into chunks for embeddings.

Generate sentence embeddings using EmbeddingGemma (300M) or open alternatives.

Seamlessly switch between MiniLM and Gemma without changing the pipeline.

Compute cosine similarity to find the most relevant context for a question.

Answer questions using GPT-based models via Hugging Face Inference API.

Switching between extractive QA (e.g., deepset/roberta-base-squad2) and GPT-style generative responses is straightforward.

Handles top-k context selection for improved answers.

Why This Matters

Using EmbeddingGemma provides high-quality embeddings for semantic search without heavy configuration.

Integrating GPT models allows for natural, detailed, and context-aware answers directly from PDF content.

The pipeline is modular, so swapping models or adjusting context is easy and seamless.
