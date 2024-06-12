"""Hyperparameters for the AI Assistant"""
import os

# API keys
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # API key for the vector database
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # API key for OpenAI models

# Locators
DATA_PATH = "./data/corpus.csv" # Path to the input corpus
VECTOR_DB_URL = "https://797a91bd-7ef2-4373-9dde-54ef042bbc74.us-east4-0.gcp.cloud.qdrant.io:6333" # URL for the vector database
GPT_URL = "gpt-3.5-turbo-16k" # URL for the GPT model (will be updated to our fine-tuned model in the future)
EMBEDDER_URL = "text-embedding-3-small" # URL for the OpenAI embeddings model (will be updated to our fine-tuned model in the future)

# RAG parameters
EMBEDDING_DIM = 1536 # OpenAI text embedding 3 small's output dimensonality
K_VALUE = 3 # Number of documents retrieved
