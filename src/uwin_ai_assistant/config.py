"""Hyperparameters for the AI Assistant"""
import os

# API keys
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # API key for the vector database
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # API key for OpenAI models
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # API key for Cohere models

# Locators
DATA_PATH = "./data/corpus/corpus.csv"  # Path to the input corpus
VECTOR_DB_URL = "https://797a91bd-7ef2-4373-9dde-54ef042bbc74.us-east4-0.gcp.cloud.qdrant.io:6333"  # URL for the vector database
GPT_MODEL = "gpt-4o-mini"  # URL for the GPT model (will be updated to our fine-tuned model in the future)
EMBEDDING_MODEL = "text-embedding-3-small"  # URL for the OpenAI embeddings model (will be updated to our fine-tuned model in the future)
RERANK_MODEL = "rerank-english-v3.0"  # URL for the reranking model (will be updated to our fine-tuned model in the future)

# RAG parameters
EMBEDDING_DIM = 1536  # OpenAI text embedding 3 small's output dimensonality
RERANK = True # Whether to rerank the documents (currently bad performance)
K_VALUE = 10  # Number of documents retrieved
N_VALUE = 3  # Number of documents reranked
