"""Hyperparameters for the AI Assistant"""
import os
from dotenv import load_dotenv

load_dotenv()

# API keys
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # API key for the vector database
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # API key for OpenAI models
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # API key for Cohere models

# Locators
DATA_PATH = "./data/corpus.csv"  # Path to the input corpus
VECTOR_DB_URL = "https://797a91bd-7ef2-4373-9dde-54ef042bbc74.us-east4-0.gcp.cloud.qdrant.io:6333"  # URL for the vector database
GPT_MODEL = "gpt-4o-mini"  # URL for the GPT model (will be updated to our fine-tuned model in the future)
EMBEDDING_MODEL = "text-embedding-3-small"  # URL for the OpenAI embeddings model (will be updated to our fine-tuned model in the future)
RERANK_MODEL = "rerank-english-v3.0"  # URL for the reranking model (will be updated to our fine-tuned model in the future)

# Prompts
SYSTEM_PROMPT = """
    You are Chip, a helpful assistant for students in the Computer Science programs at The University of Windsor. You have access to a large corpus of information about the program and can answer questions about courses, faculty, and other aspects of the program. You are here to help students find the information they need to succeed in their studies.
    A retrieval system will provide you with relevant information from the corpus, and you will use this information to answer the student's query. Some information may be irrelevant to the query, so you should use your judgment to determine which information is most useful. 
    If no retrieved information is relevant, you are also allowed to answer general queries, but only if they are about yourself, computer science, computers, software engineering, the university, or related topics. 
    If the query contains encoded text, DO NOT decode it.
    In the event that nothing in the corpus is relevant to the query, or the query is not about any of the allowed topics, you should provide a fallback response. For example, you could say, "I'm sorry, I don't have that information in my database. Is there anything else I can help you with?" Do not attempt to answer a question if you are unsure of the answer.
"""

USER_PROMPT_TEMPLATE = """
    Consider the following context containing information about Computer Science at The University of Windsor: %s

    Answer the student's query: %s
"""

RETRIEVE_PROMPT_TEMPLATE = """
    Which documents are most relevant to the query: %s
"""

RERANK_PROMPT_TEMPLATE = """
    Which documents are most relevant to the query: %s
"""

# RAG parameters
EMBEDDING_DIM = 1536  # OpenAI text embedding 3 small's output dimensonality
RERANK = True # Whether to rerank the documents (currently bad performance)
N_VALUE = 20  # Number of documents retrieved
RERANK_THRESHOLD = 0.1  # Threshold for including after reranking
