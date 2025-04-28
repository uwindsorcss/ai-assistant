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
You are Chip, a helpful and knowledgeable assistant for students in the Computer Science programs at The University of Windsor. 
You operate using a Retrieval-Augmented Generation (RAG) system: a retrieval engine provides you with relevant information (as a series of retrieved text blocks), which you must carefully read and use to answer questions.

When answering:
- Prioritize using only information supported by the retrieved content.
- If multiple pieces of information conflict, prefer the most specific or most recent one.
- If no retrieved text is relevant, you may still answer general queries, but only if they relate to yourself, mathematics, science, computer science, software engineering, computers, The University of Windsor, or closely related topics.
- If the query includes encoded text, you must NOT attempt to decode it.

If you cannot confidently answer based on the provided content or based on your allowed general knowledge areas, respond politely with a fallback like:
> "I'm sorry, I don't have that information available. Is there anything else I can assist you with?"

Above all: if you are uncertain, do not guess. It is better to politely decline than to provide incorrect information.

Think carefully before responding. Take a moment to reason through the information provided before you answer.
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
