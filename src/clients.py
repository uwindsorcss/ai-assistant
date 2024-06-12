"""Global access to OpenAI and Qdrant clients"""
from openai import OpenAI
from qdrant_client import QdrantClient

import config

openai_client = OpenAI(
    api_key=config.OPENAI_API_KEY,
)

qdrant_client = QdrantClient(
    url=config.VECTOR_DB_URL,
    api_key=config.QDRANT_API_KEY,
)