"""Interface with the vector store"""
from clients import openai_client, qdrant_client
import config

def get_documents(query: str):
    """Retrieve documents from the vector store"""
    query_vector = openai_client.embeddings.create(
        model=config.EMBEDDER_URL,
        input=[query],
    ).data[0].embedding
    
    retrieved_docs = qdrant_client.search(
        collection_name="ai_assistant",
        query_vector=query_vector,
        limit=config.K_VALUE,
    )
    return "\n\n".join([result.payload["page_content"] for result in retrieved_docs])
