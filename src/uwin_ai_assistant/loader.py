"""Interface with the vector store"""
import config as config
from clients import openai_client, qdrant_client
from reranker import rerank


def get_documents(query: str):
    """Retrieve documents from the vector store"""
    query_vector = (
        openai_client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=[query],
        )
        .data[0]
        .embedding
    )

    retrieved_docs = qdrant_client.search(
        collection_name="ai_assistant",
        query_vector=query_vector,
        limit=config.K_VALUE,
    )
    
    if config.RERANK:
        docs = rerank(query, retrieved_docs)
    else:
        docs = [doc.payload["page_content"] for doc in retrieved_docs][:config.N_VALUE+1]

    return "\n\n".join(list(docs))
