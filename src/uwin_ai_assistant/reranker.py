from . import config
from .clients import cohere_client

def rerank(query, docs):
    """Rerank the documents"""
    reranked_docs = cohere_client.rerank(
        query=query,
        documents=[doc.payload["page_content"] for doc in docs],
        top_n=config.N_VALUE,
        model=config.RERANK_MODEL,
        return_documents=True,
    )
    
    reranked_docs = [result.document.text for result in reranked_docs.results]

    return reranked_docs
