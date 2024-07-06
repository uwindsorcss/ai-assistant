from . import config
from .clients import cohere_client

def rerank(query, docs):
    """Rerank the documents"""
    print("OG: \n", "\n\n".join([doc.payload["page_content"] for doc in docs]))
    print("\n\n")
    
    reranked_docs = cohere_client.rerank(
        query=query,
        documents=[doc.payload["page_content"] for doc in docs],
        top_n=config.N_VALUE,
        model=config.RERANK_MODEL,
        return_documents=True,
    )
    
    print(f"test: {reranked_docs.results}\n")
    reranked_docs = [result.document.text for result in reranked_docs.results]

    print("Reranked: ", "\n\n".join(reranked_docs))
    print("\n\n")

    return reranked_docs
