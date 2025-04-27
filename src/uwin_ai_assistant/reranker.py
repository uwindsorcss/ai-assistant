from . import config
from .clients import cohere_client

def filter_docs(reranked_docs):
    """Return only relevant documents"""
    filtered_docs = [ # Filter out documents below the threshold
        doc
        for doc, score in reranked_docs
        if score >= config.RERANK_THRESHOLD
    ]

    return filtered_docs if len(filtered_docs) > 3 else [doc for doc, _ in reranked_docs[:3]] # Return top 3 documents if less than 3 are relevant

def rerank(query, docs):
    """Rerank the documents"""
    rerank_response = cohere_client.rerank(
        query=config.RERANK_PROMPT_TEMPLATE % query,
        documents=[doc.payload["page_content"] for doc in docs],
        top_n=config.N_VALUE,
        model=config.RERANK_MODEL,
        return_documents=True,
    )
    
    reranked_docs = sorted([ # List of tuples (document, relevance_score)
        (result.document.text, result.relevance_score)
        for result in rerank_response.results
    ], key=lambda x: x[1], reverse=True) # Sort by relevance score


    filtered_docs = filter_docs(reranked_docs)
    print(filtered_docs)

    return filtered_docs
