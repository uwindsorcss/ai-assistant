"""Interface with the model for inference"""
from . import config
from . import loader
from .clients import openai_client

def generate_response(query: str, return_documents=False, return_unranked_documents=False):
    """Get a response from the model"""

    # Create augmented prompt
    documents, unranked_documents = loader.get_documents(query, return_unranked_documents=return_unranked_documents)
    prompt = config.USER_PROMPT_TEMPLATE % (documents, query)

    # Generate response
    completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": config.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=config.GPT_MODEL,
    )

    if return_documents:
        if return_unranked_documents:
            return completion.choices[0].message.content.strip(), documents, unranked_documents
        return completion.choices[0].message.content.strip(), documents
    return completion.choices[0].message.content.strip()
