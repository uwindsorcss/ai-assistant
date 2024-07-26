"""Interface with the model for inference"""
from . import config
from . import loader
from .clients import openai_client

SYSTEM_PROMPT = """
    You are a helpful assistant for students in the Computer Science programs at The University of Windsor. You have access to a large corpus of information about the program and can answer questions about courses, faculty, and other aspects of the program. You are here to help students find the information they need to succeed in their studies.
    A retrieval system will provide you with relevant information from the corpus, and you will use this information to answer the student's query. Some information may be irrelevant to the query, so you should use your judgment to determine which information is most useful. 
    In the event that nothing in the corpus is relevant to the query, you should provide a fallback response. For example, you could say, "I'm sorry, I don't have that information in my database. Is there anything else I can help you with?" Do not attempt to answer a question if you are unsure of the answer.
"""

USER_PROMPT_TEMPLATE = """
    Consider the following context containing information about the Computer Science program at The University of Windsor: %s

    Answer the student's query: %s
"""


def generate_response(query: str, return_documents=False):
    """Get a response from the model"""

    # Create augmented prompt
    documents = loader.get_documents(query)
    prompt = USER_PROMPT_TEMPLATE % (documents, query)

    # Generate response
    completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=config.GPT_MODEL,
    )

    if return_documents:
        return completion.choices[0].message.content.strip(), documents
    return completion.choices[0].message.content.strip()
