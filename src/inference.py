"""Interface with the model for inference"""
import os

from clients import openai_client
import config
import loader

SYSTEM_PROMPT = """
    You are a helpful assistant for students in the Computer Science programs at The University of Windsor.
"""

USER_PROMPT_TEMPLATE = """
    Consider the following context containing information about the Computer Science program at The University of Windsor: %s

    Answer the student's query: %s
"""

def generate_response(query: str):
    """Get a response from the model"""

    # Create augmented prompt
    prompt = USER_PROMPT_TEMPLATE % (loader.get_documents(query), query)

    # Generate response
    completion = openai_client.chat.completions.create(
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model="gpt-3.5-turbo-0125",
    )

    return completion.choices[0].message.content.strip()
