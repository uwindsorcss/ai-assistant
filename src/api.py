"""API Endpoints for the AI Assistant"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference import generate_response

app = FastAPI()

# Allow CORS
origins = [
    "https://css.uwindsor.ca/",  # The official website
    "http://localhost:8000",  # Local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/{query}")
def get_response(query: str):
    """Get a response to a query"""
    return {"response": generate_response(query)}
