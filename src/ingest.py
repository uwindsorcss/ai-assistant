"""
    Pipeline to ingest data from the input corpus and build out the vector database. 
    ONLY RUN IF YOU ARE RECREATING THE DATABASE.
"""
import pandas as pd

import config
import uuid
from clients import openai_client, qdrant_client
from qdrant_client.http.models import PointStruct, VectorParams, Distance

# Delete previous collection
qdrant_client.delete_collection(collection_name="ai_assistant")

# Configure Qdrant collection
qdrant_client.create_collection(
    collection_name="ai_assistant",
    vectors_config=VectorParams(
        size=config.EMBEDDING_DIM, # Output dimensionality of the embedding model
        distance=Distance.COSINE, # Use cosine similarity for embeddings
    )
)

# Load input corpus into a DataFrame
df = pd.read_csv(
    config.DATA_PATH,
    delimiter=",",
    header=None,
    names=["content"],
)

# Converting blocks into points (the central entity of Qdrant)
points = []
for _, block in df.iterrows():
    if block["content"] == "content":
        continue # Skip header
    response = openai_client.embeddings.create(
        input=block["content"],
        model=config.EMBEDDER_URL,
    )
    embeddings = response.data[0].embedding
    point_id = str(uuid.uuid4()) # Generate a unique ID for each point
    points.append(PointStruct(
        id=point_id,
        payload={"page_content": block["content"]},
        vector=embeddings)
    )

qdrant_client.upsert(
    collection_name="ai_assistant",
    wait=True,
    points=points
)

# OLD LANGCHAIN APPROACH
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Qdrant
# from langchain_community.document_loaders import DataFrameLoader
# # Load the OpenAI embeddings model
# embedder = OpenAIEmbeddings(model=config.EMBEDDER_URL)

# # Load input corpus into a DataFrame
# df = pd.read_csv(config.DATA_PATH)
# loader = DataFrameLoader(df, page_content_column="Content")
# documents = loader.load()

# # Initialize the qdrant vector store
# qdrant = Qdrant.from_documents(
#     documents=documents,
#     embedding=embedder,
#     url=config.VECTOR_DB_URL,
#     collection_name="ai_assistant",
#     api_key=config.QDRANT_API_KEY,
# )
