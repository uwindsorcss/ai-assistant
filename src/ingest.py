"""
    Pipeline to ingest data from the input corpus and build out the vector database.
    ONLY RUN IF YOU ARE RECREATING THE DATABASE.
"""
import uuid

import pandas as pd
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from uwin_ai_assistant import config
from uwin_ai_assistant.clients import openai_client, qdrant_client

# Delete previous collection
qdrant_client.delete_collection(collection_name="ai_assistant")

# Configure Qdrant collection
qdrant_client.create_collection(
    collection_name="ai_assistant",
    vectors_config=VectorParams(
        size=config.EMBEDDING_DIM,  # Output dimensionality of the embedding model
        distance=Distance.COSINE,  # Use cosine similarity for embeddings
    ),
)

# Load input corpus into a DataFrame
df = pd.read_csv(
    config.DATA_PATH,
    delimiter=",",
    header=None,
    names=["title", "content"],
)

# Converting blocks into points (the central entity of Qdrant)
points = []
for _, block in df.iterrows():
    if block["content"] == "content":
        continue  # Skip header
    response = openai_client.embeddings.create(
        input=block["title"] + " " + block["content"],
        model=config.EMBEDDING_MODEL,
    )
    embeddings = response.data[0].embedding
    point_id = str(uuid.uuid4())  # Generate a unique ID for each point
    points.append(
        PointStruct(
            id=point_id, payload={"page_content": block["content"]}, vector=embeddings
        )
    )

# Insert points into Qdrant
qdrant_client.upsert(collection_name="ai_assistant", wait=True, points=points)
