
import os
import hashlib
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from embedding_engine import VECTOR_SIZE

load_dotenv()

# Connect to Qdrant Cloud
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "porsche_918_knowledge"


def create_collection():

    # Check whether the collection already exists
    if client.collection_exists(COLLECTION_NAME):

        info = client.get_collection(COLLECTION_NAME)

        existing_size = info.config.params.vectors.size

        # Recreate collection if dimensions have changed
        if existing_size != VECTOR_SIZE:
            print("Vector dimensions changed.")
            print("Recreating Qdrant collection...")

            client.delete_collection(COLLECTION_NAME)

    # Create collection if it does not exist
    if not client.collection_exists(COLLECTION_NAME):

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

        print("Qdrant collection created successfully!")


create_collection()


def store_chunks(chunks, embeddings, source_url):

    source_id = hashlib.sha256(
        source_url.encode()
    ).hexdigest()[:12]

    points = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{source_id}_chunk_{i}"
            )
        )

        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "document": chunk,
                    "source": source_url
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Stored {len(points)} chunks in Qdrant!")


def search_chunks(question_embedding, results_count=3):

    query_vector = (
        question_embedding.tolist()
        if hasattr(question_embedding, "tolist")
        else question_embedding
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=results_count,
        with_payload=True
    ).points

    return {
        "ids": [[str(point.id) for point in results]],
        "documents": [[
            point.payload.get("document", "")
            for point in results
        ]],
        "metadatas": [[
            {"source": point.payload.get("source", "")}
            for point in results
        ]],
        "distances": [[
            1 - point.score for point in results
        ]]
    }


if __name__ == "__main__":

    info = client.get_collection(COLLECTION_NAME)

    print("Qdrant connection successful!")
    print("Collection:", COLLECTION_NAME)
    print("Vector dimensions:", info.config.params.vectors.size)
    print("Stored vectors:", info.points_count)
