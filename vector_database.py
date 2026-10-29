
import chromadb
import hashlib

# Create a persistent database
client = chromadb.PersistentClient(
    path="./porsche_database"
)

# Create or access our collection
collection = client.get_or_create_collection(
    name="porsche_918_knowledge"
)


def store_chunks(chunks, embeddings, source_url):

    source_id = hashlib.sha256(
        source_url.encode()
    ).hexdigest()[:12]

    ids = [
        f"{source_id}_chunk_{i}"
        for i in range(len(chunks))
    ]
    metadatas = [
        {"source": source_url}
        for _ in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(f"Stored {len(chunks)} chunks successfully!")


def search_chunks(question_embedding, results_count=3):

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=results_count
    )

    return results


if __name__ == "__main__":
    print("Porsche vector database is ready!")

    print("Stored chunks:", collection.count())

    print("\nDatabase location:")
    print("./porsche_database")