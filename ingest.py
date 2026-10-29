
from website_reader import read_website
from chunker import split_into_chunks
from embedding_engine import create_embeddings
from vector_database import store_chunks


# Wikipedia source
URL = "https://en.wikipedia.org/wiki/Porsche_918_Spyder"


def ingest_website(url):

    print("\n1. Downloading website...")

    website_text = read_website(url)

    print("Downloaded characters:", len(website_text))

    print("\n2. Splitting text into chunks...")

    chunks = split_into_chunks(
        website_text,
        chunk_size=500,
        overlap=50
    )

    print("Total chunks:", len(chunks))

    print("\n3. Creating embeddings...")

    embeddings = create_embeddings(chunks)

    print("Embedding shape:", embeddings.shape)

    print("\n4. Storing in ChromaDB...")

    store_chunks(chunks, embeddings, url)

    print("\nWebsite ingestion completed!")


if __name__ == "__main__":
    ingest_website(URL)
