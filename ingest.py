
from website_reader import read_website
from chunker import split_into_chunks
from embedding_engine import create_embeddings
from vector_database import store_chunks


# Porsche information source
SOURCE_URL = "https://en.wikipedia.org/wiki/Porsche_918_Spyder"


def ingest_website():

    print("1. Reading Porsche website...")

    text = read_website(SOURCE_URL)

    if not text:
        print("Could not read website.")
        return

    print("2. Splitting text into chunks...")

    chunks = split_into_chunks(text)

    if not chunks:
        print("No text chunks created.")
        return

    print("Total chunks:", len(chunks))

    print("3. Creating embeddings...")

    embeddings = create_embeddings(chunks)

    print("4. Storing information in Qdrant Cloud...")

    store_chunks(
        chunks,
        embeddings,
        SOURCE_URL
    )

    print("5. Ingestion completed successfully!")


if __name__ == "__main__":
    ingest_website()
