
from embedding_engine import create_question_embedding
from vector_database import search_chunks


def retrieve_information(question):

    print("\nUser question:", question)

    # Convert question into an embedding
    question_embedding = create_question_embedding(question)

    # Search ChromaDB
    results = search_chunks(
        question_embedding,
        results_count=3
    )

    print("\nRelevant information:\n")

    documents = results["documents"][0]
    distances = results["distances"][0]

    for i, document in enumerate(documents):

        print(f"\n--- Result {i + 1} ---")

        print("Distance:", distances[i])

        print("Content:", document[:500])


if __name__ == "__main__":

    question = "What is the top speed of the Porsche 918 Spyder?"

    retrieve_information(question)
