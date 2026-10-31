
from embedding_engine import create_question_embedding
from vector_database import search_chunks


def retrieve_context(question, results_count=3):
    """
    Find the most relevant Porsche information in Qdrant.
    """

    # Step 1: Convert the question into a numerical vector
    question_embedding = create_question_embedding(question)

    # Step 2: Search Qdrant for similar vectors
    results = search_chunks(
        question_embedding,
        results_count=results_count
    )

    # Step 3: Extract the matching documents
    documents = results["documents"][0]

    # Step 4: Combine the documents into one context
    context = "\n\n".join(documents)

    return context


if __name__ == "__main__":

    question = "What engine does the Porsche 918 Spyder have?"

    context = retrieve_context(question)

    print("\nRetrieved information:")
    print(context)
