
from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(text_chunks):
    """
    Convert text chunks into numerical vectors.
    """

    embeddings = model.encode(
        text_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


def create_question_embedding(question):
    """
    Convert a user's question into a numerical vector.
    """

    embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding


if __name__ == "__main__":

    sample_chunks = [
        "The Porsche 918 Spyder is a plug-in hybrid sports car.",
        "The Porsche 918 Spyder has a V8 engine and electric motors."
    ]

    vectors = create_embeddings(sample_chunks)

    print("Number of chunks:", len(vectors))
    print("Vector dimensions:", vectors.shape)

    print("\nFirst vector:")
    print(vectors[0])
