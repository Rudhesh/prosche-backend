
import os
import numpy as np

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Connect to Gemini API
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL_NAME = "gemini-embedding-001"

# Use 768 dimensions to reduce vector storage size
VECTOR_SIZE = 768


def create_embeddings(text_chunks):
    """
    Convert multiple text chunks into numerical vectors using Gemini.
    """

    result = client.models.embed_content(
        model=MODEL_NAME,
        contents=text_chunks,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=VECTOR_SIZE
        )
    )

    embeddings = []

    for item in result.embeddings:
        vector = np.array(item.values, dtype=np.float32)

        # Normalize the vector
        vector = vector / np.linalg.norm(vector)

        embeddings.append(vector)

    return np.array(embeddings)


def create_question_embedding(question):
    """
    Convert a user's question into a numerical vector.
    """

    result = client.models.embed_content(
        model=MODEL_NAME,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=VECTOR_SIZE
        )
    )

    vector = np.array(
        result.embeddings[0].values,
        dtype=np.float32
    )

    # Normalize the vector
    vector = vector / np.linalg.norm(vector)

    return vector


if __name__ == "__main__":

    sample_chunks = [
        "The Porsche 918 Spyder is a plug-in hybrid sports car.",
        "The Porsche 918 Spyder has a V8 engine and electric motors."
    ]

    vectors = create_embeddings(sample_chunks)

    print("Number of chunks:", len(vectors))
    print("Vector dimensions:", vectors.shape)

    print("\nFirst vector:")
    print(vectors[0][:10])
