from website_reader import read_website
from chunker import split_into_chunks
from sentence_transformers import SentenceTransformer

# Step 1: Download website content
url = "https://en.wikipedia.org/wiki/Porsche_918_Spyder"

text = read_website(url)

# Step 2: Split the content into chunks
chunks = split_into_chunks(text)

print("Total chunks:", len(chunks))

# Step 3: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Convert chunks into numerical vectors
embeddings = model.encode(
    chunks,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Step 5: Display the results
print("\nEmbedding process completed!")

print("Number of embeddings:", len(embeddings))
print("Vector dimensions:", embeddings.shape)

print("\nFirst embedding (first 10 numbers):")
print(embeddings[0][:10])