import os
import uuid
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFDirectoryLoader

# === CONFIG ===
PINECONE_API_KEY = "pcsk_8ffCz_8932ui7hFxrBL1FZEGkuTF8ReGX5edN5gmo69HP8pEE8wUTkYWSg9xyYjeF1A13"
PINECONE_INDEX_NAME = "lokeshmalik"

# === Load PDF documents ===
loader = PyPDFDirectoryLoader("documents")
documents = loader.load()
print(f"✅ Loaded {len(documents)} documents")

# === Use a free Hugging Face embedding model ===
model = SentenceTransformer('all-MiniLM-L6-v2')  # Dimension = 384

# === Initialize Pinecone client (using correct pattern for latest SDK) ===
print("Importing Pinecone...")
from pinecone import Pinecone

print("Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists and create if necessary
print("Checking if index exists...")
indexes = [index.name for index in pc.list_indexes()]
if PINECONE_INDEX_NAME not in indexes:
    print(f"Creating index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine"
    )
else:
    print(f"Index {PINECONE_INDEX_NAME} already exists")

# Connect to index with correct pattern
print(f"Connecting to index: {PINECONE_INDEX_NAME}")
index = pc.Index(PINECONE_INDEX_NAME)
print(f"Successfully connected to index")

# === Embed and upload in batches ===
batch_size = 50
vectors = []

for i, doc in enumerate(documents):
    content = doc.page_content.strip()
    if not content:
        continue
    embedding = model.encode(content)

    vectors.append({
        "id": str(uuid.uuid4()),
        "values": embedding.tolist(),
        "metadata": {
            "text": content[:1000],  # limit for safety
            **doc.metadata
        }
    })

    if len(vectors) >= batch_size or i == len(documents) - 1:
        print(f"Uploading batch of {len(vectors)} vectors...")
        index.upsert(vectors)
        print(f"🔼 Uploaded {len(vectors)} vectors")
        vectors = []

print("✅ All documents embedded and uploaded using free embeddings.")