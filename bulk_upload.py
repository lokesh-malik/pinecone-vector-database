import os
import uuid
from tqdm import tqdm
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader
)

# === CONFIG ===
PINECONE_API_KEY = "pcsk_8ffCz_8932ui7hFxrBL1FZEGkuTF8ReGX5edN5gmo69HP8pEE8wUTkYWSg9xyYjeF1A13"  
PINECONE_INDEX_NAME = "lokeshmalik"     
DOCUMENTS_DIR = "documents"                 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"        
BATCH_SIZE = 100                            
FILE_TYPES = ["pdf", "txt", "docx", "csv"]  
CHUNK_SIZE = 1000                           

def get_document_loaders(documents_dir: str, file_types: List[str]) -> List[Dict[str, Any]]:
    """Configure document loaders based on specified file types."""
    loaders = []
    
    if 'pdf' in file_types:
        loaders.append({
            'name': 'PDF',
            'loader': PyPDFDirectoryLoader(documents_dir),
            'pattern': '**/*.pdf'
        })
    
    if 'txt' in file_types:
        loaders.append({
            'name': 'Text',
            'loader': DirectoryLoader(documents_dir, glob="**/*.txt", loader_cls=TextLoader),
            'pattern': '**/*.txt'
        })
        
    if 'csv' in file_types:
        loaders.append({
            'name': 'CSV',
            'loader': DirectoryLoader(documents_dir, glob="**/*.csv", loader_cls=CSVLoader),
            'pattern': '**/*.csv'
        })
        
    if 'docx' in file_types:
        loaders.append({
            'name': 'Word',
            'loader': DirectoryLoader(
                documents_dir, 
                glob="**/*.docx", 
                loader_cls=UnstructuredWordDocumentLoader
            ),
            'pattern': '**/*.docx'
        })
    
    return loaders

def main():
    print(f"🚀 Starting bulk upload to Pinecone with the following configuration:")
    print(f"   - Index name: {PINECONE_INDEX_NAME}")
    print(f"   - Documents directory: {DOCUMENTS_DIR}")
    print(f"   - Embedding model: {EMBEDDING_MODEL}")
    print(f"   - File types: {', '.join(FILE_TYPES)}")
    print(f"   - Batch size: {BATCH_SIZE}")
    
    # Load and initialize embedding model
    print(f"📚 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dimension = model.get_sentence_embedding_dimension()
    print(f"   - Embedding dimension: {embedding_dimension}")
    
    # Initialize Pinecone client
    print("🌲 Initializing Pinecone client...")
    from pinecone import Pinecone
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists and create if necessary
    print("📊 Checking if index exists...")
    indexes = [index.name for index in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in indexes:
        print(f"   - Creating index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_dimension,
            metric="cosine"
        )
    else:
        print(f"   - Index {PINECONE_INDEX_NAME} already exists")
    
    # Connect to index
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"   - Successfully connected to index: {PINECONE_INDEX_NAME}")
    
    # Load documents
    all_documents = []
    document_loaders = get_document_loaders(DOCUMENTS_DIR, FILE_TYPES)
    
    for loader_info in document_loaders:
        try:
            print(f"📄 Loading {loader_info['name']} documents...")
            docs = loader_info['loader'].load()
            print(f"   - Loaded {len(docs)} {loader_info['name']} documents")
            all_documents.extend(docs)
        except Exception as e:
            print(f"❌ Error loading {loader_info['name']} documents: {str(e)}")
    
    print(f"✅ Total loaded documents: {len(all_documents)}")
    if not all_documents:
        print("❌ No documents loaded. Exiting.")
        return
    
    # Embed and upload in batches
    vectors = []
    total_uploaded = 0
    
    print(f"🔄 Processing documents in batches of {BATCH_SIZE}...")
    for doc in tqdm(all_documents, desc="Embedding documents"):
        content = doc.page_content.strip()
        if not content:
            continue
            
        # Create embedding
        embedding = model.encode(content)
        
        # Create vector with metadata
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding.tolist(),
            "metadata": {
                "text": content[:CHUNK_SIZE],  # limit text size in metadata
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0) if "page" in doc.metadata else None,
                **{k: v for k, v in doc.metadata.items() if k not in ["text", "source", "page"]}
            }
        })
        
        # Upload batch if reached batch size
        if len(vectors) >= BATCH_SIZE:
            try:
                index.upsert(vectors)
                total_uploaded += len(vectors)
                print(f"   - Uploaded batch: {len(vectors)} vectors (Total: {total_uploaded})")
                vectors = []
            except Exception as e:
                print(f"❌ Error uploading batch: {str(e)}")
    
    # Upload any remaining vectors
    if vectors:
        try:
            index.upsert(vectors)
            total_uploaded += len(vectors)
            print(f"   - Uploaded final batch: {len(vectors)} vectors")
        except Exception as e:
            print(f"❌ Error uploading final batch: {str(e)}")
    
    print(f"✅ Successfully uploaded {total_uploaded} vectors to Pinecone index: {PINECONE_INDEX_NAME}")

if __name__ == "__main__":
    main()