import os
import uuid
import json
import hashlib
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any, Set
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader
)

# === CONFIG === (Change these values as needed)
PINECONE_API_KEY = "pcsk_8ffCz_8932ui7hFxrBL1FZEGkuTF8ReGX5edN5gmo69HP8pEE8wUTkYWSg9xyYjeF1A13"  
PINECONE_INDEX_NAME = "lokeshmalik"   
DOCUMENTS_DIR = "documents"                 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"        
BATCH_SIZE = 100                            
FILE_TYPES = ["pdf", "txt", "docx", "csv"]  
CHUNK_SIZE = 1000                           
TRACKING_FILE = "processed_files.json"     

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

def calculate_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file to track changes."""
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    except Exception as e:
        print(f"❌ Error calculating hash for {file_path}: {str(e)}")
        return ""

def load_processed_files() -> Dict[str, Dict]:
    """Load record of previously processed files."""
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading tracking file: {str(e)}")
            return {}
    return {}

def save_processed_files(processed_files: Dict[str, Dict]) -> None:
    """Save record of processed files."""
    try:
        with open(TRACKING_FILE, 'w') as f:
            json.dump(processed_files, f, indent=2)
    except Exception as e:
        print(f"❌ Error saving tracking file: {str(e)}")

def get_all_document_files() -> Dict[str, str]:
    """Get all document files and their hashes in the documents directory."""
    file_hashes = {}
    patterns = []
    
    for file_type in FILE_TYPES:
        patterns.append(f"*.{file_type}")
    
    for root, _, files in os.walk(DOCUMENTS_DIR):
        for file in files:
            for pattern in patterns:
                if file.lower().endswith(pattern[1:]):  # Remove the * from pattern
                    file_path = os.path.join(root, file)
                    file_hash = calculate_file_hash(file_path)
                    if file_hash:
                        file_hashes[file_path] = file_hash
    
    return file_hashes

def main():
    print(f"🚀 Starting incremental upload to Pinecone with the following configuration:")
    print(f"   - Index name: {PINECONE_INDEX_NAME}")
    print(f"   - Documents directory: {DOCUMENTS_DIR}")
    print(f"   - Embedding model: {EMBEDDING_MODEL}")
    print(f"   - File types: {', '.join(FILE_TYPES)}")
    print(f"   - Batch size: {BATCH_SIZE}")
    
    # Load records of previously processed files
    processed_files = load_processed_files()
    print(f"📋 Loaded records of {len(processed_files)} previously processed files")
    
    # Get current files and their hashes
    current_files = get_all_document_files()
    print(f"🔍 Found {len(current_files)} document files in {DOCUMENTS_DIR}")
    
    # Determine which files are new or modified
    new_or_modified_files = {}
    for file_path, file_hash in current_files.items():
        if file_path not in processed_files or processed_files[file_path]["hash"] != file_hash:
            new_or_modified_files[file_path] = file_hash
    
    print(f"🆕 Detected {len(new_or_modified_files)} new or modified files to process")
    
    if not new_or_modified_files:
        print("✅ No new or modified documents to process. Exiting.")
        return
    
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
    
    # Load only the new or modified documents
    all_documents = []
    file_to_document_map = {}  # Track which documents came from which files
    
    # Process only files that are new or modified
    for file_type in FILE_TYPES:
        try:
            # Filter files by type
            type_files = {path: hash for path, hash in new_or_modified_files.items() 
                         if path.lower().endswith(f".{file_type}")}
            
            if not type_files:
                continue
                
            print(f"📄 Loading {file_type.upper()} documents...")
            
            for file_path in type_files:
                try:
                    # Use appropriate loader based on file type
                    if file_type == 'pdf':
                        from langchain_community.document_loaders import PyPDFLoader
                        loader = PyPDFLoader(file_path)
                    elif file_type == 'txt':
                        loader = TextLoader(file_path)
                    elif file_type == 'csv':
                        loader = CSVLoader(file_path)
                    elif file_type == 'docx':
                        loader = UnstructuredWordDocumentLoader(file_path)
                    else:
                        continue
                        
                    docs = loader.load()
                    print(f"   - Loaded {len(docs)} documents from {os.path.basename(file_path)}")
                    
                    # Track which documents came from which file
                    for doc in docs:
                        file_to_document_map[id(doc)] = file_path
                        
                    all_documents.extend(docs)
                except Exception as e:
                    print(f"❌ Error loading file {file_path}: {str(e)}")
        except Exception as e:
            print(f"❌ Error processing {file_type} files: {str(e)}")
    
    print(f"✅ Total loaded documents from new/modified files: {len(all_documents)}")
    if not all_documents:
        print("❌ No documents loaded. Exiting.")
        return
    
    # Embed and upload in batches
    vectors = []
    total_uploaded = 0
    processed_document_ids = set()
    
    print(f"🔄 Processing documents in batches of {BATCH_SIZE}...")
    for doc in tqdm(all_documents, desc="Embedding documents"):
        content = doc.page_content.strip()
        if not content:
            continue
        
        doc_id = id(doc)
        processed_document_ids.add(doc_id)
            
        # Create embedding
        embedding = model.encode(content)
        
        # Create vector with metadata
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding.tolist(),
            "metadata": {
                "text": content[:CHUNK_SIZE],  # limit text size in metadata
                "source": doc.metadata.get("source", "Unknown"),
                "file_path": file_to_document_map.get(doc_id, "Unknown"),
                "page": doc.metadata.get("page", 0) if "page" in doc.metadata else None,
                "processed_date": datetime.now().isoformat(),
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
    
    # Update processed files record
    for file_path, file_hash in new_or_modified_files.items():
        processed_files[file_path] = {
            "hash": file_hash,
            "last_processed": datetime.now().isoformat()
        }
    
    # Save updated processed files record
    save_processed_files(processed_files)
    print(f"📝 Updated tracking file with {len(new_or_modified_files)} new or modified files")

if __name__ == "__main__":
    main()