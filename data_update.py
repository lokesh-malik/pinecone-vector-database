import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# === CONFIG ===
PINECONE_API_KEY = "pcsk_8ffCz_8932ui7hFxrBL1FZEGkuTF8ReGX5edN5gmo69HP8pEE8wUTkYWSg9xyYjeF1A13"
PINECONE_INDEX_NAME = "lokeshmalik"

def main():
    # === Initialize embedding model ===
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Same model as upload script

    # === Initialize Pinecone client ===
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"Connected to index: {PINECONE_INDEX_NAME}")

    # Define functions first before using them
    def search_text():
        """Search vectors by text similarity"""
        query = input("Enter your search query: ")
        top_k = int(input("How many results to show? "))
        
        # Encode the query text
        query_embedding = model.encode(query).tolist()
        
        # Search in Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Display results
        print(f"\n===== SEARCH RESULTS FOR: '{query}' =====")
        for i, match in enumerate(results['matches']):
            print(f"Result {i+1} [ID: {match['id']}] (Score: {match['score']:.4f})")
            
            # Show metadata preview
            if 'metadata' in match and match['metadata']:
                if 'text' in match['metadata']:
                    # Show a preview of the text (first 100 chars)
                    text_preview = match['metadata']['text'][:100] + "..." if len(match['metadata']['text']) > 100 else match['metadata']['text']
                    print(f"Text preview: {text_preview}")
                
                # Show other metadata
                for key, value in match['metadata'].items():
                    if key != 'text':  # We already showed a preview of the text
                        print(f"{key}: {value}")
            print()
    
    def update_vector_by_id():
        """Update the vector value for a specific ID"""
        vector_id = input("Enter the vector ID to update: ")
        new_text = input("Enter the new text content: ")
        
        # Check if the vector exists
        try:
            existing = index.fetch(ids=[vector_id])
            if not existing.get('vectors') or vector_id not in existing['vectors']:
                print(f"Vector with ID {vector_id} not found.")
                return
        except Exception as e:
            print(f"Error fetching vector: {e}")
            return
            
        # Encode the new text
        new_embedding = model.encode(new_text).tolist()
        
        # Get existing metadata to preserve it
        metadata = existing['vectors'][vector_id].get('metadata', {})
        
        # Update the text field in metadata
        metadata['text'] = new_text
        
        # Perform the update
        try:
            index.upsert(vectors=[{
                "id": vector_id,
                "values": new_embedding,
                "metadata": metadata
            }])
            print(f"Successfully updated vector {vector_id}")
        except Exception as e:
            print(f"Error updating vector: {e}")
    
    def update_metadata_by_id():
        """Update just the metadata for a specific vector ID"""
        vector_id = input("Enter the vector ID to update metadata: ")
        
        # Check if the vector exists
        try:
            existing = index.fetch(ids=[vector_id])
            if not existing.get('vectors') or vector_id not in existing['vectors']:
                print(f"Vector with ID {vector_id} not found.")
                return
        except Exception as e:
            print(f"Error fetching vector: {e}")
            return
        
        # Get existing vector and metadata
        vector_data = existing['vectors'][vector_id]
        metadata = vector_data.get('metadata', {})
        
        # Show current metadata
        print("\nCurrent metadata:")
        for key, value in metadata.items():
            if key == 'text':
                # Show just a preview of the text
                text_preview = value[:100] + "..." if len(value) > 100 else value
                print(f"{key}: {text_preview}")
            else:
                print(f"{key}: {value}")
        
        # Get the metadata key to update
        key = input("\nEnter metadata key to update (or new key): ")
        value = input(f"Enter new value for '{key}': ")
        
        # Update the metadata
        metadata[key] = value
        
        # Perform the update (keeping the same vector values)
        try:
            index.upsert(vectors=[{
                "id": vector_id,
                "values": vector_data['values'],
                "metadata": metadata
            }])
            print(f"Successfully updated metadata for vector {vector_id}")
        except Exception as e:
            print(f"Error updating metadata: {e}")

    # === Search and update options ===
    while True:
        print("\n===== PINECONE DATA UPDATE MENU =====")
        print("1. Search for vectors by text query")
        print("2. Update vector by ID")
        print("3. Update metadata for vector by ID")
        print("4. Quit")
        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            search_text()
        elif choice == "2":
            update_vector_by_id()
        elif choice == "3":
            update_metadata_by_id()
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()