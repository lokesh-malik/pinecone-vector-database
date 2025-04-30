import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# === CONFIG ===
PINECONE_API_KEY = "pcsk_8ffCz_8932ui7hFxrBL1FZEGkuTF8ReGX5edN5gmo69HP8pEE8wUTkYWSg9xyYjeF1A13"
PINECONE_INDEX_NAME = "lokeshmalik"

def main():
    # === Initialize Pinecone client ===
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"Connected to index: {PINECONE_INDEX_NAME}")

    # For search functionality
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Define all functions first before using them
    def delete_by_ids():
        """Delete vectors by their IDs"""
        ids_input = input("Enter vector IDs to delete (comma-separated): ")
        ids_to_delete = [id.strip() for id in ids_input.split(",")]
        
        if not ids_to_delete:
            print("No IDs provided.")
            return
            
        # Confirm deletion
        confirm = input(f"Are you sure you want to delete {len(ids_to_delete)} vectors? (y/n): ")
        if confirm.lower() != 'y':
            print("Deletion cancelled.")
            return
            
        try:
            # Delete the vectors
            index.delete(ids=ids_to_delete)
            print(f"Successfully deleted {len(ids_to_delete)} vectors.")
        except Exception as e:
            print(f"Error deleting vectors: {e}")

    def delete_by_filter():
        """Delete vectors using metadata filters"""
        print("\nDelete vectors by metadata filter")
        print("Example filters:")
        print("- Delete by source file: {\"source\": \"example.pdf\"}")
        print("- Delete by page number: {\"page\": 5}")
        
        # Get filter as JSON string
        filter_str = input("\nEnter metadata filter as JSON (or 'cancel'): ")
        if filter_str.lower() == 'cancel':
            return
            
        try:
            import json
            filter_dict = json.loads(filter_str)
            
            # First, get count of matching vectors
            # Note: This is an approximation as Pinecone doesn't directly support counting
            sample_results = index.query(
                vector=[0.0] * 384,  # Dummy vector
                filter=filter_dict,
                top_k=1,
                include_metadata=False
            )
            
            # The count is approximate - we'll need to warn the user
            print(f"Filter will match approximately {len(sample_results['matches'])} or more vectors.")
            
            # Confirm deletion
            confirm = input("Are you sure you want to delete these vectors? (y/n): ")
            if confirm.lower() != 'y':
                print("Deletion cancelled.")
                return
                
            # Delete the vectors
            index.delete(filter=filter_dict)
            print("Successfully deleted vectors matching the filter.")
        except json.JSONDecodeError:
            print("Invalid JSON format. Please try again.")
        except Exception as e:
            print(f"Error: {e}")

    def search_and_delete():
        """Search for vectors by text and then delete them"""
        query = input("Enter search text: ")
        top_k = int(input("Maximum number of results to show: "))
        
        # Encode the query text
        query_embedding = model.encode(query).tolist()
        
        # Search in Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        if not results['matches']:
            print("No matching vectors found.")
            return
            
        # Display results
        print("\n===== Search Results =====")
        ids_to_delete = []
        
        for i, match in enumerate(results['matches']):
            print(f"Result {i+1} [ID: {match['id']}] (Score: {match['score']:.4f})")
            
            # Show metadata preview
            if 'metadata' in match and match['metadata']:
                if 'text' in match['metadata']:
                    # Show a preview of the text
                    text_preview = match['metadata']['text'][:100] + "..." if len(match['metadata']['text']) > 100 else match['metadata']['text']
                    print(f"Text preview: {text_preview}")
                
                # Show other metadata
                for key, value in match['metadata'].items():
                    if key != 'text':  # We already showed a preview of the text
                        print(f"{key}: {value}")
            
            # Ask if this result should be deleted
            delete_this = input(f"Delete this vector? (y/n): ")
            if delete_this.lower() == 'y':
                ids_to_delete.append(match['id'])
            
            print()
        
        # Delete selected vectors
        if ids_to_delete:
            try:
                index.delete(ids=ids_to_delete)
                print(f"Successfully deleted {len(ids_to_delete)} vectors.")
            except Exception as e:
                print(f"Error deleting vectors: {e}")
        else:
            print("No vectors selected for deletion.")

    def delete_all():
        """Delete all vectors in the index"""
        print("⚠️ WARNING: This will delete ALL vectors from your index!")
        print("⚠️ This action cannot be undone!")
        
        # Multiple confirmations for safety
        confirm1 = input("Are you absolutely sure? Type 'DELETE ALL' to confirm: ")
        if confirm1 != "DELETE ALL":
            print("Deletion cancelled.")
            return
            
        confirm2 = input("This is your final warning. Type 'YES I AM SURE' to proceed: ")
        if confirm2 != "YES I AM SURE":
            print("Deletion cancelled.")
            return
            
        try:
            # Delete all vectors
            index.delete(delete_all=True)
            print("All vectors have been deleted from the index.")
        except Exception as e:
            print(f"Error deleting all vectors: {e}")

    # === Delete options menu ===
    while True:
        print("\n===== PINECONE DATA DELETION MENU =====")
        print("1. Delete vectors by IDs")
        print("2. Delete vectors by metadata filter")
        print("3. Search for vectors to delete")
        print("4. Delete all vectors (DANGER!)")
        print("5. Quit")
        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            delete_by_ids()
        elif choice == "2":
            delete_by_filter()
        elif choice == "3":
            search_and_delete()
        elif choice == "4":
            delete_all()
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()