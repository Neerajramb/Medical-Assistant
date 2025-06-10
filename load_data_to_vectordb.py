# load_data_to_vectordb.py

import os
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# --- Configuration ---
DATA_FILE = 'medical_data.txt'
CHROMA_DB_PATH = 'chroma_db' # Directory where ChromaDB will store its data
COLLECTION_NAME = 'medical_knowledge'
MODEL_NAME = 'all-MiniLM-L6-v2' # A good general-purpose embedding model

def load_and_chunk_data(file_path: str) -> list[str]:
    """
    Loads text from a file and chunks it by lines.
    Each non-empty line or a paragraph separated by double newlines
    can be considered a chunk. For simplicity, we'll treat each
    distinct paragraph as a chunk.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split content by double newlines to get paragraphs
    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
    print(f"Loaded {len(chunks)} chunks from {file_path}")
    return chunks

def main():
    """
    Main function to load data, generate embeddings, and populate ChromaDB.
    """
    print("Starting data loading and embedding process...")

    # Ensure ChromaDB directory exists
    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)

    # 1. Load data
    try:
        documents = load_and_chunk_data(DATA_FILE)
        if not documents:
            print(f"No documents found in {DATA_FILE}. Exiting.")
            return
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please create it with medical information.")
        return

    # 2. Initialize embedding model
    print(f"Loading Sentence Transformer model: {MODEL_NAME}...")
    try:
        # Download the model if not already present
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Error loading Sentence Transformer model: {e}")
        print("Please check your internet connection or model name.")
        return
    print("Model loaded successfully.")

    # 3. Initialize ChromaDB client
    print(f"Initializing ChromaDB at {CHROMA_DB_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Get or create the collection
    # If the collection already exists and has data, this will reuse it.
    # To clear existing data, you might want to uncomment client.delete_collection()
    # client.delete_collection(name=COLLECTION_NAME) # Uncomment to clear existing data before loading
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"ChromaDB collection '{COLLECTION_NAME}' ready.")

    # 4. Generate embeddings and add to ChromaDB
    print("Generating embeddings and adding to ChromaDB...")
    # Generate unique IDs for each document
    ids = [f"doc_{i}" for i in range(len(documents))]

    # Generate embeddings for all documents
    embeddings = model.encode(documents).tolist() # Convert numpy array to list of lists for ChromaDB

    # Add to collection in batches if you have a very large dataset
    # For this small example, we can add all at once
    try:
        if collection.count() > 0:
            print("Collection already contains data. Appending new data if any.")
            # You might want more sophisticated logic here for updates/deduplication
            existing_ids = collection.get(ids=[f"doc_{i}" for i in range(collection.count())])['ids']
            new_documents = []
            new_embeddings = []
            new_ids = []
            for i, doc in enumerate(documents):
                if f"doc_{i}" not in existing_ids:
                    new_documents.append(doc)
                    new_embeddings.append(embeddings[i])
                    new_ids.append(f"doc_{i}")
            if new_documents:
                collection.add(
                    documents=new_documents,
                    embeddings=new_embeddings,
                    ids=new_ids
                )
                print(f"Added {len(new_documents)} new documents to ChromaDB.")
            else:
                print("No new documents to add. ChromaDB is up to date.")
        else:
            collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids
            )
            print(f"Added {len(documents)} documents to ChromaDB.")

        print(f"Total documents in ChromaDB: {collection.count()}")
    except Exception as e:
        print(f"Error adding documents to ChromaDB: {e}")

    print("Data loading and embedding process finished.")

if __name__ == "__main__":
    main()