import os
from pinecone import Pinecone
from src.helper import ingest_and_store_index
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
INDEX_NAME = "finance"

pc = Pinecone(api_key=PINECONE_API_KEY)

indexes = pc.list_indexes()
if INDEX_NAME not in [index.name for index in indexes]:
    print(f"Creating new index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine"
    )
    print(f"Index {INDEX_NAME} created successfully")
else:
    print(f"Index {INDEX_NAME} already exists")

RESOURCE_DIR = os.path.join(os.getcwd(), "Resources")

def main():
    print("Starting ingestion and index creation...")
    vectorstore = ingest_and_store_index(RESOURCE_DIR, INDEX_NAME)
    if vectorstore:
        print("Indexing complete. Your vectorstore is ready for querying.")
    else:
        print("Indexing failed. Check the error messages above for details.")

if __name__ == "__main__":
    main()
