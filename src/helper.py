#streamlit run app.py --server.fileWatcherType none


import os
import pinecone
from typing import List
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
import re

load_dotenv()

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Initialize and return the embedding model."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def load_documents(resource_path: str) -> List:
    """Load PDF documents from a directory and its subdirectories."""
    import glob
    pdf_files = glob.glob(os.path.join(resource_path, "**/*.pdf"), recursive=True)
    documents = []
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            doc = loader.load()
            documents.extend(doc)
            print(f"Successfully loaded: {pdf_file}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {str(e)}")
    
    return documents

def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into chunks for better embedding."""
    if not documents:
        print("Warning: No documents to split.")
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    split_docs = splitter.split_documents(documents)
    return split_docs

def create_pinecone_index(documents, embeddings, index_name: str):
    """Update an existing Pinecone index with document embeddings."""
    
    print(f"Adding {len(documents)} document chunks to existing index '{index_name}'")
    vectorstore = Pinecone.from_documents(
        documents, 
        embeddings, 
        index_name=index_name
    )
    return vectorstore

def ingest_and_store_index(
    resource_dir: str, 
    index_name: str, 
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """Process and index PDF documents from a resource directory into an existing Pinecone index."""
    
    print(f"Loading PDF documents from {resource_dir} and subdirectories...")
    docs = load_documents(resource_dir)
    print(f"Loaded {len(docs)} PDF documents.")
    
    print(f"Splitting documents into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    split_docs = split_documents(docs, chunk_size, chunk_overlap)
    print(f"Split documents into {len(split_docs)} chunks.")
    
    print(f"Initializing embeddings using model: {model_name}")
    embeddings = get_embedding_model(model_name)
    
    print(f"Updating existing Pinecone index: {index_name}")
    vectorstore = create_pinecone_index(split_docs, embeddings, index_name)
    print("Pinecone index updated successfully.")
    
    return vectorstore
    
def query_index(query: str, index_name: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", top_k: int = 5):
    """Query the Pinecone index with a given query string."""
    embeddings = get_embedding_model(model_name)
    
    vectorstore = Pinecone.from_existing_index(
        index_name, 
        embeddings
    )
    
    results = vectorstore.similarity_search(query, k=top_k)
    
    return results



def extract_country(query):
    """Extract a country code from the query if a known country is mentioned."""
    country_mapping = {
        # North America
        "usa": "US", "us": "US", "united states": "US",
        "canada": "CA",
        "mexico": "MX",
        # South America
        "brazil": "BR",
        "argentina": "AR",
        "colombia": "CO",
        "chile": "CL",
        "peru": "PE",
        "venezuela": "VE",
        # Europe
        "uk": "GB", "united kingdom": "GB",
        "ireland": "IE",
        "france": "FR",
        "germany": "DE",
        "italy": "IT",
        "spain": "ES",
        "netherlands": "NL",
        "belgium": "BE",
        "switzerland": "CH",
        "austria": "AT",
        "sweden": "SE",
        "norway": "NO",
        "finland": "FI",
        "denmark": "DK",
        "poland": "PL",
        "czech republic": "CZ",
        "greece": "GR",
        "portugal": "PT",
        "russia": "RU",
        # Asia
        "china": "CN",
        "japan": "JP",
        "south korea": "KR", "korea": "KR",
        "india": "IN",
        "pakistan": "PK",
        "bangladesh": "BD",
        "sri lanka": "LK",
        "nepal": "NP",
        "indonesia": "ID",
        "malaysia": "MY",
        "singapore": "SG",
        "thailand": "TH",
        "vietnam": "VN",
        "philippines": "PH",
        "iran": "IR",
        "iraq": "IQ",
        # Middle East
        "israel": "IL",
        "turkey": "TR",
        "saudi arabia": "SA",
        "united arab emirates": "AE", "uae": "AE",
        "qatar": "QA",
        "kuwait": "KW",
        "oman": "OM",
        # Oceania
        "australia": "AU",
        "new zealand": "NZ",
        # Africa
        "south africa": "ZA",
        "nigeria": "NG",
        "egypt": "EG"
    }
    query_lower = query.lower()
    for country_name, code in country_mapping.items():
        if country_name in query_lower:
            return code
    return None



def extract_ticker(query):
    """Smart ticker extraction with fallback mechanisms"""
    
    # Expanded ticker map with common symbols
    ticker_map = {
        # Cryptocurrencies
        'bitcoin': 'BTC', 'btc': 'BTC',
        'ethereum': 'ETH', 'eth': 'ETH',
        'tether': 'USDT', 'usdt': 'USDT',
        'bnb': 'BNB', 'binance coin': 'BNB',
        'solana': 'SOL', 'sol': 'SOL',
        
        # Stocks (DJIA components + popular tech)
        'apple': 'AAPL', 'aapl': 'AAPL',
        'microsoft': 'MSFT', 'msft': 'MSFT',
        'amazon': 'AMZN', 'amzn': 'AMZN',
        'google': 'GOOGL', 'googl': 'GOOGL',
        'tesla': 'TSLA', 'tsla': 'TSLA',
        'nvidia': 'NVDA', 'nvda': 'NVDA',
        'meta': 'META', 'meta': 'META',
        
        # Metals and Commodities
        'gold': 'XAU', 'silver': 'XAG',
        'platinum': 'XPT', 'palladium': 'XPD',
        'oil': 'CL', 'crude': 'CL',
    }
    
    # Step 1: Direct match from known names
    q_lower = query.lower()
    for keyword, symbol in ticker_map.items():
        if keyword in q_lower:
            return symbol
    
    # Step 2: Regex pattern for ticker-like symbols
    ticker_pattern = r'\b[A-Z]{1,5}\b'
    matches = re.findall(ticker_pattern, query)
    if matches:
        return max(matches, key=len)
    
    # Step 3: Extract last noun phrase
    words = query.replace('?', '').split()
    for word in reversed(words):
        if word.lower() not in {'price', 'stock', 'value', 'of'}:
            return word.upper()
    
    # Final fallback
    return "BTC" 