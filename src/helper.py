import os
import pinecone
from typing import List
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
import re
import pycountry
from countryinfo import CountryInfo
from babel.numbers import get_territory_currencies

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


def extract_ticker(query):
    """Smart ticker extraction with fallback mechanisms"""

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
    return 'BTC'


def get_internal_context(query, index_name):
    """
    Retrieve and process internal context
    """
    try:
        internal_results = query_index(query, index_name)
        if not internal_results:
            return ""
        try:
            if hasattr(internal_results[0], 'content'):
                documents = [doc.content for doc in internal_results]
            elif hasattr(internal_results[0], 'text'):
                documents = [doc.text for doc in internal_results]
            elif hasattr(internal_results[0], 'page_content'):
                documents = [doc.page_content for doc in internal_results]
            else:
                documents = [str(doc) for doc in internal_results]
        except Exception as e:
            print(f"Error extracting document content: {str(e)}")
            return ""

        combined_text = " ".join(documents)
        sentences = combined_text.replace('\n', ' ').split('. ')
        
        def sentence_relevance_score(sentence, query):
            query_words = set(query.lower().split())
            sentence_words = set(sentence.lower().split())
            exact_phrase_score = 10 if any(phrase in sentence.lower() for phrase in [
                'financial', 'investment', 'market', 'stock', 'economy', 
                'trading', 'finance', 'economic', 'company', 'business'
            ]) else 0
            keyword_score = len(query_words.intersection(sentence_words))
            return exact_phrase_score + keyword_score
        
        scored_sentences = [
            (sentence.strip(), sentence_relevance_score(sentence, query)) 
            for sentence in sentences 
            if sentence.strip() and len(sentence.strip()) > 30
        ]
        sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
        
        max_context_length = 1200 
        current_length = 0
        selected_sentences = []
        
        for sentence, score in sorted_sentences:
            if current_length + len(sentence) <= max_context_length:
                selected_sentences.append(sentence)
                current_length += len(sentence)
            else:
                break

        context = '. '.join(selected_sentences)
        if context and not context.endswith('.'):
             context += '.'
        return context
    except Exception as e:
        print(f"Error in get_internal_context: {str(e)}")
        return ""


def get_countries_currencies(text: str, max_countries: int = 2) -> List[str]:

    results: List[str] = []
    seen_alpha2: set[str] = set()
    text_lower = text.lower()

    #  Build (alias, country, kind) list
    alias_country = []
    for country in pycountry.countries:
        # collect name‑based aliases
        names = {country.name}
        if hasattr(country, 'official_name'):
            names.add(country.official_name)
        if hasattr(country, 'common_name'):
            names.add(country.common_name)
        for name in names:
            alias_country.append((name.lower(), country, 'name'))

        alias_country.append((country.alpha_2, country, 'code'))
        alias_country.append((country.alpha_3, country, 'code'))

    #  Match longer aliases first
    alias_country.sort(key=lambda x: len(x[0]), reverse=True)

    #  Exact‐alias pass
    for alias, country, kind in alias_country:
        if len(results) >= max_countries:
            break
        if country.alpha_2 in seen_alpha2:
            continue

        hay = text_lower if kind == 'name' else text
        if not re.search(rf'\b{re.escape(alias)}\b', hay):
            continue

        # get currency via Babel CLDR
        code = None
        try:
            codes = get_territory_currencies(country.alpha_2)
            if codes:
                code = codes[0]
        except Exception:
            pass

        # fallback to CountryInfo
        if not code:
            try:
                ci = CountryInfo(country.name)
                currs = ci.currencies()
                if currs:
                    code = currs[0]
            except Exception:
                pass

        if code:
            results.append(code)
            seen_alpha2.add(country.alpha_2)

    # 4) Fuzzy‐demonym pass
    if len(results) < max_countries:
        tokens = re.findall(r'\b[A-Za-z]{4,}\b', text)
        for token in tokens:
            if len(results) >= max_countries:
                break
            try:
                country = pycountry.countries.search_fuzzy(token)[0]
            except LookupError:
                continue
            if country.alpha_2 in seen_alpha2:
                continue

            # same currency lookup
            code = None
            try:
                codes = get_territory_currencies(country.alpha_2)
                if codes:
                    code = codes[0]
            except Exception:
                pass

            if not code:
                try:
                    ci = CountryInfo(country.name)
                    currs = ci.currencies()
                    if currs:
                        code = currs[0]
                except Exception:
                    pass

            if code:
                results.append(code)
                seen_alpha2.add(country.alpha_2)

    return results


