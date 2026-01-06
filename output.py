import os
import requests
from dotenv import load_dotenv
from src.helper import query_index, extract_ticker, get_internal_context, get_countries_currencies
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from datetime import datetime
import asyncio
import platform
import re
import warnings

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
warnings.filterwarnings('ignore')
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
import logging
logging.getLogger("accelerate").setLevel(logging.ERROR)


load_dotenv()
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_KEY_2 = os.environ.get("RAPIDAPI_KEY_2")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "finance"

RAPIDAPI_HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": None 
}



def get_forex_data(base, target):
    headers = RAPIDAPI_HEADERS.copy()
    headers["X-RapidAPI-Host"] = "alpha-vantage.p.rapidapi.com"
    try:
        response = requests.get(
            "https://alpha-vantage.p.rapidapi.com/query",
            headers=headers,
            params={
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": base.upper(),
                "to_currency": target.upper(),
                "apikey": RAPIDAPI_KEY
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        exchange_rate_data = data.get("Realtime Currency Exchange Rate", {})
        if not exchange_rate_data:
            return {"error": "No exchange rate data found"}
        rate = float(exchange_rate_data.get("5. Exchange Rate", 0))
        return {
            "base": exchange_rate_data.get("1. From_Currency Code", base.upper()),
            "target": exchange_rate_data.get("3. To_Currency Code", target.upper()),
            "rate": rate,
            "timestamp": exchange_rate_data.get("6. Last Refreshed", datetime.now().strftime("%d %b %Y"))
        }
    except Exception as e:
        return {"error": str(e)}

def get_stock_data(ticker="AAPL"):
    headers = RAPIDAPI_HEADERS.copy()
    headers["X-RapidAPI-Host"] = "alpha-vantage.p.rapidapi.com"
    try:
        response = requests.get(
            "https://alpha-vantage.p.rapidapi.com/query",
            headers=headers,
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": ticker.upper(),
                "outputsize": "compact",
                "datatype": "json"
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            return {"error": "Invalid stock data"}

        latest_date = sorted(data["Time Series (Daily)"].keys())[-1]
        latest_data = data["Time Series (Daily)"][latest_date]
        formatted_date = datetime.strptime(latest_date, "%Y-%m-%d").strftime("%d %b %Y")
        
        return {
            "symbol": ticker.upper(),
            "price": float(latest_data["4. close"]),
            "volume": int(latest_data["5. volume"]),
            "date": formatted_date
        }
    except Exception as e:
        return {"error": str(e)}

def get_google_search_results(query):
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY_2,
        "X-RapidAPI-Host": "google-search74.p.rapidapi.com"
    }
    params = {"query": query, "limit": "3", "related_keywords": "false"}
    try:
        response = requests.get("https://google-search74.p.rapidapi.com/", 
                              headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        results = data.get("results") or data.get("organic_results") or []
        
        return {
            "count": len(results),
            "results": [{
                "title": result.get("title", "No title"),
                "link": result.get("url") or result.get("link"),
                "snippet": result.get("snippet") or result.get("description")
            } for result in results],
            "query_used": query
        }
    except Exception as e:
        return {"error": f"Search error: {str(e)}"}

def get_crypto_data(symbol='BTC'):
    headers = RAPIDAPI_HEADERS.copy()
    headers["X-RapidAPI-Host"] = "coinranking1.p.rapidapi.com"
    
    try:
        search_response = requests.get(
            "https://coinranking1.p.rapidapi.com/coins",
            headers=headers,
            params={"referenceCurrencyUuid": "yhjMzLPhuIDl", "search": symbol, "limit": "1"}
        )
        search_response.raise_for_status()
        search_data = search_response.json()
        
        if not search_data['data']['coins']:
            return {"error": f"Crypto not found: {symbol}"}
            
        coin_uuid = search_data['data']['coins'][0]['uuid']
        coin_name = search_data['data']['coins'][0]['name']
        
        detail_response = requests.get(
            f"https://coinranking1.p.rapidapi.com/coin/{coin_uuid}",
            headers=headers,
            params={"referenceCurrencyUuid": "yhjMzLPhuIDl"}
        )
        detail_response.raise_for_status()
        coin_data = detail_response.json()['data']['coin']
        
        return {
            "name": coin_name,
            "symbol": coin_data['symbol'],
            "price": float(coin_data['price']),
            "marketCap": float(coin_data['marketCap']),
            "24hVolume": float(coin_data['24hVolume']),
            "lastUpdated": datetime.strptime(coin_data['lastUpdated'], 
                          "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%d %b %Y") if 'lastUpdated' in coin_data else 'N/A'
        }
    except Exception as e:
        return {"error": str(e)}

def determine_api_calls(query):
    responses = {}
    q_lower = query.lower()

    # Stock
    if any(k in q_lower for k in ["stock", "share", "price", "market", "ticker"]) and \
       not any(k in q_lower for k in ["bitcoin", "ethereum", "btc", "eth", "crypto"]):
        ticker = extract_ticker(query)
        stock_data = get_stock_data(ticker)
        if "error" not in stock_data:
            responses["stock_data"] = stock_data

    # Crypto
    if any(k in q_lower for k in ["crypto", "bitcoin", "ethereum", "btc", "eth", "coin"]):
        crypto_symbol = extract_ticker(query)
        crypto_data = get_crypto_data(crypto_symbol)
        if "error" not in crypto_data:
            responses["crypto"] = crypto_data

    # Forex
    if any(k in q_lower for k in ["forex", "currency", "exchange rate", "usd", "inr", "eur"]):
        base_code = target_code = None
        m = re.search(r'\b([A-Za-z]{3})/([A-Za-z]{3})\b', query)
        if m:
            base_code, target_code = m.group(1).upper(), m.group(2).upper()
        else:
            m2 = re.search(r'\b([A-Za-z]{3})\b\s+(?:to|in)\s+\b([A-Za-z]{3})\b', query, re.IGNORECASE)
            if m2:
                base_code, target_code = m2.group(1).upper(), m2.group(2).upper()
        
        if not (base_code and target_code):
            codes = get_countries_currencies(query, max_countries=2)
            if len(codes) >= 2:
                base_code, target_code = codes[0], codes[1]
            else:
                base_code, target_code = "USD", "INR"
        
        forex_data = get_forex_data(base_code, target_code)
        if "error" not in forex_data:
            responses["forex"] = forex_data

    # Google Search
    google_data = get_google_search_results(query)
    if "error" not in google_data:
        responses["google_search"] = google_data

    return responses


MODEL_CACHE_DIR = os.path.join(os.getcwd(), "model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def load_model():
    MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
    try:
        print(f"Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE_DIR,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=MODEL_CACHE_DIR,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        gen_pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        print("✅ Model loaded successfully")
        return gen_pipeline
    except Exception as e:
        print(f"❌ Model load error: {str(e)}")
        print("Trying fallback model...")
        try:
            MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, 
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=MODEL_CACHE_DIR
            )
            gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
            print("✅ Fallback model loaded")
            return gen_pipeline
        except:
            print("❌ All models failed to load")
            return None

generator = load_model()



def build_prompt(query, api_responses):
    parts = [f"Date: {datetime.now().strftime('%d %b %Y')}\n"]
    
    # Stock data
    if "stock_data" in api_responses:
        s = api_responses["stock_data"]
        parts.append(f"Stock {s['symbol']}: ${s['price']:.2f} (Vol: {s['volume']:,})")
    
    # Crypto data
    if "crypto" in api_responses:
        c = api_responses["crypto"]
        parts.append(f"Crypto {c['symbol']}: ${c['price']:,.2f} (MCap: ${c['marketCap']:,.0f})")
    
    # Forex data
    if "forex" in api_responses:
        f = api_responses["forex"]
        parts.append(f"Forex {f['base']}/{f['target']}: {f['rate']:.4f}")
    
    # News
    if "google_search" in api_responses:
        news = api_responses["google_search"]["results"][:2]
        if news:
            parts.append("Recent news:")
            for i, n in enumerate(news, 1):
                parts.append(f"{i}. {n['title']}")
    
    prompt = "\n".join(parts)
    prompt += f"\n\nQuestion: {query}\nAnswer:"
    return prompt



def generate_final_answer(query, index_name):
    try:
        if generator is None:
            return {
                "answer": "❌ Model not loaded. Check transformers installation.",
                "api_data": {},
                "internal_context": "",
                "error": "Model unavailable"
            }

        print("Processing...")
        api_responses = determine_api_calls(query)
        prompt = build_prompt(query, api_responses)
        output = generator(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=generator.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        answer = output[0]['generated_text'].strip()
        answer = answer.split("<|end")[0].split("<|assistant")[0].strip()
        refs = []
        if "google_search" in api_responses:
            refs.extend([r['link'] for r in api_responses["google_search"]["results"][:2]])
        
        if refs:
            answer += "\n\nSources:\n" + "\n".join([f"• {r}" for r in refs])

        internal_context = ""
        if PINECONE_API_KEY:
            try:
                internal_context = get_internal_context(query, index_name)
            except Exception as e:
                print(f"Context retrieval skipped: {str(e)}")
        
        return {
            "answer": answer,
            "api_data": api_responses,
            "internal_context": internal_context,
            "error": None
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "answer": "Processing error. Please retry.",
            "internal_context": "",
            "error": str(e)
        }

if __name__ == "__main__":
    print("\n💰 Financial Assistant\n")
    
    while True:
        try:
            query = input("📩 Query: ").strip()
            if not query or query.lower() in ['exit', 'quit']:
                break

            result = generate_final_answer(query, INDEX_NAME)
            print(f"\n{result['answer']}\n")
            print("─" * 50)

            if result.get("internal_context"):
                print(f"\n📚 Context: {result['internal_context'][:200]}...\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("Session ended.")