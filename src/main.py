import os
import requests
from dotenv import load_dotenv
from src.helper import query_index
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,BitsAndBytesConfig
import torch
from datetime import datetime

load_dotenv()
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
INDEX_NAME = "finance"
SERPAPI_KEY=os.environ.get("SERPAPI_KEY")
METALAPI_KEY=os.environ.get("METALAPI_KEY")

import asyncio
import platform
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ------------------------------------------------------------------------------------------------------------------------------------------------

RAPIDAPI_HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": None 
}

# ------------------------------------------------------------------------------------------------------------------------------------------------
def get_forex_data(currency_pair="EUR-USD"):
    """Get Forex data from forex-api2"""
    headers = RAPIDAPI_HEADERS.copy()
    headers["X-RapidAPI-Host"] = "forex-api2.p.rapidapi.com"
    
    try:
        response = requests.get(
            "https://forex-api2.p.rapidapi.com/currencies",
            headers=headers,
            params={"pair": currency_pair}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_yahoo_finance(ticker="AAPL"):
    """Get stock data from yahoo-finance166 API"""
    headers = RAPIDAPI_HEADERS.copy()
    headers["X-RapidAPI-Host"] = "yahoo-finance166.p.rapidapi.com"
    
    try:
        response = requests.get(
            f"https://yahoo-finance166.p.rapidapi.com/stock/v2/get-summary",
            headers=headers,
            params={"symbol": ticker, "region": "US"}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_wsj_news(search_query="markets"):
    """Get news from Wall Street Journal API"""
    headers = RAPIDAPI_HEADERS.copy()
    headers["X-RapidAPI-Host"] = "wall-street-journal.p.rapidapi.com"
    
    try:
        response = requests.get(
            "https://wall-street-journal.p.rapidapi.com/search",
            headers=headers,
            params={"query": search_query, "limit": "5"}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_live_metal_prices(metal="gold"):
    """Get metal prices from live-metal-prices API"""
    headers = RAPIDAPI_HEADERS.copy()
    headers["X-RapidAPI-Host"] = "live-metal-prices.p.rapidapi.com"
    
    try:
        response = requests.get(
            "https://live-metal-prices.p.rapidapi.com/v1/latest/XAU,PA,XAG,PL",
            headers=headers,
            params={"currency": "USD"}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}
    
def get_coinranking(query):
    """
    Fetch detailed cryptocurrency data from Coinranking using the coin search endpoint.
    """
    url = "https://coinranking1.p.rapidapi.com/coins"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
    }
    params = {
        "referenceCurrencyUuid": "yhjMzLPhuIDl",
        "search": query,
        "limit": "1"
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        coins = data.get("data", {}).get("coins", [])
        if not coins:
            return {"error": f"No data found for {query}"}
        coin = coins[0]
        return {
            "name": coin.get("name", "Unknown"),
            "symbol": coin.get("symbol", "N/A"),
            "price": float(coin.get("price", 0)),
            "marketCap": float(coin.get("marketCap", 0)),
            "24hVolume": float(coin.get("24hVolume", 0)),
            "allTimeHigh": float(coin.get("allTimeHigh", {}).get("price", 0)),
            "lastUpdated": coin.get("lastUpdated", "N/A")
        }
    except Exception as e:
        print(f"Coinranking Error: {str(e)}")
        return {"error": str(e)}


def get_crypto_data(symbol='BTC'):
    """Get crypto data from Coinranking API for any cryptocurrency"""
    headers = RAPIDAPI_HEADERS.copy()
    headers["X-RapidAPI-Host"] = "coinranking1.p.rapidapi.com"
    
    try:
        # Step 1: Search for coin UUID using symbol/name
        search_params = {
            "referenceCurrencyUuid": "yhjMzLPhuIDl",  # USD
            "search": symbol,
            "limit": "1"
        }
        
        search_response = requests.get(
            "https://coinranking1.p.rapidapi.com/coins",
            headers=headers,
            params=search_params
        )
        search_response.raise_for_status()
        search_data = search_response.json()
        
        # Check if any coins found
        if not search_data['data']['coins']:
            return {"error": f"No cryptocurrency found for symbol: {symbol}"}
            
        coin_uuid = search_data['data']['coins'][0]['uuid']
        coin_name = search_data['data']['coins'][0]['name']
        
        # Step 2: Get detailed coin data using UUID
        detail_response = requests.get(
            f"https://coinranking1.p.rapidapi.com/coin/{coin_uuid}",
            headers=headers,
            params={"referenceCurrencyUuid": "yhjMzLPhuIDl"}
        )
        detail_response.raise_for_status()
        detail_data = detail_response.json()
        
        # Extract relevant information
        coin_data = detail_data['data']['coin']
        return {
            "name": coin_name,
            "symbol": coin_data['symbol'],
            "price": float(coin_data['price']),
            "marketCap": float(coin_data['marketCap']),
            "24hVolume": float(coin_data['24hVolume']),
            "btcPrice": float(coin_data['btcPrice']),
            "allTimeHigh": float(coin_data['allTimeHigh']['price']),
            "websiteUrl": coin_data['websiteUrl'],
            "lastUpdated": coin_data.get('lastUpdated', 'N/A')
        }
        
    except requests.HTTPError as e:
        return {"error": f"API Error: {str(e)}"}
    except KeyError as e:
        return {"error": f"Data parsing error: {str(e)}"}
    except Exception as e:
        return {"error": f"General error: {str(e)}"}

# ----------------------------------------------------------------------------------------------------------------------------------------------

def extract_ticker(query):
    """Smart ticker extraction with fallback mechanisms"""
    import re
    
    # Expanded ticker map with common symbols
    ticker_map = {
        # Cryptocurrencies (Top 20)
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

# ------------------------------------------------------------------------------------------------------------------------------------------------

def determine_api_calls(query):
    """Route queries to appropriate APIs based on content with improved error handling and debugging"""
    responses = {}
    api_status = {}
    q_lower = query.lower()

    if not os.getenv("RAPIDAPI_KEY"):
        print("ERROR: Missing RapidAPI key in .env file")
        return {"_api_status": "error", "error": "Missing RapidAPI key in .env file"}
    
    try:
        rapidapi_key = os.getenv("RAPIDAPI_KEY").strip()
        print(f"Using RapidAPI key: {rapidapi_key[:5]}...{rapidapi_key[-5:] if len(rapidapi_key) > 10 else ''}")
    except:
        print("WARNING: Unable to print RapidAPI key - may be None or invalid format")
    
    # Forex data
    if any(k in q_lower for k in ["forex", "currency", "exchange rate", "euro", "dollar", "yen", "pound"]):
        print(f"Attempting to fetch forex data...")
        try:
            forex_data = get_forex_data()
            if isinstance(forex_data, dict) and "error" in forex_data:
                api_status["forex"] = f"Error: {forex_data['error']}"
                print(f"Forex API error: {forex_data['error']}")
            else:
                responses["forex"] = forex_data
                api_status["forex"] = "Success"
                print(f"Forex data retrieved successfully")
        except Exception as e:
            err_msg = f"Forex API exception: {str(e)}"
            api_status["forex"] = f"Exception: {str(e)}"
            print(err_msg)
            responses["forex"] = {"error": err_msg}

    # Stock data
    if any(k in q_lower for k in ["stock", "share", "equity", "price", "market", "ticker"]) and not any(k in q_lower for k in ["bitcoin", "ethereum", "btc", "eth", "crypto"]):

        ticker = extract_ticker(query)
        print(f"Attempting to fetch stock data for ticker: {ticker}")
        try:
            if ticker not in ["BTC", "ETH"]:

                stock_data = get_yahoo_finance(ticker)

                stock_data = get_yahoo_finance(ticker)

                if isinstance(stock_data, dict) and "error" in stock_data:
                    api_status["yahoo_finance"] = f"Error: {stock_data['error']}"
                    print(f"Yahoo Finance API error: {stock_data['error']}")
                else:
                    responses["yahoo_finance"] = stock_data
                    api_status["yahoo_finance"] = "Success"
                    print(f"Stock data for {ticker} retrieved successfully")
        except Exception as e:
            err_msg = f"Yahoo Finance API exception: {str(e)}"
            api_status["yahoo_finance"] = f"Exception: {str(e)}"
            print(err_msg)
            responses["yahoo_finance"] = {"error": err_msg}

    # News data
    if any(k in q_lower for k in ["news", "update", "journal", "headlines", "article"]):
        print(f"Attempting to fetch WSJ news with query: {query}")
        try:
            news_data = get_wsj_news(query)
            if isinstance(news_data, dict) and "error" in news_data:
                api_status["wsj_news"] = f"Error: {news_data['error']}"
                print(f"WSJ News API error: {news_data['error']}")
            else:
                responses["wsj_news"] = news_data
                api_status["wsj_news"] = "Success"
                print(f"WSJ news retrieved successfully")
        except Exception as e:
            err_msg = f"WSJ News API exception: {str(e)}"
            api_status["wsj_news"] = f"Exception: {str(e)}"
            print(err_msg)
            responses["wsj_news"] = {"error": err_msg}

    # Metal prices
    if any(k in q_lower for k in ["metal", "gold", "silver", "platinum", "palladium", "precious metal"]):
        print(f"Attempting to fetch metal prices...")
        try:
            metal_data = get_live_metal_prices()
            if isinstance(metal_data, dict) and "error" in metal_data:
                api_status["metal_prices"] = f"Error: {metal_data['error']}"
                print(f"Metal Prices API error: {metal_data['error']}")
            else:
                responses["metal_prices"] = metal_data
                api_status["metal_prices"] = "Success"
                print(f"Metal prices retrieved successfully")
        except Exception as e:
            err_msg = f"Metal Prices API exception: {str(e)}"
            api_status["metal_prices"] = f"Exception: {str(e)}"
            print(err_msg)
            responses["metal_prices"] = {"error": err_msg}

    # Crypto data
    if any(k in q_lower for k in ["crypto", "bitcoin", "ethereum", "btc", "eth", "coin", "blockchain"]):
        crypto_symbol = extract_ticker(query) if "extract_ticker" in locals() else "BTC"
        print(f"Attempting to fetch crypto data for: {crypto_symbol}")
        try:
            crypto_data = get_crypto_data(crypto_symbol)
            if isinstance(crypto_data, dict) and "error" in crypto_data:
                api_status["crypto"] = f"Error: {crypto_data['error']}"
                print(f"Crypto API error: {crypto_data['error']}")
            else:
                responses["crypto"] = crypto_data
                api_status["crypto"] = "Success"
                print(f"Crypto data for {crypto_symbol} retrieved successfully")
        except Exception as e:
            err_msg = f"Crypto API exception: {str(e)}"
            api_status["crypto"] = f"Exception: {str(e)}"
            print(err_msg)
            responses["crypto"] = {"error": err_msg}

    responses["_api_status"] = api_status
    
    if len(responses) <= 1:
        print("WARNING: No successful API calls were made for this query")
        responses["error"] = "No financial data sources could be accessed for this query"
    
    print(f"API calls completed with status: {api_status}")
    return responses
# ------------------------------------------------------------------------------------------------------------------------------------------------

def get_internal_context(query, index_name):

    try:
        internal_results = query_index(query, index_name)
        if internal_results:
            # Check what attributes the results have
            if hasattr(internal_results[0], 'content'):
                context = " ".join([doc.content for doc in internal_results])
            elif hasattr(internal_results[0], 'text'):
                context = " ".join([doc.text for doc in internal_results])
            elif hasattr(internal_results[0], 'page_content'):
                context = " ".join([doc.page_content for doc in internal_results])
            else:
                context = str(internal_results)
        else:
            context = ""
        return context
    except Exception as e:
        print(f"Error in get_internal_context: {str(e)}")
        return ""

# ------------------------------------------------------------------------------------------------------------------------------------------------

def load_model():
    MODEL_NAME = "microsoft/Phi-4-mini-instruct"
    try:

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Basic config for model loading
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True
        )
        
   
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        
        return pipe
    except Exception as e:
        detailed_error = f"MODEL LOAD ERROR DETAILS: {str(e)}"
        print(detailed_error)
        return None

generator = load_model()


# ------------------------------------------------------------------------------------------------------------------------------------------------

def build_prompt(query, index_name):
    """Build a prompt with better context handling for Phi-4"""
    try:
        internal_context = get_internal_context(query, index_name)
        api_responses = determine_api_calls(query)

        cleaned_responses = []
        for api_name, response in api_responses.items():
            if api_name == "_api_status" or api_name == "error":
                continue 
                
            if isinstance(response, dict):
                try:
                    # Yahoo Finance formatting
                    if api_name == "yahoo_finance" and "price" in response.get('data', {}):
                        data = response['data']
                        cleaned_responses.append(
                            f"Stock Data ({data.get('symbol', 'N/A')}):\n"
                            f"Name: {data.get('shortName', 'N/A')}\n"
                            f"Price: {data.get('regularMarketPrice', {}).get('fmt', 'N/A')}\n"
                            f"Change: {data.get('regularMarketChangePercent', {}).get('fmt', 'N/A')}\n"
                            f"Market Cap: {data.get('marketCap', {}).get('fmt', 'N/A')}"
                        )

                    # WSJ News formatting
                    elif api_name == "wsj_news" and "articles" in response:
                        news_items = []
                        for item in response.get("articles", [])[:3]:
                            news_items.append(
                                f"- {item.get('title', 'No title')} "
                                f"({item.get('source', {}).get('name', 'Unknown source')})"
                            )
                        cleaned_responses.append("Latest Financial News:\n" + "\n".join(news_items))

                    # Metal Prices formatting
                    elif api_name == "metal_prices" and "data" in response:
                        metals = []
                        for metal in response.get('data', []):
                            metals.append(
                                f"{metal.get('metal', 'Unknown')}: "
                                f"${metal.get('price', 0):.2f}/oz"
                            )
                        cleaned_responses.append("Metal Prices:\n" + "\n".join(metals))

                    # Crypto Data formatting
                    elif api_name == "crypto" and "price" in response:
                        if not all(key in response for key in ["name", "symbol"]):
                            cleaned_responses.append("Crypto Data: Incomplete API response")
                            continue
                        cleaned_responses.append(
                            f"Crypto Data ({response.get('name', 'Unknown')}):\n"
                            f"Symbol: {response.get('symbol', 'N/A')}\n"
                            f"Price: ${response['price']:,.2f}\n"
                            f"Market Cap: ${response.get('marketCap', 0):,.0f}\n"
                            f"24h Volume: ${response.get('24hVolume', 0):,.0f}\n"
                            f"All Time High: ${response.get('allTimeHigh', 0):,.2f}"
                        )

                    # Forex Data formatting
                    elif api_name == "forex" and "data" in response:
                        forex_data = []
                        for pair in response.get('data', []):
                            forex_data.append(
                                f"{pair.get('pair', 'N/A')}: "
                                f"{pair.get('rate', 'N/A')}"
                            )
                        cleaned_responses.append("Forex Rates:\n" + "\n".join(forex_data))

                    # Fallback formatting for other APIs
                    else:
                        important_keys = ['price', 'rate', 'value', 'name', 'symbol']
                        relevant_data = []
                        for k, v in response.items():
                            if k in important_keys and not isinstance(v, (dict, list)):
                                relevant_data.append(f"{k}: {v}")
                        
                        if relevant_data:
                            cleaned_responses.append(f"{api_name}:\n" + "\n".join(relevant_data))
                        else:
                            cleaned_responses.append(f"{api_name}: Data unavailable")

                except Exception as e:
                    print(f"Error formatting {api_name} response: {str(e)}")
                    cleaned_responses.append(f"{api_name}: Error processing data")

            else:
                cleaned_responses.append(f"{api_name}: {str(response)[:300]}")

        prompt_template = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are a financial expert assistant providing detailed, comprehensive answers. "
            f"Always analyze all available data and provide thorough explanations with specific insights. "
            f"Include market analysis, trends, and actionable recommendations when appropriate. "
            f"Your answers should be detailed, complete, and directly address the user's question.\n\n"
            
            f"Internal Context Information:\n{internal_context[:1500]}\n\n"
            
            f"External Data Sources:\n" + "\n".join(cleaned_responses) + "\n\n"
            
            f"Guidelines:\n"
            f"1. Start with a concise summary answering the query directly\n"
            f"2. Use exact numbers from data when available\n"
            f"3. Analyze trends using historical context\n"
            f"4. Provide risk assessment and recommendations\n"
            f"5. Mention data limitations if any\n"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{query}\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
        
        return prompt_template
    
    except Exception as e:
        print(f"Prompt building error: {str(e)}")
        return f"System Error: Failed to build prompt. Original query: {query}"
# ------------------------------------------------------------------------------------------------------------------------------------------------

def generate_final_answer(query, index_name):
    if not generator:
        return {
            "error": "Model not loaded properly",
            "internal_context": "",
            "api_data": {}
        }
    
    try:
        internal_context = get_internal_context(query, index_name)
        api_responses = determine_api_calls(query)
        
        prompt = build_prompt(query, index_name)

        print(f"\n{'='*40} DEBUG PROMPT {'='*40}\n{prompt}\n{'='*94}\n")

        output = generator(
            prompt,
            do_sample=True,
            temperature=0.5,
            max_length=2500,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=generator.tokenizer.eos_token_id,
            pad_token_id=generator.tokenizer.eos_token_id,
            truncation=True,
            num_return_sequences=1
        )

        full_response = output[0]['generated_text']
        print(f"\n{'='*40} RAW MODEL OUTPUT {'='*40}\n{full_response}\n{'='*94}\n")

        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            answer_text = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            clean_answer = answer_text.split("<|")[0].strip()
        else:
            clean_answer = full_response.replace(prompt, "").strip()

        clean_answer = "\n".join([line.strip() for line in clean_answer.split("\n") if line.strip()])

        return {
            "answer": clean_answer,
            "internal_context": internal_context,
            "api_data": api_responses,
            "error": None
        }
            
    except Exception as e:
        return {
            "error": str(e),
            "internal_context": "",
            "api_data": {}
        }
