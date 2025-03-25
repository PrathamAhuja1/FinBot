import os
import requests
from dotenv import load_dotenv
from src.helper import query_index
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,BitsAndBytesConfig
import torch
from datetime import datetime
from src.helper import extract_country
from src.helper import extract_ticker
from src.helper import get_country_name
import re



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

def get_forex_data(base="USD", target="INR"):
    """
    Returns the current exchange rate from base to target currency.
    """
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
            "timestamp": exchange_rate_data.get("6. Last Refreshed", datetime.now().isoformat())
        }
    except Exception as e:
        return {"error": str(e)}



def get_stock_data(ticker="AAPL"):
    """
    Get stock data from Alphavantage API via RapidAPI.
    Returns the latest closing price and volume.
    """
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
            return {"error": "Invalid stock data from Alphavantage"}

        latest_date = sorted(data["Time Series (Daily)"].keys())[-1]
        latest_data = data["Time Series (Daily)"][latest_date]
        
        return {
            "symbol": ticker.upper(),
            "price": float(latest_data["4. close"]),
            "volume": int(latest_data["5. volume"]),
            "date": latest_date
        }
    except Exception as e:
        return {"error": str(e)}



def get_google_search_results(query):
    """Fetches search results from the Google Search API"""
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "google-search74.p.rapidapi.com"
    }
    params = {
        "query": query,
        "limit": "5",
        "related_keywords": "false"
    }
    try:
        response = requests.get(
            "https://google-search74.p.rapidapi.com/",
            headers=headers,
            params=params,
            timeout=15
        )
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
    except requests.HTTPError as e:
        return {"error": f"Google Search Error {e.response.status_code}: {str(e)}"}
    except Exception as e:
        return {"error": f"Google Search Error: {str(e)}"}



def get_metal_rate(metal="gold", currency="USD"):
    """Get metal prices using live-gold-prices API"""
    url = "https://live-gold-prices.p.rapidapi.com/latest"
    headers = RAPIDAPI_HEADERS.copy()
    headers["X-RapidAPI-Host"] = "live-gold-prices.p.rapidapi.com"
    
    try:
        response = requests.get(
            url,
            headers=headers,
            params={
                "metal": metal.upper(),
                "currency": currency.upper()
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get("statusCode") != 200:
            return {"error": data.get("message", "Unknown error from API")}
        price_per_gram = float(data["pricePerGram"])
        
        return {
            "metal": metal.capitalize(),
            "currency": currency.upper(),
            "rate": round(price_per_gram, 4),
            "unit": "per gram",
            "timestamp": data.get("timestamp", datetime.now().isoformat())
        }
    except requests.HTTPError as e:
        return {"error": f"API Error {e.response.status_code}: {str(e)}"}
    except Exception as e:
        return {"error": f"General error: {str(e)}"}



def get_crypto_data(symbol='BTC'):
    """Get crypto data from Coinranking API for any cryptocurrency"""
    headers = RAPIDAPI_HEADERS.copy()
    headers["X-RapidAPI-Host"] = "coinranking1.p.rapidapi.com"
    
    try:
        # Step 1: Search for coin UUID using symbol/name
        search_params = {
            "referenceCurrencyUuid": "yhjMzLPhuIDl",
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


# ------------------------------------------------------------------------------------------------------------------------------------------------

def determine_api_calls(query):
    responses = {}
    api_status = {}
    q_lower = query.lower()

    # Stock data
    if any(k in q_lower for k in ["stock", "share", "price", "market", "ticker"]) and not any(k in q_lower for k in ["bitcoin", "ethereum", "btc", "eth", "crypto"]):
        ticker = extract_ticker(query)
        print(f"Attempting to fetch stock data for ticker: {ticker}")
        try:
            stock_data = get_stock_data(ticker)
            if "error" in stock_data:
                api_status["stock_data"] = f"Error: {stock_data['error']}"
                print(f"Stock API error: {stock_data['error']}")
            else:
                responses["stock_data"] = stock_data
                api_status["stock_data"] = "Success"
                print(f"Stock data for {ticker} retrieved successfully")
        except Exception as e:
            err_msg = f"Stock API exception: {str(e)}"
            api_status["stock_data"] = f"Exception: {str(e)}"
            print(err_msg)
            responses["stock_data"] = {"error": err_msg}



    # Metal prices
    if any(k in q_lower for k in ["metal", "gold", "silver", "platinum", "palladium", "precious"]):
        print("Attempting to fetch metal prices...")
        try:
            metal_types = ['gold', 'silver', 'platinum', 'palladium']
            currency_codes = ['USD', 'INR', 'EUR', 'GBP']
            
            metal = next((m for m in metal_types if m in q_lower), 'gold')
            currency = next((c for c in currency_codes if c.lower() in q_lower), 'USD')
            
            metal_data = get_metal_rate(metal, currency)
            if "error" in metal_data:
                api_status["metal_prices"] = f"Error: {metal_data['error']}"
            else:
                responses["metal_prices"] = metal_data
                api_status["metal_prices"] = "Success"
        except Exception as e:
            api_status["metal_prices"] = f"Exception: {str(e)}"
            responses["metal_prices"] = {"error": str(e)}

    # Crypto data
    if any(k in q_lower for k in ["crypto", "bitcoin", "ethereum", "btc", "eth", "coin", "blockchain"]):
        crypto_symbol = extract_ticker(query)
        print(f"Fetching crypto data for: {crypto_symbol}")
        crypto_data = get_crypto_data(crypto_symbol)
        if isinstance(crypto_data, dict) and "error" in crypto_data:
            api_status["crypto"] = f"Error: {crypto_data['error']}"
        else:
            responses["crypto"] = crypto_data
            api_status["crypto"] = "Success"

    # Forex data
    if any(k in q_lower for k in ["forex", "currency", "exchange rate"]):
        print("Attempting to fetch forex data...")
        forex_data = get_forex_data("USD", "INR")
        if "error" in forex_data:
            api_status["forex"] = f"Error: {forex_data['error']}"
        else:
            responses["forex"] = forex_data
            api_status["forex"] = "Success"        



    responses["_api_status"] = api_status


    # Google Search
    if len(responses) <= 1:
        print("Executing Google Search API.")
        google_data = get_google_search_results(query)
        if "error" in google_data:
            api_status["google_search"] = google_data["error"]
        else:
            responses["google_search"] = google_data
            api_status["google_search"] = "Success"

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
        print(f"Loading model: {MODEL_NAME}")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        try:
            first_param_device = next(model.parameters()).device
            print(f"First parameter device: {first_param_device}")
        except Exception as e:
            print(f"Unable to check parameter device: {str(e)}")

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
    except Exception as e:
        print(f"MODEL LOAD ERROR: {str(e)}")
        return None

generator = load_model()


# ------------------------------------------------------------------------------------------------------------------------------------------------

def build_prompt(query, index_name, api_responses=None):
    """Build a prompt with proper context handling for Phi-4 """
    try:
        internal_context = get_internal_context(query, index_name)

        if api_responses is None:
            api_responses = determine_api_calls(query)

        cleaned_responses = []
        for api_name, response in api_responses.items():
            if api_name == "_api_status" or api_name == "error":
                continue 
                
            if isinstance(response, dict):
                try:

                    # Alpha Vantage Stock Data formatting
                    if api_name == "stock_data" and "price" in response:
                        cleaned_responses.append(
                            f"Stock Data ({response.get('symbol', 'N/A')}):\n"
                            f"Closing Price: {response.get('price', 0):.2f} USD\n"
                            f"Volume: {response.get('volume', 0):,}\n"
                            f"Last Updated: {response.get('date', 'N/A')}"
                        )


                    # Alpha Vantage Forex Data formatting
                    elif api_name == "forex" and "rate" in response:
                        base_currency = response.get('base', 'Unknown')
                        target_currency = response.get('target', 'Unknown')
                        exchange_rate = response.get('rate', 0)

                        cleaned_responses.append(
                            f"Forex ({base_currency}-{target_currency}):\n"
                            f"Exchange Rate: {exchange_rate:.4f}\n"
                            f"Last Updated: {response.get('timestamp', 'N/A')}"
                        )


                    # Metal Prices formatting
                    elif api_name == "metal_prices":
                        cleaned_responses.append(
                            f"Metal Prices:\n"
                            f"{response.get('metal', 'Unknown')}: "
                            f"{response.get('rate', 0):.4f} {response.get('currency', 'USD')} "
                            f"{response.get('unit', 'per gram')}"
                        )


                    # Crypto Data formatting
                    elif api_name == "crypto" and "price" in response:
                        cleaned_responses.append(
                            f"Crypto Data ({response.get('name', 'Unknown')}):\n"
                            f"Symbol: {response.get('symbol', 'N/A')}\n"
                            f"Price: ${response['price']:,.2f}\n"
                            f"Market Cap: ${response.get('marketCap', 0):,.0f}\n"
                            f"24h Volume: ${response.get('24hVolume', 0):,.0f}\n"
                            f"All Time High: ${response.get('allTimeHigh', 0):,.2f}"
                        )


                    # Google Search formatting
                    elif api_name == "google_search":
                        if "error" in response:
                            cleaned_responses.append(f"Google Search Error: {response['error']}")
                        else:
                            items = response.get("results", [])
                            if items:
                                search_items = [
                                    f"- {item.get('title', 'No title')}\n  {item.get('link')}"
                                    for item in items[:3]
                                ]
                                cleaned_responses.append("Web Search Results:\n" + "\n".join(search_items))
                            else:
                                cleaned_responses.append("Web Search: No relevant results found")

                             
                                
                    else:
                        important_keys = ['price', 'rate', 'value', 'name', 'symbol']
                        relevant_data = [
                            f"{k}: {v}" for k, v in response.items() 
                            if k in important_keys and not isinstance(v, (dict, list))
                        ]
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
            "answer": "Model not loaded properly. Please check server logs.",
            "error": "Model not loaded properly",
            "internal_context": "",
            "api_data": {},
            "raw_output": ""
        }
    
    try:

        internal_context = get_internal_context(query, index_name)

        api_responses = determine_api_calls(query)
        
        prompt = build_prompt(query, index_name, api_responses)

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
            "raw_output": full_response,
            "error": None
        }
            
    except Exception as e:
        return {
            "answer": f"An error occurred: {str(e)}",
            "error": str(e),
            "internal_context": "",
            "api_data": {},
            "raw_output": ""
        }

# -------------------------------------------------------------------------------------------------------------------------------#