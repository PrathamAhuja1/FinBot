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
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables
load_dotenv()
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
INDEX_NAME = "finance"
RAPIDAPI_KEY_2 = os.environ.get("RAPIDAPI_KEY_2")

RAPIDAPI_HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": None 
}

# -----------------------------------------------------------------------------------------------------------------------------------------------

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
            return {"error": "Invalid stock data from Alphavantage"}

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
    params = {
        "query": query,
        "limit": "3",
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


def get_crypto_data(symbol='BTC'):
    headers = RAPIDAPI_HEADERS.copy()
    headers["X-RapidAPI-Host"] = "coinranking1.p.rapidapi.com"
    
    try:
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
        
        if not search_data['data']['coins']:
            return {"error": f"No cryptocurrency found for symbol: {symbol}"}
            
        coin_uuid = search_data['data']['coins'][0]['uuid']
        coin_name = search_data['data']['coins'][0]['name']
        
        detail_response = requests.get(
            f"https://coinranking1.p.rapidapi.com/coin/{coin_uuid}",
            headers=headers,
            params={"referenceCurrencyUuid": "yhjMzLPhuIDl"}
        )
        detail_response.raise_for_status()
        detail_data = detail_response.json()
        
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
            "lastUpdated": datetime.strptime(coin_data['lastUpdated'], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%d %b %Y") if 'lastUpdated' in coin_data else 'N/A'
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
    sources = {}
    q_lower = query.lower()
    financial_api_called = False




    # Stock data
    if any(k in q_lower for k in ["stock", "share", "price", "market", "ticker"]) and not any(k in q_lower for k in ["bitcoin", "ethereum", "btc", "eth", "crypto"]):
        financial_api_called = True
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
                sources["stock"] = "Alpha Vantage"
                print(f"Stock data for {ticker} retrieved successfully")
        except Exception as e:
            api_status["stock_data"] = f"Exception: {str(e)}"
            responses["stock_data"] = {"error": str(e)}

   


    # Crypto data
    if any(k in q_lower for k in ["crypto", "bitcoin", "ethereum", "btc", "eth", "coin", "blockchain"]):
        financial_api_called = True
        crypto_symbol = extract_ticker(query)
        print(f"Fetching crypto data for: {crypto_symbol}")
        crypto_data = get_crypto_data(crypto_symbol)
        if isinstance(crypto_data, dict) and "error" in crypto_data:
            api_status["crypto"] = f"Error: {crypto_data['error']}"
        else:
            responses["crypto"] = crypto_data
            api_status["crypto"] = "Success"
            sources["crypto"] = "Coinranking "




     # Forex data
    if any(k in q_lower for k in ["forex", "currency", "exchange rate"]):
        financial_api_called = True
        print("Attempting to fetch forex data…")

        base_code = target_code = None

        m = re.search(r'\b([A-Za-z]{3})/([A-Za-z]{3})\b', query)
        if m:
            base_code, target_code = m.group(1).upper(), m.group(2).upper()
        else:

            m2 = re.search(
                r'\b([A-Za-z]{3})\b\s+(?:to|in)\s+\b([A-Za-z]{3})\b',
                query, re.IGNORECASE
            )
            if m2:
                base_code, target_code = m2.group(1).upper(), m2.group(2).upper()
            else:

                m3 = re.search(
                    r'from\s+([A-Za-z]{3})\s+to\s+([A-Za-z]{3})',
                    query, re.IGNORECASE
                )
                if m3:
                    base_code, target_code = m3.group(1).upper(), m3.group(2).upper()

        if not (base_code and target_code):
            codes = get_countries_currencies(query, max_countries=2)
            if len(codes) >= 2:
                base_code, target_code = codes[0], codes[1]
            else:
                base_code, target_code = "USD", "INR"

        forex_data = get_forex_data(base_code, target_code)
        if "error" in forex_data:
            api_status["forex"] = f"Error: {forex_data['error']}"
        else:
            responses["forex"]   = forex_data
            api_status["forex"]  = "Success"
            sources["forex"]     = "Alpha Vantage Forex Data"




    #  Google Search 
    if True:
        google_data = get_google_search_results(query)
        if "error" not in google_data:
            responses["google_search"] = google_data
            api_status["google_search"] = "Success"
            sources["news"] = "Web Results"
        else:
            api_status["google_search"] = google_data["error"]

    responses["_api_status"] = api_status
    responses["_sources"] = sources

    if len(responses) <= 1 and not financial_api_called:
        print("WARNING: No financial APIs could process this query")
        responses["error"] = "No relevant data sources found"

    return responses



#-----------------------------------------------------------------------------------------------------------------------------------------------

def load_model():
    MODEL_NAME = "microsoft/Phi-4-mini-instruct"
    try:
        print(f"Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"MODEL LOAD ERROR: {str(e)}")
        return None

generator = load_model()

#------------------------------------------------------------------------------------------------------------------------------------------------

def build_prompt(query, index_name, api_responses=None):
    try:

        prompt_template = (f"Current Date: {datetime.now().strftime('%d %b %Y')}\n\n")

        if api_responses is None:
            api_responses = determine_api_calls(query)

        # Build dynamic context
        data_sources = []
        
        # Stock Data Formatting
        if "stock_data" in api_responses and "error" not in api_responses["stock_data"]:
            stock = api_responses["stock_data"]
            data_sources.append(
                f"Stock Data ({stock['symbol']}):\n"
                f"Price: ${stock['price']:.2f}\n"
                f"Volume: {stock['volume']:,}\n"
                f"As of: {stock['date']}"
            )
        
        # Crypto Data Formatting
        if "crypto" in api_responses and "error" not in api_responses["crypto"]:
            crypto = api_responses["crypto"]
            data_sources.append(
                f"Crypto Data ({crypto['symbol']}):\n"
                f"Price: ${crypto['price']:,.2f}\n"
                f"24h Vol: ${crypto['24hVolume']:,.0f}\n"
                f"Market Cap: ${crypto['marketCap']:,.0f}"
            )
        
        # Forex Data Formatting
        if "forex" in api_responses and "error" not in api_responses["forex"]:
            forex = api_responses["forex"]
            data_sources.append(
                f"Forex Rates ({forex['base']}/{forex['target']}):\n"
                f"Rate: {forex['rate']:.4f}\n"
                f"Updated: {forex['timestamp']}"
            )
        
        # Google Search Formatting
        if "google_search" in api_responses and "results" in api_responses["google_search"]:
            news = api_responses["google_search"]["results"]
            if news:
                news_context = [
                    "Web Updates:",
                    *[f"{i+1}. {result['title']}: {result['snippet']}\n   Link: {result['link']}"
                      for i, result in enumerate(news[:3])]
                ]
                data_sources.append("\n".join(news_context))

        prompt_template += "\n".join(data_sources) + "\n"
        
        return prompt_template
        
    except Exception as e:
        print(f"Prompt error: {str(e)}")
        return f"System Error: {query}"

#------------------------------------------------------------------------------------------------------------------------------------------------

def generate_final_answer(query, index_name):

    try:
        print("\nProcessing query...")
        api_responses = determine_api_calls(query)


        prompt = build_prompt(query, INDEX_NAME, api_responses)

        print(prompt[:1350] + "..." if len(prompt) > 1350 else prompt)

        output = generator(
            prompt,
            do_sample=True,
            temperature=0.5,
            max_length=2500,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=generator.tokenizer.eos_token_id,
            truncation=False,
            return_full_text=False,
            max_time=2
        )

        full_response = output[0]['generated_text']
        split_marker = "<|start_header_id|>assistant<|end_header_id|>"
        
        if split_marker in full_response:
            clean_answer = full_response.split(split_marker)[-1]

            clean_answer = clean_answer.split("<|end_of_text|>")[0].strip()
            clean_answer = clean_answer.split("<|")[0].strip()
        else:
            clean_answer = full_response

        formatted_lines = []
        current_number = 1

        content_type = "general"
        if "stock_data" in api_responses:
            content_type = "stock"
            base_header = f"{api_responses['stock_data']['symbol']} Stock Analysis - {api_responses['stock_data']['date']}"
        elif "crypto" in api_responses:
            content_type = "crypto"
            base_header = f"{api_responses['crypto']['name']} Update"
        elif "forex" in api_responses:
            content_type = "forex"
            base_header = f"{api_responses['forex']['base']}/{api_responses['forex']['target']} Rates"
        else:
            base_header = "Financial Analysis"

        formatted_lines.append(base_header)

        for line in clean_answer.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith(("1.", "•", "-", "*")):
                line = f"{current_number}. {line[2:].strip()}"
                current_number += 1

            formatted_lines.append(line)

        if not any(word in clean_answer.lower() for word in ["note:", "closing", "reminder", "summary"]):
            formatted_lines.append("\nNeed more details or clarification? Just ask!")


        references = []
        if "google_search" in api_responses and "results" in api_responses["google_search"]:
            references.extend([result['link'] for result in api_responses["google_search"]["results"][:3]])

        for api_type in ["stock", "crypto", "forex"]:
            if api_type in api_responses.get("_sources", {}):
                references.append(api_responses["_sources"][api_type])

        if references:
            formatted_lines.append("\nReferences:")
            formatted_lines.extend([f"• {ref}" for ref in references if ref])

        min_expected_length = 5
        if len(formatted_lines) < min_expected_length:
            formatted_lines.append("\n[Additional analysis unavailable due to response constraints]")
            formatted_lines.append("Please refine your query or check market data availability.")

        formatted_answer = "\n".join(formatted_lines)


        internal_context = get_internal_context(query, index_name)

        return {
            "answer": formatted_answer,
            "api_data": api_responses,
            "internal_context": internal_context,
            "error": None
        }
        
    except Exception as e:
        print(f"\nProcessing Error: {str(e)}")
        return {
            "answer": "Error processing request. Please try again.",
            "internal_context": "",
            "error": str(e)
        }

#------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("**********************************Enter your queries below**********************************")
    print("\n💰 Financial Assistant - Ctrl+C to exit")
    while True:
        try:
            query = input("\n📩 Your query: ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit']:
                break

            result = generate_final_answer(query, INDEX_NAME)

            print("\n💡 Analysis Results:")
            print("━" * 50)
            print(result["answer"])
            print("━" * 50)

            internal_ctx = result.get("internal_context", "").strip()
            if internal_ctx:
                print("\n🔎 Internal Context:")
                print(internal_ctx)
                print("━" * 50)

            if result.get("error"):
                print(f"\n⚠️error: {result['error']}")

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n🔥 Unexpected error: {str(e)}")

    print("\n✨ Session ended. Thank you for using the financial assistant!")


#------------------------------------------------------------------------------------------------------------------------------------------------