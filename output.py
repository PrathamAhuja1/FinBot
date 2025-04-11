import os
import requests
from dotenv import load_dotenv
from src.helper import query_index, extract_ticker, get_internal_context
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from datetime import datetime
import asyncio
import platform

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables
load_dotenv()
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
INDEX_NAME = "finance"

RAPIDAPI_HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": None 
}

# -----------------------------------------------------------------------------------------------------------------------------------------------

def get_forex_data(base="USD", target="INR"):
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
                sources["stock"] = "Alpha Vantage API"
                print(f"Stock data for {ticker} retrieved successfully")
        except Exception as e:
            api_status["stock_data"] = f"Exception: {str(e)}"
            responses["stock_data"] = {"error": str(e)}

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
            sources["crypto"] = "Coinranking API"

    # Forex data
    if any(k in q_lower for k in ["forex", "currency", "exchange rate"]):
        print("Attempting to fetch forex data...")
        forex_data = get_forex_data("USD", "INR")
        if "error" in forex_data:
            api_status["forex"] = f"Error: {forex_data['error']}"
        else:
            responses["forex"] = forex_data
            api_status["forex"] = "Success"        
            sources["forex"] = "Alpha Vantage Forex Data"

    # Google Search
    if len(responses) <= 1:
        google_data = get_google_search_results(query)
        if "error" in google_data:
            api_status["google_search"] = google_data["error"]
        else:
            responses["google_search"] = google_data
            api_status["google_search"] = "Success"
            sources["news"] = "Web Results"

    responses["_api_status"] = api_status
    responses["_sources"] = sources

    if len(responses) <= 1:
        print("WARNING: No successful API calls were made for this query")
        responses["error"] = "No financial data sources could be accessed for this query"

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
        # Dynamic response template
        prompt_template = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are a financial expert. Format responses with these rules:\n"
            f"1. Use numbered lists with ▸ symbols\n"
            f"2. Include relevant emojis in headers\n"
            f"3. Show key metrics in bold\n"
            f"4. Always cite sources\n"
            f"5. Add a closing remark\n"
            f"6. Keep explanations concise\n\n"
            f"Format Examples:\n"
            f"Stock Example:\n"
            f"📈 AAPL Stock Analysis - 15 Mar 2024\n"
            f"1▸ 💵 Price: $175.42 (+1.2% from yesterday)\n"
            f"2▸ 📊 Volume: 45M shares traded\n\n"
            f"Crypto Example:\n"
            f"🔐 Bitcoin Update\n"
            f"1▸ 💵 Price: $63,450.80\n"
            f"2▸ 📈 24h Change: +3.2%\n\n"
            f"News Example:\n"
            f"📰 Market Updates\n"
            f"1▸ Fed maintains interest rates at 5.25%\n\n"
            f"Current Date: {datetime.now().strftime('%d %b %Y')}\n"
        )

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
        if "google_search" in api_responses:
            news = api_responses["google_search"]
            if "results" in news:
                data_sources.append("News Context:\n" + "\n".join(
                    f"{result['title']}: {result['snippet']}" 
                    for result in news["results"][:3]
                ))

        prompt_template += "\n".join(data_sources) + "\n"
        prompt_template += f"<|start_header_id|>user<|end_header_id|>\n{query}\n"
        prompt_template += f"<|start_header_id|>assistant<|end_header_id|>\n"
        
        return prompt_template
        
    except Exception as e:
        print(f"Prompt error: {str(e)}")
        return f"System Error: {query}"

#------------------------------------------------------------------------------------------------------------------------------------------------

def generate_final_answer(query, index_name):
    if not generator:
        print("❌ Error: AI model failed to initialize")
        return {
            "answer": "System initialization failed. Please check your model configuration.",
            "error": "Model not loaded"
        }
    
    try:
        print("\n🔍 Processing query...")
        api_responses = determine_api_calls(query)

        print("\n📦 Raw API Responses:")
        for api_name, data in api_responses.items():
            print(f"{api_name.upper():<15}: {str(data)[:100]}...")

        prompt = build_prompt(query, INDEX_NAME, api_responses)

        print("\n📝 Generated Prompt:")
        print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)

        output = generator(
            prompt,
            do_sample=True,
            temperature=0.5,
            max_length=2000,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=generator.tokenizer.eos_token_id,
            truncation=True
        )

        full_response = output[0]['generated_text']
        split_marker = "<|start_header_id|>assistant<|end_header_id|>"
        
        if split_marker in full_response:
            clean_answer = full_response.split(split_marker)[-1].strip()
            clean_answer = clean_answer.split("<|")[0].strip()
        else:
            clean_answer = full_response

        formatted_lines = []
        current_number = 1
        source_counter = 1
        
        # Detect content type for header
        content_type = "general"
        header_emoji_map = {
            "stock": "📈",
            "crypto": "🔐",
            "forex": "💱",
            "news": "📰"
        }
        
        if "stock_data" in api_responses:
            content_type = "stock"
            base_header = f"{api_responses['stock_data']['symbol']} Stock Analysis - {api_responses['stock_data']['date']}"
        elif "crypto" in api_responses:
            content_type = "crypto"
            base_header = f"{api_responses['crypto']['name']} Update"
        elif "google_search" in api_responses:
            content_type = "news"
            base_header = "Market Updates"
        else:
            base_header = "Financial Analysis"

        # Add formatted header
        formatted_lines.append(f"{header_emoji_map.get(content_type, '📊')} {base_header}")
        
        # Process answer lines
        for line in clean_answer.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith(("1.", "•", "-", "*")):
                line = f"{current_number}▸ {line[2:].strip()}"
                current_number += 1

            for pattern in ["$", "%", "USD", "INR", "volume", "price", "rate", "cap"]:
                if pattern in line:
                    line = line.replace(pattern, f"**{pattern}**")
                    
            formatted_lines.append(line)

        # Add closing remark if missing
        if not any(word in clean_answer.lower() for word in ["note:", "closing", "reminder"]):
            formatted_lines.append("\n💡 Need more details or clarification? Just ask!")

        # Add references
        references = []
        source_map = {
            "stock": "Alpha Vantage API Data",
            "crypto": "Coinranking API",
            "forex": "Alpha Vantage Forex Data",
            "news": "Web Results"
        }
        
        for idx, (api_type, source_name) in enumerate(api_responses.get("_sources", {}).items(), 1):
            references.append(f"{idx}. {source_map.get(api_type, source_name)}")

        if references:
            formatted_lines.append("\n📚 References:")
            formatted_lines.extend(references)

        # Final emoji formatting
        emoji_map = {
            "stock": "📈",
            "crypto": "🔐",
            "forex": "💱",
            "price": "💵",
            "volume": "📊",
            "news": "📰",
            "rate": "📈"
        }
        
        formatted_answer = "\n".join(formatted_lines)
        for keyword, emoji in emoji_map.items():
            formatted_answer = formatted_answer.replace(f"**{keyword}**", f"{emoji} {keyword.title()}")

        return {
            "answer": formatted_answer,
            "api_data": api_responses,
            "error": None
        }
        
    except Exception as e:
        print(f"\n⚠️ Processing Error: {str(e)}")
        return {
            "answer": "🚨 Oops! Something went wrong while processing your request. Please try again.",
            "error": str(e)
        }


#------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
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
            
            if result.get("error"):
                print(f"\n⚠️ Behind the scenes error: {result['error']}")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n🔥 Unexpected error: {str(e)}")

    print("\n✨ Session ended. Thank you for using the financial assistant!")

#-----------------------------------------------------------------------------------------------------------------------------------------------