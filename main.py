import os
import requests
from dotenv import load_dotenv
from src.helper import query_index
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


load_dotenv()
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
INDEX_NAME = "finance"



def get_google_news(query):
    url = "https://google-news13.p.rapidapi.com/business"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "google-news13.p.rapidapi.com"
    }
    params = {"q": query, "lang": "en"}
    response = requests.get(url, headers=headers, params=params)
    return response.json()



def get_yahoo_finance(query):
    url =  "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/quotes"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
    }
    params = {"q": query}
    response = requests.get(url, headers=headers, params=params)
    return response.json()



def get_alpha_vantage(query):
    url = "https://alpha-vantage.p.rapidapi.com/query"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
    }
    params = {"function": "TIME_SERIES_INTRADAY", "symbol": query, "interval": "5min"}
    response = requests.get(url, headers=headers, params=params)
    return response.json()



def get_live_metal_prices(query):
    url = "https://live-metal-prices.p.rapidapi.com/v1/latest/XAU,XAG,PA,PL,GBP,EUR/EUR"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "live-metal-prices.p.rapidapi.com"
    }
    params = {"metal": query}
    response = requests.get(url, headers=headers, params=params)
    return response.json()



def get_coinranking(query):
    url = "https://coinranking1.p.rapidapi.com/stats"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
    }
    params = {"search": query}
    response = requests.get(url, headers=headers, params=params)
    return response.json()

# ----------------------------------------------------------------------------------------------------------------------------------------------


def determine_api_calls(query):

    responses = {}
    q_lower = query.lower()
    if "news" in q_lower or "headline" in q_lower:
        responses["google_news"] = get_google_news(query)

    if any(keyword in q_lower for keyword in ["stock", "price", "market", "finance"]):
        responses["yahoo_finance"] = get_yahoo_finance(query)
        responses["alpha_vantage"] = get_alpha_vantage(query)

    if any(keyword in q_lower for keyword in ["metal", "gold", "silver", "copper"]):
        responses["live_metal_prices"] = get_live_metal_prices(query)
        
    if any(keyword in q_lower for keyword in ["crypto", "bitcoin", "ethereum", "coin"]):
        responses["coinranking"] = get_coinranking(query)
    return responses


# ------------------------------------------------------------------------------------------------------------------------------------------------

def get_internal_context(query, index_name):
    """
    Retrieve internal resources from your vector store.
    Returns a concatenated string of retrieved content.
    """
    internal_results = query_index(query, index_name)
    if internal_results:
        context = " ".join([doc.content for doc in internal_results])
    else:
        context = ""
    return context

# ------------------------------------------------------------------------------------------------------------------------------------------------

MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# ------------------------------------------------------------------------------------------------------------------------------------------------

def build_prompt(query, index_name):
    """
    Build a prompt by combining internal context and external API data.
    """
    internal_context = get_internal_context(query, index_name)
    
    api_responses = determine_api_calls(query)
    external_context = ""
    for api_name, response in api_responses.items():
        external_context += f"\n{api_name}: {response}"
    
    prompt = (
        f"Context from internal resources: {internal_context}\n"
        f"External data: {external_context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    return prompt

def generate_final_answer(query, index_name):
    """
    Generate a final answer using the Mixtral model given a query.
    """
    prompt = build_prompt(query, index_name)
    output = generator(prompt, max_length=1024, do_sample=True, temperature=0.7)
    answer = output[0]['generated_text'][len(prompt):].strip()
    return answer

# ------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    user_query = input("Enter your query: ")

    final_answer = generate_final_answer(user_query, INDEX_NAME)
    print("Final Answer:")
    print(final_answer)
