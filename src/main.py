import os
import requests
from dotenv import load_dotenv
from src.helper import query_index
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,BitsAndBytesConfig
import torch

load_dotenv()
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
INDEX_NAME = "finance"


import asyncio
import platform
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())



def get_google_news(query):
    url = "https://google-news13.p.rapidapi.com/business"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "google-news13.p.rapidapi.com"
    }
    params = {"q": query, "lang": "en"}
    response = requests.get(url, headers=headers, params=params,timeout=10)
    return response.json()



def get_yahoo_finance(query):
    url =  "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/quotes"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
    }
    params = {"q": query}
    response = requests.get(url, headers=headers, params=params,timeout=10)
    return response.json()



def get_alpha_vantage(query):
    url = "https://alpha-vantage.p.rapidapi.com/query"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
    }
    params = {"function": "TIME_SERIES_INTRADAY", "symbol": query, "interval": "5min"}
    response = requests.get(url, headers=headers, params=params,timeout=10)
    return response.json()



def get_live_metal_prices(query):
    url = "https://live-metal-prices.p.rapidapi.com/v1/latest/XAU,XAG,PA,PL,GBP,EUR/EUR"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "live-metal-prices.p.rapidapi.com"
    }
    params = {"metal": query}
    response = requests.get(url, headers=headers, params=params,timeout=10)
    return response.json()



def get_coinranking(query):
    url = "https://coinranking1.p.rapidapi.com/stats"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
    }
    params = {"search": query}
    response = requests.get(url, headers=headers, params=params,timeout=10)
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
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Try a simpler approach first
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )
    except Exception as e:
        print(f"MODEL LOAD ERROR DETAILS: {str(e)}")
        return None

generator = load_model()


# ------------------------------------------------------------------------------------------------------------------------------------------------

def build_prompt(query, index_name):
    """Build a prompt with better context handling"""
    try:
        internal_context = get_internal_context(query, index_name)
        api_responses = determine_api_calls(query)
        
        # Clean API responses
        cleaned_responses = []
        for api_name, response in api_responses.items():
            if isinstance(response, dict):
                cleaned = "\n".join([f"{k}: {v}" for k,v in response.items()][:3])
                cleaned_responses.append(f"{api_name}:\n{cleaned}")
            else:
                cleaned_responses.append(f"{api_name}: {str(response)[:200]}")

        prompt_template = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are a financial expert assistant. Use the following context to answer the question.\n"
            f"Internal Context: {internal_context[:1000]}\n"
            f"External Data:\n" + "\n".join(cleaned_responses) + "\n"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"Question: {query}\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"Answer:"
        )
        
        return prompt_template
    except Exception as e:
        print(f"Prompt building error: {str(e)}")
        return query



def generate_final_answer(query, index_name):
    if not generator:
        return "Error: Model not loaded properly"
    
    try:
        prompt = build_prompt(query, index_name)
        output = generator(
            prompt,
            max_new_tokens=256, 
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=generator.tokenizer.eos_token_id,
            pad_token_id=generator.tokenizer.eos_token_id, 
            truncation=True,  
            max_time=30,  
            num_return_sequences=1 
        )
        return output[0]['generated_text'].split("Answer:")[-1].strip()
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return f"Error generating answer: {str(e)}"