from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quant_config,
    use_auth_token=True
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "Provide a clear and concise summary of today's major financial market trends,including key stock movements, market sentiment, and economic news:"
output = generator(prompt, max_new_tokens=100, do_sample=False)
print("Output:", output)
