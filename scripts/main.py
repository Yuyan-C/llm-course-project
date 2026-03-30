import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.tools.utils import logger, pipeline

if __name__ == "__main__":


    MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,device_map="auto", dtype=torch.float16, trust_remote_code=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token_id

    print("Model and tokenizer loaded successfully.")

    image_path = 'data/demo.jpg'
    user_query = "How many bears are there in the image: {}.".format(image_path)
    model_response = pipeline(model, tokenizer, user_query, max_new_tokens=512)
    logger.info(f'\nModel response:\n{model_response}')

    