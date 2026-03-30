import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.tools.utils import logger, pipeline
import yaml

def load_config(config_path="config.yml"):
    with open(config_path, "r") as f:
        # safe_load avoids executing arbitrary code
        return yaml.safe_load(f)


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto", dtype=torch.float16, trust_remote_code=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token_id
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

# Load configuration


if __name__ == "__main__":
    config = load_config('configs/eval.yml')
    model, tokenizer = load_model(config["model"]["name"])
    image_path = 'data/demo.jpg'
    user_query = "How many bears are there in the image: {}.".format(image_path)
    model_response = pipeline(model, tokenizer, user_query, config=config)
    logger.info(f'\nModel response:\n{model_response}')

    