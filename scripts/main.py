import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.tools.utils import logger, pipeline, parse_execution_plan, execute_execution_plan_batch
import yaml
from types import SimpleNamespace
from datasets import load_dataset
import numpy as np
import json
from pathlib import Path

def to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj


def load_config(config_path="config.yml"):
    with open(config_path, "r") as f:
        # safe_load avoids executing arbitrary code
        config = yaml.safe_load(f)
    return to_namespace(config)

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
    model, tokenizer = load_model(config.model.name)
    split = "val"
    dataset = load_dataset("evendrow/INQUIRE-Rerank", split=("validation" if split == "val" else "test"))
    queries = np.unique(dataset["query"]).tolist()

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, query_text in enumerate(queries):
        logger.info(f"[{i+1}/{len(queries)}] Processing query: {query_text}")
        user_query = query_text
        model_response = pipeline(model, tokenizer, user_query, config=config)

        plan_calls = parse_execution_plan(model_response)
        query_ds = dataset.select(np.argwhere(np.asarray(dataset["query"]) == query_text).squeeze())
        image_paths = [os.path.join(config.image_dir, str(p)) for p in query_ds["inat24_file_name"]]

        batch_results = execute_execution_plan_batch(plan_calls, image_paths, batch_size=4)

        output_payload = {
            "query": query_text,
            "model_response": model_response,
            "plan_calls": plan_calls,
            "num_images": len(image_paths),
            "results": batch_results,
        }

        output_path = output_dir / f"batch_results_{i:04d}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        logger.info("Saved batch results to %s", output_path)

        break

   


        
    # image_path = 'data/demo.jpg'
    # user_query = "How many bears are there in the image: {}.".format(image_path)
    # model_response = pipeline(model, tokenizer, user_query, config=config)
    # logger.info(f'\nModel response:\n{model_response}')

    
