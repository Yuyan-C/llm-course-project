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
import re

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


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _extract_step_result(steps: list[dict], function_name: str) -> dict | None:
    for step in steps:
        if step.get("function") == function_name:
            return step.get("result")
    return None


def _max_dino_score(dino_result: dict | None) -> float:
    if not dino_result:
        return 0.0
    scores = dino_result.get("scores", []) or []
    return max(scores) if scores else 0.0


def _best_bioclip_match(
    bioclip_result: dict | None,
    query_tokens: set[str],
    top_k: int = 5,
) -> tuple[str | None, float]:
    if not bioclip_result:
        return None, 0.0

    sorted_items = sorted(bioclip_result.items(), key=lambda item: item[1], reverse=True)
    best_name = None
    best_score = 0.0
    for species_name, score in sorted_items[:top_k]:
        species_tokens = _tokenize(species_name)
        if species_tokens & query_tokens:
            if score > best_score:
                best_name = species_name
                best_score = score
    return best_name, best_score



# TODO: fix this function; we should not use hardcoded thresholds, and we should not require both DINO and BioCLIP to be present
def select_relevant_images(
    batch_results: list[dict],
    query_text: str,
    dino_threshold: float = 0.3,
    bioclip_threshold: float = 0.25,
    top_k: int = 5,
) -> list[dict]:
    query_tokens = _tokenize(query_text)
    selected: list[dict] = []

    for item in batch_results:
        steps = item.get("steps", []) or []
        dino_result = _extract_step_result(steps, "run_grounding_dino")
        bioclip_result = _extract_step_result(steps, "run_bioclip")

        dino_score = _max_dino_score(dino_result)
        best_name, best_score = _best_bioclip_match(bioclip_result, query_tokens, top_k=top_k)

        dino_ok = dino_score >= dino_threshold if dino_result is not None else True
        bioclip_ok = best_score >= bioclip_threshold if bioclip_result is not None else True

        if dino_ok and bioclip_ok:
            selected.append(
                {
                    "image_path": item.get("image_path"),
                    "dino_score": dino_score,
                    "bioclip_match": best_name,
                    "bioclip_score": best_score,
                }
            )

    selected.sort(key=lambda x: (x["bioclip_score"], x["dino_score"]), reverse=True)
    return selected


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

        relevant_images = select_relevant_images(
            batch_results,
            query_text=query_text,
            dino_threshold=0.3,
            bioclip_threshold=0.25,
            top_k=5,
        )

        output_payload = {
            "query": query_text,
            "model_response": model_response,
            "plan_calls": plan_calls,
            "num_images": len(image_paths),
            "results": batch_results,
            "relevant_images": relevant_images,
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

    
