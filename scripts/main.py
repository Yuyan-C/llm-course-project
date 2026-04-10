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
import argparse

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


def _normalize_bioclip_predictions(bioclip_result: object | None) -> list[dict]:
    if not bioclip_result:
        return []

    predictions: list[dict] = []

    if isinstance(bioclip_result, dict):
        if "predictions" in bioclip_result and isinstance(bioclip_result["predictions"], list):
            for item in bioclip_result["predictions"]:
                if isinstance(item, dict) and "species" in item and "score" in item:
                    predictions.append(item)
            return predictions
        if "species" in bioclip_result and "score" in bioclip_result:
            return [bioclip_result]
        if all(isinstance(value, (int, float)) for value in bioclip_result.values()):
            for species_name, score in bioclip_result.items():
                predictions.append({"species": species_name, "score": float(score)})
            return predictions
        return []

    if isinstance(bioclip_result, list):
        for item in bioclip_result:
            if isinstance(item, dict) and "predictions" in item and isinstance(item["predictions"], list):
                for pred in item["predictions"]:
                    if isinstance(pred, dict) and "species" in pred and "score" in pred:
                        predictions.append(pred)
                continue
            if isinstance(item, dict) and "species" in item and "score" in item:
                predictions.append(item)
        return predictions

    return []


def _best_bioclip_match(
    bioclip_result: object | None,
    query_tokens: set[str],
    top_k: int = 5,
) -> tuple[str | None, float]:
    predictions = _normalize_bioclip_predictions(bioclip_result)
    if not predictions:
        return None, 0.0

    sorted_items = sorted(predictions, key=lambda item: item.get("score", 0.0), reverse=True)
    best_name = None
    best_score = 0.0
    for item in sorted_items[:top_k]:
        species_name = str(item.get("species", ""))
        score = float(item.get("score", 0.0))
        species_tokens = _tokenize(species_name)
        if species_tokens & query_tokens:
            if score > best_score:
                best_name = species_name
                best_score = score
    return best_name, best_score


def _summarize_dino(dino_result: dict | None, max_items: int = 5) -> list[dict]:
    if not dino_result:
        return []
    labels = dino_result.get("labels", []) or []
    scores = dino_result.get("scores", []) or []
    items: list[dict] = []
    for idx, label in enumerate(labels):
        score = scores[idx] if idx < len(scores) else None
        if label is None:
            continue
        items.append({"label": str(label), "score": float(score) if score is not None else None})
    items.sort(key=lambda item: (item["score"] is not None, item["score"]), reverse=True)
    return items[:max_items]


def _summarize_bioclip(bioclip_result: object | None, top_k: int = 5) -> list[dict]:
    predictions = _normalize_bioclip_predictions(bioclip_result)
    if not predictions:
        return []
    sorted_items = sorted(predictions, key=lambda item: item.get("score", 0.0), reverse=True)
    summary: list[dict] = []
    for item in sorted_items[:top_k]:
        species = str(item.get("species", ""))
        score = float(item.get("score", 0.0))
        entry = {"species": species, "score": score}
        common_name = item.get("common_name")
        if common_name:
            entry["common_name"] = str(common_name)
        summary.append(entry)
    return summary


def _extract_first_json_block(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:idx + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _get_single_token_id(tokenizer, text: str) -> int:
    for variant in (f" {text}", text):
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(token_ids) == 1:
            return token_ids[0]
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return token_ids[0]


def judge_relevance_with_model(
    model,
    tokenizer,
    query_text: str,
    dino_result: dict | None,
    bioclip_result: object | None,
) -> tuple[bool, float, dict]:
    dino_summary = _summarize_dino(dino_result)
    bioclip_summary = _summarize_bioclip(bioclip_result)

    tool_context = (
        "Tool results:\n"
        f"Grounding DINO: {json.dumps(dino_summary)}\n"
        f"BioCLIP: {json.dumps(bioclip_summary)}"
    )

    user_prompt = (
        f"Based on the result(s) of tool calling, does this picture show {query_text}? "
        "Respond with yes or no.\n\n"
        f"{tool_context}"
    )

    messages = [
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[:, -1, :]
    yes_token_id = _get_single_token_id(tokenizer, "Yes")
    no_token_id = _get_single_token_id(tokenizer, "No")
    logit_yes = float(logits[0, yes_token_id].item())
    logit_no = float(logits[0, no_token_id].item())

    score_yes = float(np.exp(logit_yes))
    score_no = float(np.exp(logit_no))
    score = score_yes / (score_yes + score_no) if (score_yes + score_no) > 0 else 0.0
    is_relevant = score >= 0.5

    return is_relevant, score, {"yes": logit_yes, "no": logit_no}


def select_relevant_images(
    batch_results: list[dict],
    query_text: str,
    dino_threshold: float = 0.3,
    bioclip_threshold: float = 0.25,
    top_k: int = 5,
    model=None,
    tokenizer=None,
    use_model_judge: bool = False,
) -> list[dict]:
    query_tokens = _tokenize(query_text)
    selected: list[dict] = []

    for item in batch_results:
        steps = item.get("steps", []) or []
        dino_result = _extract_step_result(steps, "run_grounding_dino")
        bioclip_result = _extract_step_result(steps, "run_bioclip")

        dino_score = _max_dino_score(dino_result)
        best_name, best_score = _best_bioclip_match(bioclip_result, query_tokens, top_k=top_k)

        if use_model_judge and model is not None and tokenizer is not None:
            is_relevant, model_score, model_logits = judge_relevance_with_model(
                model,
                tokenizer,
                query_text,
                dino_result,
                bioclip_result,
            )
            reason = None
        else:
            dino_ok = dino_score >= dino_threshold if dino_result is not None else True
            bioclip_ok = best_score >= bioclip_threshold if bioclip_result is not None else True
            is_relevant = dino_ok and bioclip_ok
            reason = None
            model_score = None
            model_logits = None

        item["model_relevant"] = is_relevant
        item["model_score"] = model_score
        item["model_logits"] = model_logits

        if is_relevant:
            selected.append(
                {
                    "image_path": item.get("image_path"),
                    "dino_score": dino_score,
                    "bioclip_match": best_name,
                    "bioclip_score": best_score,
                    "model_relevant": is_relevant,
                    "model_reason": reason,
                    "model_score": model_score,
                    "model_logits": model_logits,
                }
            )

    selected.sort(key=lambda x: (x["bioclip_score"], x["dino_score"]), reverse=True)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the reranking pipeline")
    parser.add_argument("--debug", action="store_true", help="Use only the first 20 images")
    args = parser.parse_args()

    config = load_config('configs/eval.yml')
    model, tokenizer = load_model(config.model.name)
    split = "test"
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
        if args.debug:
            image_paths = image_paths[:20]

        batch_results = execute_execution_plan_batch(
            plan_calls,
            image_paths,
            batch_size=4,
            batch_sizes={
                "run_grounding_dino": 16,
                "run_bioclip": 128,
            },
        )

        relevant_images = select_relevant_images(
            batch_results,
            query_text=query_text,
            dino_threshold=0.3,
            bioclip_threshold=0.25,
            top_k=5,
            model=model,
            tokenizer=tokenizer,
            use_model_judge=True,
        )

        output_payload = {
            "query": query_text,
            "model_response": model_response,
            "plan_calls": plan_calls,
            "num_images": len(image_paths),
            "results": batch_results,
            "relevant_images": relevant_images,
            "relevance_score_note": (
                "The logits of the \"Yes\" and \"No\" tokens are then used to compute the score: "
                "s = sy/(sy + sn), where sy = exp(logitYes) and sn = exp(logitNo)."
            ),
        }

        output_path = output_dir / f"batch_results_{i:04d}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        logger.info("Saved batch results to %s", output_path)

        if args.debug:
            if i == 3:
                break

    
        

    


    
