import argparse
import json
from types import SimpleNamespace

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.tools.utils import pipeline
import numpy as np
from datasets import load_dataset
from pathlib import Path

# REWRITE_INIT_PROMPT = """
# You are an intelligent orchestrator for a multimodal image retrieval and filtering pipeline. Given a user-provided text prompt and a set of candidate images (N = 1000), your task is to plan and execute a sequence of operations to maximize retrieval quality.

#   Analyze the input text prompt. Determine whether it contains ecological or domain-specific jargon that may hinder text-to-image retrieval. If such jargon exists, rewrite the prompt into a more visually descriptive and generalizable form. Preserve the original semantic meaning while emphasizing observable attributes such as color, shape, texture, habitat, and pose. If not, keep the original prompt. If the prompt requires species identification and contains common names, consider rewriting it to include scientific names or broader taxonomic groups to improve retrieval. Use web search to look up common names, scientific names, habitats, and visual descriptors that help retrieve images. Use web search only when necessary.

# Output format (JSON only, no extra text):
# {"rewritten_prompt": "...", "used_search": if search was used, "notes": "..."}
# """.strip()


REWRITE_INIT_PROMPT = """
You are a prompt rewriting assistant for CLIP-based text-image similarity search.

Analyze the input text prompt. Determine whether it contains ecological or domain-specific jargon that may hinder text-to-image retrieval. If such jargon exists, rewrite the prompt into a more visually descriptive and generalizable form. Preserve the original semantic meaning while emphasizing observable attributes such as color, shape, texture, habitat, and pose. If not, keep the original prompt. If the prompt requires species identification and contains common names, consider rewriting it to include scientific names or broader taxonomic groups to improve retrieval. 

Use `run_web_search` to look up common names, scientific names, habitats, and visual descriptors that help retrieve images. 

Your job is to augment the original query, not replace it. Expand the original prompt to be as visually descriptive as possible while preserving its intent. The rewritten prompt must include the original prompt verbatim, followed by an
expanded description that adds observable attributes (color, shape, texture,
pose, habitat, background, count) and clarifies any ambiguity. The augmented prompt should be descriptive enough to help retrieve the correct images, but not so verbose that it dilutes the original intent. Do not include questions. 

Output format (JSON only, no extra text):
{"rewritten_prompt": "...", "used_search": depends on whether the model used search, "notes": "..."}
""".strip()


def to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj


def load_config(config_path="config.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return to_namespace(config)


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token_id
    return model, tokenizer


def build_rewrite_config(base_config):
    return SimpleNamespace(
        init_prompt=REWRITE_INIT_PROMPT,
        max_tool_calls=base_config.max_tool_calls,
        chat_template=base_config.chat_template,
        generate=base_config.generate,
    )


def extract_rewritten_prompt(text: str) -> str | None:
    start = text.find("{")
    if start != -1:
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        payload = json.loads(text[start:idx + 1])
                        value = payload.get("rewritten_prompt")
                        if isinstance(value, str) and value.strip():
                            return value.strip()
                    except json.JSONDecodeError:
                        break
    return None


def rewrite_query(query: str, model, tokenizer, rewrite_config) -> tuple[str, str]:
    response = pipeline(model, tokenizer, query, config=rewrite_config)
    rewritten = extract_rewritten_prompt(response) or query
    return rewritten, response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite queries using web search tool calls.")
    parser.add_argument("--query", type=str, help="Single query to rewrite")
    parser.add_argument("--config", type=str, default="configs/eval.yml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()

    base_config = load_config(args.config)
    rewrite_config = build_rewrite_config(base_config)
    model, tokenizer = load_model(base_config.model.name)

    if args.query:
        queries = [args.query]
    else:
        split = args.split
        dataset = load_dataset("evendrow/INQUIRE-Rerank", split=("validation" if split == "val" else "test"))
        queries = np.unique(dataset["query"]).tolist()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rewritten_cache = {}
    response_cache = {}

    for i, query_text in enumerate(queries):
        rewritten, response = rewrite_query(query_text, model, tokenizer, rewrite_config)
        rewritten_cache[query_text] = rewritten
        response_cache[query_text] = response

    responses_path = output_dir / "rewrite_tool_responses_extended.json"
    with responses_path.open("w", encoding="utf-8") as f:
        json.dump(response_cache, f, indent=2, ensure_ascii=True)

    rewrites_path = output_dir / "rewrite_prompts_extended.json"
    rewrite_pairs = [
        {"original": original, "rewritten": rewritten_cache[original]}
        for original in rewritten_cache
    ]
    with rewrites_path.open("w", encoding="utf-8") as f:
        json.dump(rewrite_pairs, f, indent=2, ensure_ascii=True)

    print(f"Saved tool responses to {responses_path}")
    print(f"Saved rewrites to {rewrites_path}")
       