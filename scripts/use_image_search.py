import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.tools.utils import pipeline
from src.tools.search.web_search import run_web_image_search

from datasets import load_dataset
import numpy as np


SEARCH_INIT_PROMPT = """
You are a tool-calling assistant that finds images for a query.

Use the tool `run_web_image_search` to fetch image results. Pick the 3 most
relevant image URLs for the query.

Output format (JSON only, no extra text):
{"image_urls": ["...", "...", "..."]}
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


def build_search_config(base_config):
    return SimpleNamespace(
        init_prompt=SEARCH_INIT_PROMPT,
        max_tool_calls=base_config.max_tool_calls,
        chat_template=base_config.chat_template,
        generate=base_config.generate,
    )


def extract_image_urls(text: str) -> list[str]:
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
                        urls = payload.get("image_urls")
                        if isinstance(urls, list):
                            return [str(url).strip() for url in urls if str(url).strip()]
                    except json.JSONDecodeError:
                        break
    urls = re.findall(r"https?://\S+", text)
    return urls


def download_image(url: str, output_dir: Path, filename: str) -> Path | None:
    output_path = output_dir / filename
    try:
        request = Request(url, headers={"User-Agent": "llm-course-project/1.0"})
        with urlopen(request, timeout=20) as response:
            content = response.read()
        output_path.write_bytes(content)
        return output_path
    except Exception:
        return None


def infer_extension(url: str) -> str:
    path = urlparse(url).path
    if "." in path:
        ext = path.rsplit(".", 1)[-1].lower()
        if 1 <= len(ext) <= 5:
            return ext
    return "jpg"


def fetch_images_for_query(query: str, model, tokenizer, search_config, output_dir: Path) -> dict:
    tool_result = run_web_image_search(query, max_results=3, region="us", safesearch="on", backend="v2")
    urls = [item.get("image") for item in tool_result.get("results", []) if item.get("image")]

    image_paths = []
    for idx, url in enumerate(urls, start=1):
        ext = infer_extension(url)
        safe_query = re.sub(r"[^a-z0-9_-]+", "_", query.lower()).strip("_")
        filename = f"{safe_query}_{idx}.{ext}"
        saved = download_image(url, output_dir, filename)
        if saved:
            image_paths.append(str(saved))

    return {
        "query": query,
        "image_urls": urls,
        "image_paths": image_paths,
        "tool_result": tool_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 3 web images per query using tool-calling.")
   #  parser.add_argument("--query", type=str, required=True, help="Query to search for images")
    parser.add_argument("--config", type=str, default="configs/eval.yml")
    parser.add_argument("--output", type=str, default="output/web_images")
    args = parser.parse_args()

    base_config = load_config(args.config)
    search_config = build_search_config(base_config)
    model, tokenizer = load_model(base_config.model.name)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = "test"
    dataset = load_dataset("evendrow/INQUIRE-Rerank", split=("validation" if split == "val" else "test"))
    queries = np.unique(dataset["query"]).tolist()

    all_metadata = []
    for i, query_text in enumerate(queries):
        payload = fetch_images_for_query(query_text, model, tokenizer, search_config, output_dir)
        all_metadata.append(payload)
        print(f"Downloaded {len(payload['image_paths'])} images to {output_dir}")

    metadata_path = output_dir / "image_search_metadata_test.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=True)

    print(f"Saved metadata to {metadata_path}")

      
