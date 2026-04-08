import json
from typing import Any, Dict, List
from collections.abc import Callable
import re
from copy import deepcopy

import argparse
import logging
from transformers.utils import get_json_schema
from PIL import Image

from src.tools.object_detection.grounding_dino import run_grounding_dino
from src.tools.classification.bioclip_cls import run_bioclip

FUNCTION_MAP: Dict[str, Callable] = {
    "run_grounding_dino": run_grounding_dino,
    "run_bioclip": run_bioclip,
}

TOOLS = [get_json_schema(func) for func in FUNCTION_MAP.values()]

LOGGING_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

parser = argparse.ArgumentParser(description="Set the logging level via command line")
parser.add_argument(
        '--log', 
        default='DEBUG', 
        choices=LOGGING_LEVELS.keys(),
        help=f"Set the logging level. Choices: {list(LOGGING_LEVELS.keys())}. Default: WARNING."
    )
args = parser.parse_args()

logger = logging.getLogger(__name__)
level = LOGGING_LEVELS.get(args.log.upper(), logging.WARNING)
logger.setLevel(level)
ch = logging.StreamHandler()
ch.setLevel(level)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
            
            

def execute_tool_call(function_map:Dict[str, Callable], func_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single tool call.
    """

    if func_name not in function_map:
        return {
            "error": f"Unknown function: {func_name}",
            "avaialbe_functions": list(function_map.keys())
        }

    try: 
        return function_map[func_name](**args)
    except Exception as e:
        return {
            "error": f"Error executing {func_name}: {str(e)}",
            "function": func_name,
            "arguments": args
        }



def parse_tool_calls(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from model response.
    Supports multiple formats:
    - <tool_call>{...}</tool_call>
    - Multiple individual JSON objects on separate lines
    - [TOOL_CALLS] JSON_ARRAY [/TOOL_CALLS]
    - Generic JSON objects with "name" and "arguments"
    """
    tool_calls = []

    try:
        tool_call_pattern = r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>"
        tool_call_matches = re.findall(tool_call_pattern, response_text)

        if tool_call_matches:
            for match in tool_call_matches:
                try:
                    json_str = match.strip()
                    tool_call = json.loads(json_str)
                    tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        pass

    if not tool_calls:
        try:
            json_pattern = r"\{[^{}]*\"name\"[^{}]*\"arguments\"[^{}]*\}"
            json_matches = re.findall(json_pattern, response_text)
            for match in json_matches:
                try:
                    tool_call = json.loads(match)
                    if "name" in tool_call:
                        tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            pass
    
    if not tool_calls:
        try:
            match = re.search(r"\[TOOL_CALLS\](.*?)(\[/TOOL_CALLS\]|$)", response_text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                if json_str.startswith("["):
                    tool_calls = json.loads(json_str)
                else:
                    tool_calls = [json.loads(json_str)]
        except Exception as e:
            pass
    
    return tool_calls


def _extract_json_block_after_header(response_text: str, header: str) -> str | None:
    """
    Extract the first JSON object that appears after a given header.
    """
    start_idx = response_text.find(header)
    if start_idx == -1:
        return None

    candidate = response_text[start_idx + len(header):]
    brace_start = candidate.find("{")
    if brace_start == -1:
        return None

    brace_depth = 0
    json_start = start_idx + len(header) + brace_start
    for i in range(json_start, len(response_text)):
        if response_text[i] == "{":
            brace_depth += 1
        elif response_text[i] == "}":
            brace_depth -= 1
            if brace_depth == 0:
                return response_text[json_start:i + 1]

    return None


def _to_sorted_plan_list(plan_obj: Any) -> List[Dict[str, Any]]:
    """
    Convert a plan object to a sorted list of tool calls.
    Accepts:
    - {"Step 1": {"function": ..., "arguments": ...}, ...}
    - [{"function": ..., "arguments": ...}, ...]
    """
    plan_calls: List[Dict[str, Any]] = []

    if isinstance(plan_obj, list):
        for step in plan_obj:
            if isinstance(step, dict) and "function" in step:
                plan_calls.append({"name": step["function"], "arguments": step.get("arguments", {})})
        return plan_calls

    if not isinstance(plan_obj, dict):
        return plan_calls

    def step_sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        step_name = item[0]
        match = re.search(r"\d+", str(step_name))
        if match:
            return (int(match.group()), str(step_name))
        return (10**9, str(step_name))

    for _, step in sorted(plan_obj.items(), key=step_sort_key):
        if not isinstance(step, dict):
            continue
        function_name = step.get("function") or step.get("name")
        if not function_name:
            continue
        plan_calls.append({
            "name": function_name,
            "arguments": step.get("arguments", {}),
        })

    return plan_calls


def parse_execution_plan(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse an execution plan from the final model response.
    """
    plan_json: str | None = None

    plan_json = _extract_json_block_after_header(response_text, "Execution Plan:")
    if plan_json is None:
        plan_json = _extract_json_block_after_header(response_text, "Execution Plan")

    if plan_json is not None:
        try:
            parsed = json.loads(plan_json)
            return _to_sorted_plan_list(parsed)
        except json.JSONDecodeError:
            pass

    return []


def _clamp_box(box: List[float], width: int, height: int) -> List[int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(round(x0)), width - 1))
    y0 = max(0, min(int(round(y0)), height - 1))
    x1 = max(0, min(int(round(x1)), width))
    y1 = max(0, min(int(round(y1)), height))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return [x0, y0, x1, y1]


def crop_image_by_box(image: Image.Image, box: List[float]) -> Image.Image:
    """
    Crop a PIL image using a single [x0, y0, x1, y1] box.
    """
    width, height = image.size
    x0, y0, x1, y1 = _clamp_box(box, width, height)
    return image.crop((x0, y0, x1, y1))


def crop_images_from_dino(
    image: Image.Image,
    dino_result: Dict[str, Any],
    score_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Convert Grounding DINO detections into cropped PIL images.
    Returns a list of {"crop", "box", "score", "label", "index"} dicts.
    """
    crops: List[Dict[str, Any]] = []
    boxes = dino_result.get("boxes", []) or []
    scores = dino_result.get("scores", []) or []
    labels = dino_result.get("labels", []) or []

    for idx, box in enumerate(boxes):
        score = scores[idx] if idx < len(scores) else None
        if score is not None and score < score_threshold:
            continue
        crop = crop_image_by_box(image, box)
        label = labels[idx] if idx < len(labels) else None
        crops.append({
            "index": idx,
            "box": box,
            "score": score,
            "label": label,
            "crop": crop,
        })

    return crops


def _replace_image_path_placeholders(value: Any, image_path: str) -> Any:
    """
    Recursively replace common image path placeholders with the actual image path.
    """
    placeholders = {
        "<image_path>",
        "image_path",
        "{image_path}",
        "<IMAGE_PATH>",
        "IMAGE_PATH",
    }

    if isinstance(value, str):
        return image_path if value.strip() in placeholders else value

    if isinstance(value, list):
        return [_replace_image_path_placeholders(v, image_path) for v in value]

    if isinstance(value, dict):
        return {k: _replace_image_path_placeholders(v, image_path) for k, v in value.items()}

    return value


def _prepare_step_arguments(
    arguments: Dict[str, Any],
    image_path: str | None,
) -> Dict[str, Any]:
    resolved_args = deepcopy(arguments)
    if image_path is not None:
        resolved_args = _replace_image_path_placeholders(resolved_args, image_path)
        resolved_args.setdefault("image_path", image_path)
    return resolved_args


def _strip_non_serializable(value: Any) -> Any:
    if isinstance(value, Image.Image):
        return "<PIL.Image>"
    if isinstance(value, list):
        return [_strip_non_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _strip_non_serializable(v) for k, v in value.items()}
    return value


def execute_execution_plan_batch(
    plan_calls: List[Dict[str, Any]],
    image_paths: List[str] | None = None,
    function_map: Dict[str, Callable] | None = None,
    deduplicate_steps: bool = True,
    batch_size: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Execute a model-provided execution plan across a batch of image paths.
    """
    if function_map is None:
        function_map = FUNCTION_MAP

    if deduplicate_steps:
        unique_plan_calls: List[Dict[str, Any]] = []
        seen = set()
        for step in plan_calls:
            key = json.dumps(step, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            unique_plan_calls.append(step)
        plan_calls = unique_plan_calls

    batch_results: List[Dict[str, Any]] = []

    image_paths = image_paths or []
    max_len = len(image_paths)
    # Initialize per-image result containers
    for image_idx in range(max_len):
        image_path = image_paths[image_idx] if image_idx < len(image_paths) else None
        batch_results.append(
            {
                "image_index": image_idx,
                "image_path": image_path,
                "steps": [],
            }
        )

    for step_idx, step in enumerate(plan_calls):
        func_name = step.get("name")
        step_args = step.get("arguments", {})

        chunk_size = batch_size or max_len
        for start in range(0, max_len, chunk_size):
            end = min(start + chunk_size, max_len)
            image_paths_batch = image_paths[start:end] if image_paths else None

            resolved_args = deepcopy(step_args)
            if image_paths_batch is not None:
                resolved_args = _replace_image_path_placeholders(resolved_args, "image_path")
                resolved_args["image_path"] = image_paths_batch

            logger.debug(
                "[step %s] Executing %s on batch size %s (%s-%s)",
                step_idx + 1,
                func_name,
                end - start,
                start,
                end - 1,
            )

            step_result = execute_tool_call(function_map, func_name, resolved_args)
            step_result = _strip_non_serializable(step_result)

            per_image_results = step_result if isinstance(step_result, list) else [step_result] * (end - start)

            for offset, image_idx in enumerate(range(start, end)):
                arguments_snapshot = _strip_non_serializable(
                    _prepare_step_arguments(
                        step_args,
                        image_paths[image_idx] if image_idx < len(image_paths) else None,
                    )
                )
                batch_results[image_idx]["steps"].append(
                    {
                        "step_index": step_idx + 1,
                        "function": func_name,
                        "arguments": arguments_snapshot,
                        "result": _strip_non_serializable(
                            per_image_results[offset] if offset < len(per_image_results) else None
                        ),
                    }
                )

    return batch_results
                


def pipeline(model: Any, tokenizer: Any, user_query: str, config=None) -> str:
    """
    Main pipeline to process user query, generate model response, parse tool calls, execute tools, and update conversation history.
    """
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant. Use the provided tools to answer the user's question."},
    #     {"role": "user", "content": user_query}
    # ]

    messages = [
        {"role": "system", "content": config.init_prompt},
        {"role": "user", "content": user_query}
    ]


    round_idx = 0

    thinking = config.chat_template.enable_thinking
    max_new_tokens = config.generate.max_new_tokens
    max_tool_calls = config.max_tool_calls
    temperature = config.generate.decode.temperature
    do_sample = config.generate.decode.do_sample
    top_p = config.generate.decode.top_p
    
    while round_idx < max_tool_calls:
        logger.debug(f"Round {round_idx+1}")

        inputs = tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, return_tensors="pt", enable_thinking=thinking).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample, top_p=top_p)

        out_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

        tool_calls_parsed = parse_tool_calls(out_text)

        if not tool_calls_parsed:
            logger.debug(f"No tool calls found, final model response:\n{out_text}")
            return out_text 

        if round_idx == max_tool_calls - 1:
            logger.debug(f"Maximum tool call rounds reached, final model response:\n{out_text}")
            return out_text

        logger.debug(f"Tool calls found: {tool_calls_parsed}")

        tool_calls = [{"type": "function", "function": f} for f in tool_calls_parsed]
        messages.append({"role": "assistant", "tool_calls": tool_calls})

        for i, tool_call in enumerate(tool_calls_parsed):
            func_name = tool_call.get('name')
            args = tool_call.get('arguments', {})

            logger.debug(f"[{i+1}] Executing tool call: {func_name}({args})")
            result = execute_tool_call(FUNCTION_MAP, func_name, args)

            if "error" in result:
                logger.debug(f"{result}, skipping appending it to the conversation history.")
                continue
            elif "result" in result:
                logger.debug(f"Result: {result['result']}")
            else:
                logger.debug(f"Result: {result}")
            

            tool_message = {
                'role': 'tool',
                'name': func_name,
                'content': json.dumps(result),
            }

            if "id" in tool_call:
                tool_message['tool_call_id'] = tool_call['id']
            
            messages.append(tool_message)
        
        round_idx += 1
    