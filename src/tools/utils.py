import json
from typing import Any, Dict, List
from collections.abc import Callable
import re

import argparse
import logging
from transformers.utils import get_json_schema

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
                


def pipeline(model: Any, tokenizer: Any, user_query: str, config=None) -> str:
    """
    Main pipeline to process user query, generate model response, parse tool calls, execute tools, and update conversation history.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools to answer the user's question."},
        {"role": "user", "content": user_query}
    ]

    round_idx = 0

    thinking = config['chat_template']["enable_thinking"]
    max_new_tokens = config['generate']["max_new_tokens"]
    max_tool_calls = config["max_tool_calls"]
    temperature, do_sample, top_p = config['generate']['decode']['temperature'], config['generate']['decode']['do_sample'], config['generate']['decode']['top_p']
    
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
    