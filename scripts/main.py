from typing import Any, Dict, List
import json
import re
from transformers.utils import get_json_schema
from collections.abc import Callable

from src.tools.object_detection.grounding_dino import run_grounding_dino
from src.tools.classification.bioclip_cls import run_bioclip

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pipeline(user_query: str, thinking: bool=False, max_new_tokens: int=512, max_tool_calls: int=5) -> str:
    """
    Main pipeline to process user query, generate model response, parse tool calls, execute tools, and update conversation history.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools to answer the user's question."},
        {"role": "user", "content": user_query}
    ]

    round_idx = 0

    while round_idx < max_tool_calls:
        print(f"\n{'='*60}")
        print(f"Round {round_idx+1}")
        print(f"{'='*60}")


        inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_tensors="pt", enable_thinking=thinking).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2, do_sample=True, top_p= 0.95)

        out_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

        tool_calls_parsed = parse_tool_calls(out_text)

        if not tool_calls_parsed:
            print("No tool calls found, final model response:\n", out_text)
            return out_text 

        if round_idx == max_tool_calls - 1:
            print("Maximum tool call rounds reached, final model response:\n", out_text)
            return out_text

        print(f"Tool calls found: {tool_calls_parsed}")

        tool_calls = [{"type": "function", "function": f} for f in tool_calls_parsed]
        messages.append({"role": "assistant", "tool_calls": tool_calls})

        for i, tool_call in enumerate(tool_calls_parsed):
            func_name = tool_call.get('name')
            args = tool_call.get('arguments', {})

            print(f"\n [{i+1}] Executing tool call: {func_name}({args})")
            result = execute_tool_call(func_name, args)

            if "error" in result:
                print(f"{result}, skipping appending it to the conversation history.")
                continue
            elif "result" in result:
                print(f"Result: {result['result']}")
            else:
                print(f"Result: {result}")
            

            tool_message = {
                'role': 'tool',
                'name': func_name,
                'content': json.dumps(result),
            }

            if "id" in tool_call:
                tool_message['tool_call_id'] = tool_call['id']
            
            messages.append(tool_message)
        
        round_idx += 1
    

    model_response = pipeline(user_query, max_new_tokens=512)
    print('\nModel response:\n', model_response)

            
            

def execute_tool_call(func_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
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
            match = re.sarch(r"\[TOOL_CALLS\](.*?)(\[/TOOL_CALLS\]|$)", response_text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                if json_str.startwith("["):
                    tool_calls = json.loads(json_str)
                else:
                    tool_calls = [json.loads(json_str)]
        except Exception as e:
            pass
    
    return tool_calls
                

def add(a: float, b: float) -> Dict[str, Any]:
    """
    Add two numbers together.

    Args:
        a: The first number.
        b: The second number.
    
    returns:
        A dictionary containing the result of the addition.
    """
    result = a + b
    return {"operation": "add",
            "operands": [a, b],
            "result": result,
            "expression": f"{a} + {b} = {result}"}


tools: List[Dict[str, Dict]] = [
    get_json_schema(run_grounding_dino),
    get_json_schema(run_bioclip),
    get_json_schema(add)
]


function_map: Dict[str, Callable] = {
    "run_grounding_dino": run_grounding_dino,
    "run_bioclip": run_bioclip,
    "add": add
}


# print("Available tools:\n")
# print(json.dumps(tools, indent=4))


if __name__ == "__main__":


    MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,device_map="auto", dtype=torch.float16, trust_remote_code=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token_id

    print("Model and tokenizer loaded successfully.")

    image_path = 'data/demo.jpg'
    user_query = "Which species is it in this image? Here is the image path: {}.".format(image_path)
    pipeline(user_query, max_new_tokens=512)
    