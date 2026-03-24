from typing import Any, Dict, List
import json
import re
from transformers.utils import get_json_schema
from collections.abc import Callable

from src.tools.object_detection.grounding_dino import run_grounding_dino
from src.tools.classification.bioclip_cls import run_bioclip

import torch
from transformers import Auto

tools: List[Dict[str, Dict]] = [
    get_json_schema(run_grounding_dino),
    get_json_schema(run_bioclip)
]


function_map: Dict[str, Callable] = {
    "run_grounding_dino": run_grounding_dino,
    "run_bioclip": run_bioclip
}


# print("Available tools:\n")
# print(json.dumps(tools, indent=4))


if __name__ == "__main__":
   

    image_path = "/network/scratch/y/yuyan.chen/inquire/train/00261_Animalia_Arthropoda_Insecta_Coleoptera_Cerambycidae_Typocerus_velutinus/4e9c98d9-1fcd-41f3-a1e3-f206982e0210.jpg"

    text_labels = [["an insect", "a white flower","a bear"]]



    