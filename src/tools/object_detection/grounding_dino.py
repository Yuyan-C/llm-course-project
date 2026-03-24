from transformers import AutoProcessor, GroundingDinoForObjectDetection
from PIL import Image
import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List

def run_grounding_dino(image_path: str, text_labels: List) -> Dict[str, Any]:
    """
    Run Grounding DINO object detection on the given image and text labels.
    Args:
        image_path: The path to the input image.
        text_labels: The text labels to be used for object detection.
    Returns:
        dict: A dictionary containing the detected objects and their corresponding bounding boxes, labels, and scores.
    """
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny", use_fast=True)
    model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to("cuda")
    model.eval()
    image =  Image.open(image_path)

    inputs = processor(images=image, text=text_labels, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.3,
        text_threshold=0.6,
        target_sizes=[image.size[::-1]]
    )

    result = results[0]

    return result


   
   



