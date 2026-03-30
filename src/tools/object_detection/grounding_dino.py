from transformers import AutoProcessor, GroundingDinoForObjectDetection
from PIL import Image
import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List

def run_grounding_dino(image_path: str, text_labels: List) -> Dict[str, Any]:
    """
    An object detection model to detect instances of the specified text labels in the input image. It uses the GroundingDINO model for object detection.
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
        text_threshold=0.8,
        target_sizes=[image.size[::-1]]
    )

    result = results[0]

    result['scores'] = result['scores'].cpu().numpy().tolist()
    result['boxes'] = result['boxes'].cpu().numpy().tolist()

    return result


if __name__ == "__main__":
    

    image_path = "/network/scratch/y/yuyan.chen/inquire/train/00261_Animalia_Arthropoda_Insecta_Coleoptera_Cerambycidae_Typocerus_velutinus/4e9c98d9-1fcd-41f3-a1e3-f206982e0210.jpg"

    text_labels = [["an insect", "a white flower","a bear"]]
    result = run_grounding_dino(image_path, text_labels)

    print(result)
  


