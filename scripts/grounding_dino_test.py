from transformers import AutoProcessor, GroundingDinoForObjectDetection
from PIL import Image
import os
import torchvision
import torch
import cv2
import numpy as np
from PIL import Image

def draw_bounding_boxes(image_path, result, output_path=None):
    """
    Draw bounding boxes on an image based on detection results.
    
    Args:
        image_path: Path to the input image
        result: Dictionary containing "boxes", "scores", and "labels"
        output_path: Optional path to save the result image. If None, displays the image.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]
    
    # Draw each detection
    for box, score, label in zip(boxes, scores, labels):
        box = box.tolist() if hasattr(box, 'tolist') else box
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare text
        confidence = round(score.item() if hasattr(score, 'item') else score, 3)
        text = f"{label} ({confidence})"
        
        # Draw text background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        cv2.rectangle(image, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, text, (x1, y1 - 2), font, font_scale, (0, 0, 0), thickness)
    
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Result saved to {output_path}")
    
    for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
        box = [round(x, 2) for x in box.tolist()]
        print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")

    
 
    return image


def run_grounding_dino(image_path, text_labels):
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
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


   
   



image_path = "/network/scratch/y/yuyan.chen/inquire/train/00261_Animalia_Arthropoda_Insecta_Coleoptera_Cerambycidae_Typocerus_velutinus/4e9c98d9-1fcd-41f3-a1e3-f206982e0210.jpg"

text_labels = [["an insect", "a white flower","a bear"]]
result = run_grounding_dino(image_path, text_labels)
draw_bounding_boxes(image_path, result, output_path="output.jpg")



