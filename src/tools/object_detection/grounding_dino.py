from transformers import AutoProcessor, GroundingDinoForObjectDetection
from PIL import Image
import torch
import numpy as np
from typing import Any, Dict, List, Iterable

DINO_PROCESSOR: AutoProcessor | None = None
DINO_MODEL: GroundingDinoForObjectDetection | None = None


def _get_dino_components(device: str = "cuda") -> tuple[AutoProcessor, GroundingDinoForObjectDetection]:
    global DINO_PROCESSOR, DINO_MODEL
    if DINO_PROCESSOR is None:
        DINO_PROCESSOR = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-tiny",
            backend="torchvision",
        )
    if DINO_MODEL is None:
        DINO_MODEL = GroundingDinoForObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).to(device)
        DINO_MODEL.eval()
    return DINO_PROCESSOR, DINO_MODEL


def _normalize_images(
    image_path: str | List[str] | None,
    image: Image.Image | List[Image.Image] | None,
) -> List[Image.Image]:
    if image is None and not image_path:
        raise ValueError("run_grounding_dino requires image_path or image")
    if image is not None:
        return image if isinstance(image, list) else [image]
    if isinstance(image_path, list):
        return [Image.open(path) for path in image_path]
    return [Image.open(image_path)]


def _normalize_text_labels(
    text_labels: str | List[str] | List[List[str]] | None,
    image_count: int,
) -> List[List[str]]:
    if not text_labels:
        return [[] for _ in range(image_count)]

    if isinstance(text_labels, str):
        text_labels = [text_labels]

    # Single-image case: accept list[str] or list[list[str]]
    if image_count == 1:
        if isinstance(text_labels[0], list):
            return [text_labels[0]]
        return [text_labels]  # type: ignore[list-item]

    # Multi-image case: ensure one list of labels per image
    if isinstance(text_labels[0], list):
        if len(text_labels) == 1:
            return [text_labels[0] for _ in range(image_count)]
        if len(text_labels) == image_count:
            return text_labels  # type: ignore[return-value]
        raise ValueError(
            "text_labels length must be 1 or match number of images when nested lists are provided."
        )

    # list[str] provided, repeat for each image
    return [text_labels for _ in range(image_count)]  # type: ignore[list-item]

def run_grounding_dino(
    image_path: str | List[str] | None = None,
    text_labels: List | None = None,
    image: Image.Image | List[Image.Image] | None = None,
    device: str = "cuda",
) -> Dict[str, Any] | List[Dict[str, Any]]:
    """
    An object detection model to detect instances of the specified text labels in the input image. It uses the GroundingDINO model for object detection.
    Args:
        image_path: The path to the input image.
        text_labels: The text labels to be used for object detection.
        image: A PIL Image to use instead of loading from image_path.
        device: The device to run inference on (e.g., "cuda" or "cpu").
    Returns:
        dict: A dictionary containing the detected objects and their corresponding bounding boxes, labels, and scores.
    """
    processor, model = _get_dino_components(device=device)
    images = _normalize_images(image_path, image)
    normalized_labels = _normalize_text_labels(text_labels, len(images))
    inputs = processor(images=images, text=normalized_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.3,
        text_threshold=0.8,
        target_sizes=[img.size[::-1] for img in images],
    )

    formatted: List[Dict[str, Any]] = []
    for result in results:
        result["scores"] = result["scores"].cpu().numpy().tolist()
        result["boxes"] = result["boxes"].cpu().numpy().tolist()
        formatted.append(result)

    if len(formatted) == 1:
        return formatted[0]
    return formatted


if __name__ == "__main__":
    

    image_path = "/network/scratch/y/yuyan.chen/inquire/train/00261_Animalia_Arthropoda_Insecta_Coleoptera_Cerambycidae_Typocerus_velutinus/4e9c98d9-1fcd-41f3-a1e3-f206982e0210.jpg"

    image_paths = [image_path, image_path]
    text_labels = [["an insect", "a white flower","a bear"]]
    result = run_grounding_dino(image_paths, text_labels)

    print(result)
  


