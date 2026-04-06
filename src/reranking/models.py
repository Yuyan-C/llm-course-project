from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from bioclip import Rank, TreeOfLifeClassifier
from transformers import AutoProcessor, GroundingDinoForObjectDetection


@dataclass
class BioClipPrediction:
    image_path: str
    species_scores: Dict[str, float]

    @property
    def top_score(self) -> float:
        return max(self.species_scores.values()) if self.species_scores else 0.0


class CachedBioClipClassifier:
    """Load BioCLIP once and reuse it across many images."""

    def __init__(self) -> None:
        self.classifier = TreeOfLifeClassifier()

    def predict(self, image_path: str) -> BioClipPrediction:
        predictions = self.classifier.predict(image_path, Rank.SPECIES)
        species_scores = {
            prediction["species"]: float(prediction["score"])
            for prediction in predictions
        }
        return BioClipPrediction(image_path=image_path, species_scores=species_scores)

    def predict_batch(self, image_paths: List[str]) -> List[BioClipPrediction]:
        return [self.predict(image_path) for image_path in image_paths]


class CachedGroundingDinoDetector:
    """Load Grounding DINO once and reuse it across many images."""

    def __init__(self, model_name: str = "IDEA-Research/grounding-dino-tiny", device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def detect(self, image_path: str, text_labels: List[str]) -> Dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3,
            text_threshold=0.8,
            target_sizes=[image.size[::-1]],
        )

        result = results[0]
        result["scores"] = result["scores"].cpu().numpy().tolist()
        result["boxes"] = result["boxes"].cpu().numpy().tolist()
        return result
