from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import os

from tqdm import tqdm

from .models import CachedBioClipClassifier, CachedGroundingDinoDetector
from .planner import RerankingPlan


@dataclass
class ImageRerankRecord:
    image_path: str
    coarse_score: float = 0.0
    fine_score: float = 0.0
    final_score: float = 0.0
    detector_output: Optional[Dict[str, Any]] = None
    bioclip_output: Optional[Dict[str, float]] = None


class BatchProcessor:
    """Apply planned actions across many images in bounded batches."""

    def __init__(self, batch_size: int = 16, device: Optional[str] = None) -> None:
        self.batch_size = batch_size
        self.detector = CachedGroundingDinoDetector(device=device)
        self.classifier = CachedBioClipClassifier()

    def process(self, image_paths: List[str], plan: RerankingPlan) -> List[ImageRerankRecord]:
        records: List[ImageRerankRecord] = []
        for batch in tqdm(list(self._iter_batches(image_paths)), desc="reranking", leave=False):
            for image_path in batch:
                if not os.path.exists(image_path):
                    continue

                detector_output = None
                coarse_score = 0.0
                if plan.needs_object_detection and plan.detector_labels:
                    detector_output = self.detector.detect(image_path, [plan.detector_labels])
                    scores = detector_output.get("scores", [])
                    coarse_score = float(max(scores)) if scores else 0.0

                bioclip_output = self.classifier.predict(image_path)
                fine_score = bioclip_output.top_score

                final_score = self._combine_scores(coarse_score, fine_score, plan)
                records.append(
                    ImageRerankRecord(
                        image_path=image_path,
                        coarse_score=coarse_score,
                        fine_score=fine_score,
                        final_score=final_score,
                        detector_output=detector_output,
                        bioclip_output=bioclip_output.species_scores,
                    )
                )

        records.sort(key=lambda item: item.final_score, reverse=True)
        return records

    def _combine_scores(self, coarse_score: float, fine_score: float, plan: RerankingPlan) -> float:
        if plan.needs_object_detection:
            return (0.35 * coarse_score) + (0.65 * fine_score)
        return fine_score

    def _iter_batches(self, items: List[str]):
        for start in range(0, len(items), self.batch_size):
            yield items[start:start + self.batch_size]
