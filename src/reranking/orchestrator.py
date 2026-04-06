from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import re
import os
import tempfile

import numpy as np
import torch

from src.tools.classification.bioclip_cls import run_bioclip
from src.tools.object_detection.grounding_dino import run_grounding_dino
from src.tools.search.web_search import run_web_search
from src.utils import load_clip


@dataclass
class RerankingPlan:
    original_prompt: str
    rewritten_prompt: str
    has_jargon: bool
    used_web_search: bool
    use_object_detection: bool
    detector_labels: List[str]
    jargon_terms: List[str]


@dataclass
class RerankedImage:
    rank: int
    image_path: str
    clip_score: float
    detector_score: float
    bioclip_score: float
    top_species: str


@dataclass
class RerankingResult:
    plan: Dict[str, Any]
    ranked_images: List[Dict[str, Any]]
    stats: Dict[str, Any]


class EcologicalRerankingOrchestrator:
    """Tool-only ecological reranking pipeline.

    Pipeline:
    1) Detect ecological jargon in text prompt.
    2) If jargon exists, use web search to rewrite to visual prompt.
    3) Decide whether open-vocabulary detection helps.
    4) If yes, run Grounding DINO to filter irrelevant images.
    5) Run BioCLIP as fine-grained species filter.
    6) Run CLIP text-image similarity for final reranking.
    """

    def __init__(
        self,
        clip_model_name: str = "bioclip",
        clip_batch_size: int = 64,
        detector_threshold: float = 0.30,
        bioclip_threshold: float = 0.10,
        max_detector_labels: int = 8,
        device: Optional[str] = None,
    ) -> None:
        self.clip_model_name = clip_model_name
        self.clip_batch_size = clip_batch_size
        self.detector_threshold = detector_threshold
        self.bioclip_threshold = bioclip_threshold
        self.max_detector_labels = max_detector_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model, self.clip_preprocess, self.clip_tokenizer = load_clip(
            clip_model_name,
            use_jit=False,
            device=self.device,
        )

    def run(self, prompt: str, image_paths: List[str], raw_images: List, top_k: int = 50) -> RerankingResult:
        if not image_paths:
            return RerankingResult(plan={}, ranked_images=[], stats={"error": "No image paths provided"})
        if len(image_paths) != len(raw_images):
            return RerankingResult(
                plan={},
                ranked_images=[],
                stats={"error": "image_paths and raw_images must have the same length"},
            )

        jargon_terms = self._find_jargon_terms(prompt)
        has_jargon = bool(jargon_terms)

        rewritten_prompt = prompt
        used_web_search = False
        search_results: List[Dict[str, Any]] = []
        if has_jargon:
            web_payload = run_web_search(prompt, max_results=5)
            search_results = web_payload.get("results", [])
            rewritten_prompt = self._rewrite_prompt_with_search(prompt, jargon_terms, search_results)
            used_web_search = True

        use_object_detection = self._should_use_object_detection(rewritten_prompt, has_jargon)
        detector_labels = self._build_detector_labels(rewritten_prompt, jargon_terms)
        plan = RerankingPlan(
            original_prompt=prompt,
            rewritten_prompt=rewritten_prompt,
            has_jargon=has_jargon,
            used_web_search=used_web_search,
            use_object_detection=use_object_detection,
            detector_labels=detector_labels,
            jargon_terms=jargon_terms,
        )

        candidates = [
            {
                "uid": idx,
                "image_path": image_path,
                "raw_image": raw_image,
            }
            for idx, (image_path, raw_image) in enumerate(zip(image_paths, raw_images))
        ]

        detector_scores = {candidate["uid"]: 0.0 for candidate in candidates}
        bioclip_scores = {candidate["uid"]: 0.0 for candidate in candidates}
        bioclip_species = {candidate["uid"]: "" for candidate in candidates}

        with tempfile.TemporaryDirectory(prefix="eco_rerank_") as tmp_dir:
            for candidate in candidates:
                candidate["tool_path"] = self._ensure_tool_path(
                    image_path=candidate["image_path"],
                    raw_image=candidate["raw_image"],
                    temp_dir=tmp_dir,
                    idx=candidate["uid"],
                )

            detector_candidates = list(candidates)
            if use_object_detection and detector_labels:
                detector_candidates = []
                for candidate in candidates:
                    try:
                        detection = run_grounding_dino(image_path=candidate["tool_path"], text_labels=[detector_labels])
                        scores = detection.get("scores", []) or []
                        det_score = float(max(scores)) if scores else 0.0
                        detector_scores[candidate["uid"]] = det_score
                        if det_score >= self.detector_threshold:
                            detector_candidates.append(candidate)
                    except Exception:
                        detector_scores[candidate["uid"]] = 0.0

                # Avoid empty candidate set due to strict detector threshold.
                if not detector_candidates:
                    detector_candidates = list(candidates)

            bioclip_candidates: List[Dict[str, Any]] = []
            for candidate in detector_candidates:
                try:
                    pred = run_bioclip(candidate["tool_path"])
                    if not pred:
                        continue
                    top_species, top_score = max(pred.items(), key=lambda x: x[1])
                    top_score = float(top_score)
                    relevance_bonus = self._species_name_overlap_score(top_species, rewritten_prompt)
                    score = (0.8 * top_score) + (0.2 * relevance_bonus)

                    bioclip_scores[candidate["uid"]] = score
                    bioclip_species[candidate["uid"]] = top_species
                    if score >= self.bioclip_threshold:
                        bioclip_candidates.append(candidate)
                except Exception:
                    continue

            if not bioclip_candidates:
                bioclip_candidates = list(detector_candidates)

            clip_scores = self._compute_clip_scores(rewritten_prompt, bioclip_candidates)
        reranked = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)
        reranked = reranked[: min(top_k, len(reranked))]

        candidate_by_uid = {candidate["uid"]: candidate for candidate in candidates}

        ranked_images: List[Dict[str, Any]] = []
        for rank, (uid, clip_score) in enumerate(reranked, start=1):
            candidate = candidate_by_uid[uid]
            ranked_images.append(
                asdict(
                    RerankedImage(
                        rank=rank,
                        image_path=candidate["image_path"],
                        clip_score=float(clip_score),
                        detector_score=float(detector_scores.get(uid, 0.0)),
                        bioclip_score=float(bioclip_scores.get(uid, 0.0)),
                        top_species=bioclip_species.get(uid, ""),
                    )
                )
            )

        stats = {
            "num_input_images": len(image_paths),
            "num_after_detector": len(detector_candidates),
            "num_after_bioclip": len(bioclip_candidates),
            "num_ranked": len(ranked_images),
            "used_web_search": used_web_search,
            "num_search_results": len(search_results),
        }

        return RerankingResult(plan=asdict(plan), ranked_images=ranked_images, stats=stats)

    def _find_jargon_terms(self, prompt: str) -> List[str]:
        patterns = [
            r"\b[A-Z][a-z]+idae\b",
            r"\b[A-Z][a-z]+iformes\b",
            r"\b[A-Z][a-z]+ales\b",
            r"\belytra\b",
            r"\bpronotum\b",
            r"\bcerambycidae\b",
            r"\bcoleoptera\b",
            r"\blepidoptera\b",
            r"\bneotropical\b",
            r"\bhabitus\b",
        ]
        found: List[str] = []
        for pattern in patterns:
            found.extend(re.findall(pattern, prompt, flags=re.IGNORECASE))
        return self._dedupe(found)

    def _rewrite_prompt_with_search(
        self,
        prompt: str,
        jargon_terms: List[str],
        search_results: List[Dict[str, Any]],
    ) -> str:
        mapped_terms: List[str] = []
        for term in jargon_terms:
            mapped_terms.extend(self._taxon_to_visual_terms(term))

        for item in search_results[:5]:
            snippet = str(item.get("snippet", ""))
            mapped_terms.extend(self._extract_visual_terms(snippet))

        mapped_terms.extend(self._extract_visual_terms(prompt))
        mapped_terms = self._dedupe(mapped_terms)
        if not mapped_terms:
            return prompt
        return f"Visual description: {', '.join(mapped_terms[:14])}. Original query: {prompt}"

    def _should_use_object_detection(self, rewritten_prompt: str, has_jargon: bool) -> bool:
        if has_jargon:
            return True
        keywords = [
            "with",
            "showing",
            "contains",
            "tagged",
            "markings",
            "pattern",
            "stripe",
            "spotted",
            "nest",
            "shell",
            "fruit",
            "wing",
            "antenna",
        ]
        text = rewritten_prompt.lower()
        return any(key in text for key in keywords)

    def _build_detector_labels(self, rewritten_prompt: str, jargon_terms: List[str]) -> List[str]:
        labels: List[str] = []
        for term in jargon_terms:
            labels.extend(self._taxon_to_visual_terms(term))

        labels.extend(self._extract_visual_terms(rewritten_prompt))
        labels.extend(self._fallback_labels(rewritten_prompt))
        return self._dedupe(labels)[: self.max_detector_labels]

    def _taxon_to_visual_terms(self, term: str) -> List[str]:
        table = {
            "coleoptera": ["beetle", "insect", "hard wing covers"],
            "cerambycidae": ["longhorn beetle", "long antennae", "beetle"],
            "lepidoptera": ["moth", "butterfly", "insect"],
            "aves": ["bird", "beak", "wing"],
            "mammalia": ["mammal", "fur", "body"],
            "fungi": ["mushroom", "fungus"],
            "plantae": ["plant", "leaf", "flower"],
        }
        return table.get(term.lower(), [])

    def _extract_visual_terms(self, text: str) -> List[str]:
        if not text:
            return []

        terms: List[str] = []
        lower = text.lower()
        patterns = [
            r"\b(?:long|short|large|small|red|green|blue|yellow|black|white|brown|dark|light|spotted|striped)\s+[a-z]+(?:\s+[a-z]+)?",
            r"\b[a-z]+\s+(?:wing|wings|antennae|antenna|tail|shell|fur|feathers|leaf|leaves|flower|fruit|beak)\b",
            r"\b(?:beetle|moth|butterfly|bird|frog|crab|sloth|whale|condor|flower|plant|reef|nest|shell)\b",
        ]
        for pattern in patterns:
            terms.extend(re.findall(pattern, lower))
        return self._dedupe(terms)

    def _fallback_labels(self, text: str) -> List[str]:
        text = text.lower()
        labels: List[str] = []
        if "sloth" in text:
            labels.extend(["sloth", "mammal", "tree"])
        if "beetle" in text or "coleoptera" in text:
            labels.extend(["beetle", "insect", "long antennae"])
        if "crab" in text:
            labels.extend(["crab", "shell"])
        if "frog" in text:
            labels.extend(["frog", "amphibian"])
        if "bird" in text or "condor" in text:
            labels.extend(["bird", "wing", "beak"])
        if "flower" in text or "plant" in text:
            labels.extend(["flower", "plant", "fruit"])
        return labels

    def _species_name_overlap_score(self, species_name: str, prompt: str) -> float:
        species_tokens = set(re.findall(r"[a-z]+", species_name.lower()))
        prompt_tokens = set(re.findall(r"[a-z]+", prompt.lower()))
        if not species_tokens or not prompt_tokens:
            return 0.0
        overlap = len(species_tokens.intersection(prompt_tokens))
        return overlap / max(1, len(species_tokens))

    def _compute_clip_scores(self, text_query: str, candidates: List[Dict[str, Any]]) -> Dict[int, float]:
        if not candidates:
            return {}

        text_tokens = self.clip_tokenizer(text_query).to(self.device)
        with torch.no_grad():
            text_emb = self.clip_model.encode_text(text_tokens).squeeze().cpu()
            text_emb /= text_emb.norm(dim=-1, keepdim=True)

        scores: Dict[int, float] = {}
        for start in range(0, len(candidates), self.clip_batch_size):
            batch_candidates = candidates[start:start + self.clip_batch_size]
            try:
                batch_pixels = torch.cat(
                    [
                        self.clip_preprocess(candidate["raw_image"].convert("RGB")).unsqueeze(0)
                        for candidate in batch_candidates
                    ]
                ).to(self.device)
            except Exception:
                # If a batch contains a bad image, fall back to per-image processing.
                for candidate in batch_candidates:
                    try:
                        pixel = self.clip_preprocess(candidate["raw_image"].convert("RGB")).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            image_emb = self.clip_model.encode_image(pixel).cpu()
                            image_emb /= image_emb.norm(dim=-1, keepdim=True)
                        score = float((image_emb.squeeze().float() @ text_emb.float()).item())
                        scores[candidate["uid"]] = score
                    except Exception:
                        scores[candidate["uid"]] = -1.0
                continue

            with torch.no_grad():
                image_embs = self.clip_model.encode_image(batch_pixels).cpu()
                image_embs /= image_embs.norm(dim=-1, keepdim=True)

            batch_scores = (image_embs.float() @ text_emb.float()).numpy()
            for candidate, score in zip(batch_candidates, batch_scores):
                scores[candidate["uid"]] = float(score)

        return scores

    def _ensure_tool_path(self, image_path: str, raw_image: Any, temp_dir: str, idx: int) -> str:
        if isinstance(image_path, str) and image_path and os.path.exists(image_path):
            return image_path

        temp_path = os.path.join(temp_dir, f"img_{idx}.jpg")
        raw_image.convert("RGB").save(temp_path)
        return temp_path

    def _dedupe(self, items: List[str]) -> List[str]:
        seen = set()
        output: List[str] = []
        for item in items:
            norm = item.strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            output.append(norm)
        return output