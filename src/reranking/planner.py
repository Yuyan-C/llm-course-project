from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import re

from src.tools.search.web_search import run_web_search


_JARGON_PATTERNS = [
    r"\b[A-Z][a-z]+idae\b",
    r"\b[A-Z][a-z]+ales\b",
    r"\b[A-Z][a-z]+iformes\b",
    r"\b[a-z]+[- ]?toed\b",
    r"\belytra\b",
    r"\bantenna(e|l)?\b",
    r"\bcerambycid\b",
    r"\bcoleoptera\b",
    r"\blepidoptera\b",
    r"\bhabitus\b",
    r"\btarsus\b",
    r"\bpronotum\b",
    r"\bthorax\b",
    r"\bneotropical\b",
]

_COMMON_VISUAL_TERMS = {
    "beetle",
    "moth",
    "butterfly",
    "insect",
    "bird",
    "mammal",
    "plant",
    "flower",
    "leaf",
    "fruit",
    "fungus",
    "sloth",
    "frog",
    "snake",
    "lizard",
    "fish",
    "shell",
    "tree",
    "bark",
    "wing",
    "antenna",
    "horn",
    "spine",
    "stripe",
    "spotted",
    "long antennae",
    "red elytra",
    "dark coloration",
}

_TAXON_TO_COMMON = {
    "Coleoptera": ["beetle"],
    "Cerambycidae": ["longhorn beetle"],
    "Lepidoptera": ["moth", "butterfly"],
    "Aves": ["bird"],
    "Mammalia": ["mammal"],
    "Reptilia": ["reptile"],
    "Amphibia": ["amphibian"],
    "Plantae": ["plant"],
    "Fungi": ["fungus", "mushroom"],
}


@dataclass
class RerankingPlan:
    original_prompt: str
    rewritten_prompt: str
    needs_jargon_rewrite: bool
    needs_object_detection: bool
    detector_labels: List[str] = field(default_factory=list)
    jargon_terms: List[str] = field(default_factory=list)
    reasoning: str = ""
    search_results: List[Dict[str, Any]] = field(default_factory=list)


class EcologicalRerankingPlanner:
    """Plan reranking actions for ecological prompts."""

    def __init__(self, max_detector_labels: int = 8) -> None:
        self.max_detector_labels = max_detector_labels

    def plan(
        self,
        prompt: str,
        model: Any = None,
        tokenizer: Any = None,
    ) -> RerankingPlan:
        if model is not None and tokenizer is not None:
            llm_plan = self._plan_with_llm(prompt, model, tokenizer)
        else:
            llm_plan = self._heuristic_plan(prompt)

        if llm_plan.needs_jargon_rewrite:
            search_payload = run_web_search(prompt, max_results=5)
            llm_plan.search_results = search_payload.get("results", [])
            llm_plan.rewritten_prompt = self._rewrite_prompt(prompt, llm_plan.search_results, model, tokenizer)

        llm_plan.detector_labels = self._build_detector_labels(
            llm_plan.rewritten_prompt or prompt,
            llm_plan.jargon_terms,
        )
        return llm_plan

    def _plan_with_llm(self, prompt: str, model: Any, tokenizer: Any) -> RerankingPlan:
        system = (
            "You plan ecological image reranking. Decide whether the prompt contains ecological jargon, "
            "whether open-vocabulary detection would help, and which detector labels to use. "
            "Return JSON only with keys: needs_jargon_rewrite, needs_object_detection, detector_labels, jargon_terms, reasoning."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        raw_text = self._generate_text(model, tokenizer, messages, max_new_tokens=256)
        payload = self._extract_json_object(raw_text)
        if not payload:
            return self._heuristic_plan(prompt)

        jargon_terms = [str(item) for item in payload.get("jargon_terms", []) if str(item).strip()]
        detector_labels = [str(item) for item in payload.get("detector_labels", []) if str(item).strip()]
        return RerankingPlan(
            original_prompt=prompt,
            rewritten_prompt=prompt,
            needs_jargon_rewrite=bool(payload.get("needs_jargon_rewrite", False)),
            needs_object_detection=bool(payload.get("needs_object_detection", False)),
            detector_labels=detector_labels,
            jargon_terms=jargon_terms,
            reasoning=str(payload.get("reasoning", "")),
        )

    def _heuristic_plan(self, prompt: str) -> RerankingPlan:
        jargon_terms = self._find_jargon_terms(prompt)
        needs_jargon_rewrite = bool(jargon_terms)
        needs_object_detection = self._looks_visual(prompt) or needs_jargon_rewrite
        return RerankingPlan(
            original_prompt=prompt,
            rewritten_prompt=prompt,
            needs_jargon_rewrite=needs_jargon_rewrite,
            needs_object_detection=needs_object_detection,
            detector_labels=self._build_detector_labels(prompt, jargon_terms),
            jargon_terms=jargon_terms,
            reasoning="heuristic-fallback",
        )

    def _rewrite_prompt(
        self,
        prompt: str,
        search_results: List[Dict[str, Any]],
        model: Any = None,
        tokenizer: Any = None,
    ) -> str:
        if model is not None and tokenizer is not None:
            search_summary = self._summarize_search_results(search_results)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Rewrite ecological prompts into concise visual descriptions for image retrieval. "
                        "Avoid jargon and return plain text only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original prompt: {prompt}\n\n"
                        f"Web search context:\n{search_summary}\n\n"
                        "Rewrite this as a short visual description for a vision-language model."
                    ),
                },
            ]
            rewritten = self._generate_text(model, tokenizer, messages, max_new_tokens=192)
            cleaned = rewritten.strip().strip('"')
            return cleaned or self._heuristic_rewrite(prompt, search_results)
        return self._heuristic_rewrite(prompt, search_results)

    def _heuristic_rewrite(self, prompt: str, search_results: List[Dict[str, Any]]) -> str:
        plain_terms = []
        for jargon, common_terms in _TAXON_TO_COMMON.items():
            if re.search(rf"\b{re.escape(jargon)}\b", prompt, flags=re.IGNORECASE):
                plain_terms.extend(common_terms)

        plain_terms.extend(self._extract_visual_terms(prompt))
        plain_terms.extend(self._extract_visual_terms(self._summarize_search_results(search_results)))
        ordered_terms = self._dedupe_preserve_order([term for term in plain_terms if term])
        if not ordered_terms:
            return prompt
        return f"Visual description: {', '.join(ordered_terms[:12])}. Original query: {prompt}"

    def _build_detector_labels(self, prompt: str, jargon_terms: List[str]) -> List[str]:
        labels: List[str] = []
        for term in jargon_terms:
            labels.extend(_TAXON_TO_COMMON.get(term, []))

        labels.extend(self._extract_visual_terms(prompt))
        labels.extend(self._fallback_labels(prompt))
        return self._dedupe_preserve_order(labels)[: self.max_detector_labels]

    def _fallback_labels(self, prompt: str) -> List[str]:
        prompt_lower = prompt.lower()
        labels: List[str] = []
        if "beetle" in prompt_lower or "coleoptera" in prompt_lower or "cerambycidae" in prompt_lower:
            labels.extend(["beetle", "long antennae", "hard wing covers"])
        if "sloth" in prompt_lower:
            labels.extend(["sloth", "mammal", "tree"])
        if "bird" in prompt_lower or "aves" in prompt_lower:
            labels.extend(["bird", "wing", "beak"])
        if "flower" in prompt_lower or "plant" in prompt_lower:
            labels.extend(["flower", "leaf", "plant"])
        if "fung" in prompt_lower or "mushroom" in prompt_lower:
            labels.extend(["fungus", "mushroom"])
        return labels

    def _extract_visual_terms(self, text: str) -> List[str]:
        if not text:
            return []
        lower = text.lower()
        candidates: List[str] = []
        for pattern in [
            r"\b(?:long|short|large|small|red|green|blue|yellow|black|white|brown|spotted|striped|dark|light)\s+[a-z]+(?:\s+[a-z]+)?",
            r"\b[a-z]+\s+(?:antennae|wing|wings|legs|eyes|body|head|tail|shell|fur|feathers|leaves|flowers|bark|fruit)\b",
        ]:
            candidates.extend(re.findall(pattern, lower))
        for token in re.split(r"[^a-zA-Z\- ]+", lower):
            token = token.strip()
            if len(token) < 4:
                continue
            if token in _COMMON_VISUAL_TERMS:
                candidates.append(token)
        return self._dedupe_preserve_order(candidates)

    def _find_jargon_terms(self, prompt: str) -> List[str]:
        terms: List[str] = []
        for pattern in _JARGON_PATTERNS:
            terms.extend(re.findall(pattern, prompt, flags=re.IGNORECASE))
        for taxon in _TAXON_TO_COMMON:
            if re.search(rf"\b{re.escape(taxon)}\b", prompt, flags=re.IGNORECASE):
                terms.append(taxon)
        return self._dedupe_preserve_order([term for term in terms if term])

    def _looks_visual(self, prompt: str) -> bool:
        prompt_lower = prompt.lower()
        visual_markers = [
            "with",
            "has",
            "having",
            "long",
            "short",
            "striped",
            "spotted",
            "wing",
            "antenna",
            "leaf",
            "flower",
            "bark",
            "fur",
            "feather",
            "color",
            "shape",
        ]
        return any(marker in prompt_lower for marker in visual_markers)

    def _summarize_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        snippets: List[str] = []
        for result in search_results[:5]:
            title = str(result.get("title", "")).strip()
            snippet = str(result.get("snippet", "")).strip()
            if title or snippet:
                snippets.append(f"- {title}: {snippet}".strip())
        return "\n".join(snippets)

    def _generate_text(self, model: Any, tokenizer: Any, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                top_p=1.0,
            )
            return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        prompt = "\n\n".join(message["content"] for message in messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                obj, _ = decoder.raw_decode(text[match.start():])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        return {}

    def _dedupe_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for item in items:
            key = item.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(item)
        return ordered
