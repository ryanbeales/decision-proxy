"""
Core Decision Engine: translates Jev decision primitives into single-token
logprob requests sent to any OpenAI-compatible backend.
"""

import base64
import json
import math
import mimetypes
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx

from .config import settings
from .models import QuestionDefinition, SystemOneAnswer, UsageInfo


def normalize_image(img: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalizes diverse image formats (URL, base64 data URI, raw base64, local file path, or dict)
    into a standard OpenAI-compatible image_url content block.
    """
    if isinstance(img, dict):
        if "image_url" in img and isinstance(img["image_url"], dict):
            return {"type": "image_url", "image_url": img["image_url"]}
        if "image_url" in img and isinstance(img["image_url"], str):
            return {"type": "image_url", "image_url": {"url": img["image_url"]}}
        if "url" in img and isinstance(img["url"], str):
            detail = img.get("detail", "auto")
            return {"type": "image_url", "image_url": {"url": img["url"], "detail": detail}}
        if img.get("type") == "image_url" and "image_url" in img:
            return img
        if "base64" in img:
            mime = img.get("mime_type", "image/jpeg")
            return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img['base64']}"}}
        return {"type": "image_url", "image_url": {"url": str(img)}}

    if isinstance(img, str):
        img_str = img.strip()
        if img_str.startswith("http://") or img_str.startswith("https://") or img_str.startswith("data:image/"):
            return {"type": "image_url", "image_url": {"url": img_str}}

        # Check if local file exists
        if os.path.isfile(img_str):
            mime, _ = mimetypes.guess_type(img_str)
            mime = mime or "image/jpeg"
            with open(img_str, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
            return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}

        # Raw base64 string
        if len(img_str) > 64 and re.match(r"^[A-Za-z0-9+/=\s]+$", img_str[:128]):
            clean_b64 = re.sub(r"\s+", "", img_str)
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean_b64}"}}

        return {"type": "image_url", "image_url": {"url": img_str}}

    return {"type": "image_url", "image_url": {"url": str(img)}}


def extract_images_and_state(
    state: Union[str, Any],
    images: Optional[List[Union[str, Dict[str, Any]]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extracts images from explicit arguments or embedded inside state,
    and returns a clean textual state representation to avoid injecting massive base64 payloads into prompts.
    """
    normalized: List[Dict[str, Any]] = []
    if images:
        for im in images:
            if im:
                normalized.append(normalize_image(im))

    state_text = ""
    # Check if state itself is an image
    if isinstance(state, str):
        s = state.strip()
        if s.startswith("data:image/") or (s.startswith(("http://", "https://")) and any(s.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif"])):
            normalized.append(normalize_image(s))
            state_text = "[Image provided]"
        elif not images and len(s) > 128 and re.match(r"^[A-Za-z0-9+/=\s]+$", s[:128]):
            # Possible raw base64 image in state
            normalized.append(normalize_image(s))
            state_text = "[Image provided]"
        else:
            state_text = state
    elif isinstance(state, dict):
        if "image_url" in state or "image" in state or "url" in state:
            im_val = state.get("image_url") or state.get("image") or state.get("url")
            if im_val:
                normalized.append(normalize_image(im_val))
                other_keys = {k: v for k, v in state.items() if k not in ("image_url", "image", "url", "base64")}
                state_text = json.dumps(other_keys, ensure_ascii=False) if other_keys else "[Image provided]"
            else:
                state_text = json.dumps(state, ensure_ascii=False)
        else:
            state_text = json.dumps(state, ensure_ascii=False)
    elif isinstance(state, list):
        filtered_list = []
        for item in state:
            if isinstance(item, dict) and (item.get("type") == "image_url" or "image_url" in item or "image" in item):
                normalized.append(normalize_image(item))
            elif isinstance(item, str) and (item.startswith("data:image/") or any(item.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"])):
                normalized.append(normalize_image(item))
            else:
                filtered_list.append(item)
        if len(filtered_list) != len(state):
            state_text = json.dumps(filtered_list, ensure_ascii=False) if filtered_list else "[Image provided]"
        else:
            state_text = json.dumps(state, ensure_ascii=False)
    else:
        state_text = str(state) if state is not None else ""

    return state_text, normalized


class DecisionEngine:
    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.endpoint = (endpoint or settings.upstream_endpoint).rstrip("/")
        self.model = model or settings.upstream_model
        self.api_key = api_key or settings.upstream_api_key
        self.timeout = timeout_seconds or settings.request_timeout_seconds

    async def _post_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1,
        enable_thinking: bool = False,
        model_override: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Sends an async chat completion request to the OpenAI-compatible backend."""
        url = f"{self.endpoint}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload: Dict[str, Any] = {
            "model": model_override or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": settings.top_logprobs,
        }

        if not enable_thinking:
            # Tell reasoning models (Qwen, Bonsai, DeepSeek, etc.) not to emit thinking tokens first
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            latency = time.perf_counter() - t0
            resp.raise_for_status()
            data = resp.json()

        return data, latency

    def _extract_token_logprobs(self, choice_obj: Dict[str, Any]) -> Dict[str, float]:
        """Extracts and cleans candidate token probabilities from choice logprobs."""
        token_probs: Dict[str, float] = {}
        logprob_data = choice_obj.get("logprobs") or {}
        content_logprobs = logprob_data.get("content") or []

        if not content_logprobs:
            return token_probs

        # Use the first decision token (or last token if thinking preceded it)
        target_token_data = content_logprobs[0]
        for item in target_token_data.get("top_logprobs", []):
            token_clean = item.get("token", "").strip().upper()
            lp = item.get("logprob")
            if lp is not None and not math.isnan(lp):
                prob = math.exp(lp)
                token_probs[token_clean] = token_probs.get(token_clean, 0.0) + prob

        return token_probs

    async def solve_question(
        self,
        state: Union[str, Any],
        q_key: str,
        q_def: QuestionDefinition,
        enable_thinking: Optional[bool] = None,
        max_thinking_tokens: Optional[int] = None,
        model_override: Optional[str] = None,
        images: Optional[List[Union[str, Dict[str, Any]]]] = None,
    ) -> Tuple[SystemOneAnswer, UsageInfo, float]:
        """Solves a single Jev question (choice, noul, or score) using logprob extraction."""
        use_thinking = (
            enable_thinking
            if enable_thinking is not None
            else settings.default_enable_thinking
        )
        thinking_budget = (
            max_thinking_tokens
            if max_thinking_tokens is not None
            else settings.default_max_thinking_tokens
        )
        max_tokens = (thinking_budget + 16) if use_thinking else 1

        state_text, normalized_images = extract_images_and_state(state, images)
        qtype = q_def.type
        instructions = q_def.instructions
        criteria = q_def.criteria

        state_block = f"Context / State:\n{state_text}\n\n" if state_text and state_text != "[Image provided]" else ""

        # 1. Format prompt by question type
        if qtype == "choice":
            # Extract candidate labels
            if q_def.labels:
                labels = [str(x) for x in q_def.labels]
            elif isinstance(criteria, dict):
                labels = list(criteria.keys())
            elif isinstance(criteria, list):
                labels = [str(i) for i in range(len(criteria))]
            else:
                labels = ["A", "B"]

            letters = [chr(65 + i) for i in range(len(labels))]
            options_lines = []
            for letter, lbl in zip(letters, labels):
                desc = criteria.get(lbl, "") if isinstance(criteria, dict) else ""
                desc_str = f": {desc}" if desc else ""
                options_lines.append(f"{letter}) {lbl}{desc_str}")

            system_prompt = (
                "You are a high-speed, calibrated decision classifier.\n"
                "Respond ONLY with the single best choice letter corresponding to the winning option."
            )
            user_prompt = (
                f"{state_block}"
                f"Objective / Instructions:\n{instructions}\n\n"
                f"Options:\n" + "\n".join(options_lines) + "\n\n"
                f"Select the single best option letter ({', '.join(letters)}):"
            )
            target_map = {letter: lbl for letter, lbl in zip(letters, labels)}

        elif qtype == "noul":
            labels = ["no", "yes"]
            desc_true = criteria.get("true", "") if isinstance(criteria, dict) else ""
            desc_false = criteria.get("false", "") if isinstance(criteria, dict) else ""

            system_prompt = (
                "You are a low-latency boolean decision evaluator.\n"
                "Respond ONLY with Y for Yes, or N for No."
            )
            user_prompt = (
                f"{state_block}"
                f"Policy / Condition to evaluate:\n{instructions}\n"
                f"Condition for Yes: {desc_true}\n"
                f"Condition for No:  {desc_false}\n\n"
                f"Is this condition satisfied? Answer Y for Yes, N for No:"
            )
            target_map = {"Y": "yes", "N": "no"}

        elif qtype == "score":
            if q_def.labels:
                labels = [str(x) for x in q_def.labels]
            elif isinstance(criteria, dict):
                labels = [str(k) for k in criteria.keys()]
            elif isinstance(criteria, list):
                labels = [str(i) for i in range(len(criteria))]
            else:
                labels = ["1", "2", "3", "4", "5"]

            options_lines = []
            for lbl in labels:
                desc = criteria.get(lbl, "") if isinstance(criteria, dict) else ""
                options_lines.append(f"{lbl}: {desc}")

            system_prompt = (
                "You are a calibrated rubric scoring evaluator.\n"
                "Respond ONLY with the single integer score."
            )
            user_prompt = (
                f"{state_block}"
                f"Scoring Rubric:\n{instructions}\n"
                + "\n".join(options_lines) + "\n\n"
                f"Assign a score from ({', '.join(labels)}):"
            )
            target_map = {lbl: lbl for lbl in labels}

        # 2. Execute upstream API call
        if normalized_images:
            user_content: Union[str, List[Dict[str, Any]]] = [
                {"type": "text", "text": user_prompt},
                *normalized_images,
            ]
        else:
            user_content = user_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        data, latency = await self._post_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            enable_thinking=use_thinking,
            model_override=model_override,
        )

        choice_obj = data["choices"][0]
        raw_content = choice_obj.get("message", {}).get("content") or ""
        reasoning_text = (
            choice_obj.get("message", {}).get("reasoning_content")
            or getattr(choice_obj.get("message", {}), "reasoning_content", None)
        )

        # 3. Extract and normalize probabilities
        token_probs = self._extract_token_logprobs(choice_obj)

        if qtype == "noul":
            p_yes = token_probs.get("Y", 1e-4) + token_probs.get("YES", 0.0)
            p_no = token_probs.get("N", 1e-4) + token_probs.get("NO", 0.0)
            total = p_yes + p_no
            p_yes_norm = round(p_yes / total, 4)
            p_no_norm = round(p_no / total, 4)
            probs = {"yes": p_yes_norm, "no": p_no_norm}
            answer = SystemOneAnswer(
                type="noul",
                noul=p_yes_norm,
                probabilities=probs,
                reasoning=reasoning_text,
            )
        elif qtype == "score":
            matched = {lbl: token_probs.get(lbl, 1e-4) for lbl in labels}
            total = sum(matched.values()) or 1.0
            probs = {k: round(v / total, 4) for k, v in matched.items()}
            # Ordinal Expected Value
            ev = sum(float(k) * v for k, v in probs.items())
            answer = SystemOneAnswer(
                type="score",
                score=round(ev, 2),
                probabilities=probs,
                reasoning=reasoning_text,
            )
        else:  # choice
            matched = {lbl: token_probs.get(letter, 1e-4) for letter, lbl in target_map.items()}
            total = sum(matched.values()) or 1.0
            probs = {k: round(v / total, 4) for k, v in matched.items()}
            winner = max(probs, key=probs.get)
            answer = SystemOneAnswer(
                type="choice",
                choice=winner,
                probabilities=probs,
                reasoning=reasoning_text,
            )

        raw_usage = data.get("usage", {})
        usage = UsageInfo(
            input_tokens=raw_usage.get("prompt_tokens", 0),
            output_tokens=raw_usage.get("completion_tokens", 1),
            cached_tokens=raw_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        )

        return answer, usage, latency
