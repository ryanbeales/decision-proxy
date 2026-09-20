"""
Core Decision Engine: translates Jev decision primitives into single-token
logprob requests sent to any OpenAI-compatible backend.
"""

import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx

from .config import settings
from .models import QuestionDefinition, SystemOneAnswer, UsageInfo


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

        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        qtype = q_def.type
        instructions = q_def.instructions
        criteria = q_def.criteria

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
                f"Context / State:\n{state_text}\n\n"
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
                f"Context / State:\n{state_text}\n\n"
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
                f"Context / State:\n{state_text}\n\n"
                f"Scoring Rubric:\n{instructions}\n"
                + "\n".join(options_lines) + "\n\n"
                f"Assign a score from ({', '.join(labels)}):"
            )
            target_map = {lbl: lbl for lbl in labels}

        # 2. Execute upstream API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
