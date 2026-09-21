"""
FastAPI application for decision-proxy:
A high-speed Jev-compatible proxy using single-token logprobs against OpenAI-compatible backends.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .config import settings
from .engine import DecisionEngine
from .models import (
    DecisionRequest,
    DecisionResponse,
    QuestionDefinition,
    SystemOneAnswer,
    SystemOneRequest,
    SystemOneResponse,
    UsageInfo,
)

engine = DecisionEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup check
    print(f"🚀 decision-proxy started on port {settings.port}")
    print(f"   Upstream Endpoint: {settings.upstream_endpoint}")
    print(f"   Upstream Model:    {settings.upstream_model}")
    yield


app = FastAPI(
    title="decision-proxy",
    description="High-speed, Jev-compatible decision proxy using 1-token logprobs on OpenAI-compatible backends",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz", tags=["Health"])
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "upstream_endpoint": settings.upstream_endpoint,
        "upstream_model": settings.upstream_model,
        "default_thinking": settings.default_enable_thinking,
    }


@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "decision-proxy",
        "description": "Drop-in TypeSafe / Jev compatible decision proxy using 1-token logprobs",
        "endpoints": {
            "typesafe_systemone": "/v1/systemone",
            "direct_decision": "/v1/decision",
            "health": "/healthz",
        },
        "upstream_model": settings.upstream_model,
    }


@app.post(
    "/v1/systemone",
    response_model=SystemOneResponse,
    tags=["Jev Compatibility"],
    summary="Drop-in TypeSafe Jev systemone endpoint",
)
async def systemone_endpoint(req: SystemOneRequest):
    """
    Executes Jev decision primitives (choice, noul, score) against the upstream backend.
    """
    try:
        answers: Dict[str, SystemOneAnswer] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_tokens = 0

        opts = req.options or {}
        enable_thinking = opts.get("enable_thinking")
        max_thinking = opts.get("max_thinking_tokens")

        for q_key, q_def in req.questions.items():
            ans, usage, _ = await engine.solve_question(
                state=req.state if req.state is not None else "",
                q_key=q_key,
                q_def=q_def,
                enable_thinking=enable_thinking,
                max_thinking_tokens=max_thinking,
                model_override=req.model,
                images=req.images,
            )
            answers[q_key] = ans
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            total_cached_tokens += (usage.cached_tokens or 0)

        return SystemOneResponse(
            answers=answers,
            usage=UsageInfo(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cached_tokens=total_cached_tokens,
            ),
            model=req.model or settings.upstream_model,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Decision execution failed: {str(e)}",
        )


@app.post(
    "/v1/decision",
    response_model=DecisionResponse,
    tags=["Direct Decision"],
    summary="Convenient direct decision endpoint",
)
async def direct_decision_endpoint(req: DecisionRequest):
    """
    Direct endpoint for quick classification, guardrails (noul), or rubric scoring.
    """
    try:
        q_def = QuestionDefinition(
            type=req.type,
            instructions=req.question,
            criteria=req.rubric,
            labels=req.options,
        )

        images = []
        if req.image:
            images.append(req.image)
        if req.images:
            images.extend(req.images)

        ans, usage, latency = await engine.solve_question(
            state=req.context if req.context is not None else "",
            q_key="decision",
            q_def=q_def,
            enable_thinking=req.enable_thinking,
            max_thinking_tokens=req.max_thinking_tokens,
            model_override=req.model,
            images=images if images else None,
        )

        selected: Any
        if req.type == "noul":
            selected = (ans.noul or 0.0) >= 0.5
            confidence = (ans.noul or 0.0) if selected else (1.0 - (ans.noul or 0.0))
        elif req.type == "score":
            selected = ans.score or 0.0
            confidence = max((ans.probabilities or {}).values()) if ans.probabilities else 1.0
        else:
            selected = ans.choice or ""
            confidence = (ans.probabilities or {}).get(selected, 0.0)

        return DecisionResponse(
            type=req.type,
            selected=selected,
            confidence=round(confidence, 4),
            probabilities=ans.probabilities or {},
            latency_ms=round(latency * 1000, 1),
            reasoning=ans.reasoning,
            usage=usage,
            model=req.model or settings.upstream_model,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Direct decision execution failed: {str(e)}",
        )
