"""
Pydantic schemas for TypeSafe Jev API and direct decision endpoints.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# --- TypeSafe / Jev API Schemas (/v1/systemone) ---

class QuestionDefinition(BaseModel):
    type: Literal["choice", "noul", "score"]
    instructions: str
    criteria: Optional[Union[Dict[str, Any], List[str]]] = None
    labels: Optional[List[str]] = None


class SystemOneRequest(BaseModel):
    state: Union[str, Dict[str, Any], List[Any]]
    model: Optional[str] = None
    questions: Dict[str, QuestionDefinition]
    options: Optional[Dict[str, Any]] = None


class UsageInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: Optional[int] = 0


class SystemOneAnswer(BaseModel):
    type: Literal["choice", "noul", "score"]
    choice: Optional[str] = None
    noul: Optional[float] = None
    score: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    reasoning: Optional[str] = None


class SystemOneResponse(BaseModel):
    answers: Dict[str, SystemOneAnswer]
    usage: UsageInfo
    model: str


# --- Direct Decision API Schemas (/v1/decision) ---

class DecisionRequest(BaseModel):
    context: Union[str, Dict[str, Any], List[Any]] = Field(..., description="Context, document, state or user query")
    question: str = Field(..., description="Question or classification objective")
    type: Literal["choice", "noul", "score"] = Field("choice", description="Decision primitive type")
    options: Optional[List[str]] = Field(None, description="Candidate options for 'choice' primitive")
    rubric: Optional[Union[Dict[str, str], List[str]]] = Field(None, description="Criteria rubric or level descriptions")
    scale_max: Optional[int] = Field(5, description="Max integer score for 'score' primitive")
    model: Optional[str] = Field(None, description="Override upstream model name")
    enable_thinking: Optional[bool] = Field(None, description="Allow limited chain-of-thought reasoning before deciding")
    max_thinking_tokens: Optional[int] = Field(None, description="Token budget for thinking")


class DecisionResponse(BaseModel):
    type: Literal["choice", "noul", "score"]
    selected: Union[str, bool, float]
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    reasoning: Optional[str] = None
    usage: UsageInfo
    model: str
