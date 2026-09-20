"""
Unit tests for decision-proxy endpoints.
"""

from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "upstream_endpoint" in data
    assert "upstream_model" in data


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "decision-proxy"
    assert "endpoints" in data


@pytest.mark.asyncio
async def test_systemone_choice_mocked():
    mock_upstream_response = {
        "choices": [
            {
                "message": {"content": "B"},
                "logprobs": {
                    "content": [
                        {
                            "token": "B",
                            "logprob": -0.05,
                            "top_logprobs": [
                                {"token": "B", "logprob": -0.05},
                                {"token": "A", "logprob": -3.5},
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 1},
    }

    with patch("app.engine.DecisionEngine._post_chat_completion", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = (mock_upstream_response, 0.15)

        payload = {
            "state": "I was double charged on my card.",
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Classify intent",
                    "criteria": {
                        "Tech Support": "Technical issue",
                        "Billing": "Payment or charge issue",
                    },
                }
            },
        }

        resp = client.post("/v1/systemone", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "answers" in data
        assert "category" in data["answers"]
        ans = data["answers"]["category"]
        assert ans["type"] == "choice"
        assert ans["choice"] == "Billing"
        assert ans["probabilities"]["Billing"] > 0.90


@pytest.mark.asyncio
async def test_direct_decision_noul_mocked():
    mock_upstream_response = {
        "choices": [
            {
                "message": {"content": "Y"},
                "logprobs": {
                    "content": [
                        {
                            "token": "Y",
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "Y", "logprob": -0.1},
                                {"token": "N", "logprob": -2.4},
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 1},
    }

    with patch("app.engine.DecisionEngine._post_chat_completion", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = (mock_upstream_response, 0.12)

        payload = {
            "context": "Customer has receipt and returned within 10 days.",
            "question": "Is customer eligible for refund?",
            "type": "noul",
        }

        resp = client.post("/v1/decision", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "noul"
        assert data["selected"] is True
        assert data["confidence"] > 0.80


@pytest.mark.asyncio
async def test_direct_decision_score_mocked():
    mock_upstream_response = {
        "choices": [
            {
                "message": {"content": "4"},
                "logprobs": {
                    "content": [
                        {
                            "token": "4",
                            "logprob": -0.2,
                            "top_logprobs": [
                                {"token": "4", "logprob": -0.2},
                                {"token": "5", "logprob": -1.5},
                                {"token": "3", "logprob": -2.8},
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 45, "completion_tokens": 1},
    }

    with patch("app.engine.DecisionEngine._post_chat_completion", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = (mock_upstream_response, 0.14)

        payload = {
            "context": "Agent apologized and solved customer issue quickly.",
            "question": "Rate customer satisfaction on a 1-5 scale.",
            "type": "score",
            "scale_max": 5,
        }

        resp = client.post("/v1/decision", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "score"
        assert data["selected"] >= 3.5
        assert "4" in data["probabilities"]

