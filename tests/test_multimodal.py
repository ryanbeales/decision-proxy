"""
Unit tests for multimodal decision-proxy capabilities.
"""

from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from app.engine import extract_images_and_state, normalize_image
from app.main import app

client = TestClient(app)

SAMPLE_BASE64_IMG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_normalize_image():
    # URL
    url_norm = normalize_image("https://example.com/photo.jpg")
    assert url_norm["type"] == "image_url"
    assert url_norm["image_url"]["url"] == "https://example.com/photo.jpg"

    # Data URI
    uri_norm = normalize_image(SAMPLE_BASE64_IMG)
    assert uri_norm["type"] == "image_url"
    assert uri_norm["image_url"]["url"] == SAMPLE_BASE64_IMG

    # Dict with url
    dict_norm = normalize_image({"url": "https://example.com/photo.png", "detail": "low"})
    assert dict_norm["type"] == "image_url"
    assert dict_norm["image_url"]["url"] == "https://example.com/photo.png"
    assert dict_norm["image_url"]["detail"] == "low"

    # Dict with image_url
    dict_norm2 = normalize_image({"image_url": {"url": "https://example.com/pic.jpg"}})
    assert dict_norm2["type"] == "image_url"
    assert dict_norm2["image_url"]["url"] == "https://example.com/pic.jpg"


def test_extract_images_and_state():
    # Base64 in state should be replaced with placeholder
    text, imgs = extract_images_and_state(SAMPLE_BASE64_IMG)
    assert text == "[Image provided]"
    assert len(imgs) == 1
    assert imgs[0]["image_url"]["url"] == SAMPLE_BASE64_IMG

    # Explicit images with clean text state
    text, imgs = extract_images_and_state("User profile picture", [SAMPLE_BASE64_IMG])
    assert text == "User profile picture"
    assert len(imgs) == 1


@pytest.mark.asyncio
async def test_direct_decision_multimodal_noul():
    mock_upstream_response = {
        "choices": [
            {
                "message": {"content": "Y"},
                "logprobs": {
                    "content": [
                        {
                            "token": "Y",
                            "logprob": -0.05,
                            "top_logprobs": [
                                {"token": "Y", "logprob": -0.05},
                                {"token": "N", "logprob": -3.2},
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 60, "completion_tokens": 1},
    }

    with patch("app.engine.DecisionEngine._post_chat_completion", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = (mock_upstream_response, 0.20)

        payload = {
            "image": SAMPLE_BASE64_IMG,
            "question": "Is this image safe for work (SFW)?",
            "type": "noul",
        }

        resp = client.post("/v1/decision", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "noul"
        assert data["selected"] is True
        assert data["confidence"] > 0.90
        assert data["latency_ms"] == 200.0

        # Verify that messages sent to upstream included the image_url content block
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        messages = call_kwargs["messages"]
        user_msg = messages[1]
        assert isinstance(user_msg["content"], list)
        assert any(block.get("type") == "image_url" for block in user_msg["content"])


@pytest.mark.asyncio
async def test_systemone_multimodal_choice():
    mock_upstream_response = {
        "choices": [
            {
                "message": {"content": "A"},
                "logprobs": {
                    "content": [
                        {
                            "token": "A",
                            "logprob": -0.01,
                            "top_logprobs": [
                                {"token": "A", "logprob": -0.01},
                                {"token": "B", "logprob": -4.6},
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 80, "completion_tokens": 1},
    }

    with patch("app.engine.DecisionEngine._post_chat_completion", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = (mock_upstream_response, 0.18)

        payload = {
            "images": [SAMPLE_BASE64_IMG],
            "questions": {
                "content_type": {
                    "type": "choice",
                    "instructions": "Identify image visual category",
                    "criteria": {
                        "Illustration": "Drawn or digital art",
                        "Photo": "Real-world photograph",
                    },
                }
            },
        }

        resp = client.post("/v1/systemone", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "content_type" in data["answers"]
        ans = data["answers"]["content_type"]
        assert ans["type"] == "choice"
        assert ans["choice"] == "Illustration"
        assert ans["probabilities"]["Illustration"] > 0.95
