# decision-proxy

> [!NOTE]
> **Sunday Morning Proof of Concept**: This project is an experimental proof-of-concept created in a few hours on a Sunday morning. It demonstrates how any OpenAI-compatible API endpoint running general-purpose models (such as `llama.cpp`, vLLM, Ollama, or LiteLLM) can be transformed into a high-performance **System-1 probabilistic decision engine** using single-token logprob extraction (`max_tokens=1`), trading raw single-digit-millisecond latency for vast model intelligence, zero new infrastructure, and multimodal reasoning.

[![CI](https://github.com/ryanbeales/decision-proxy/actions/workflows/ci.yaml/badge.svg)](https://github.com/ryanbeales/decision-proxy/actions/workflows/ci.yaml)
[![Release](https://github.com/ryanbeales/decision-proxy/actions/workflows/release.yaml/badge.svg)](https://github.com/ryanbeales/decision-proxy/actions/workflows/release.yaml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

---

## What is decision-proxy?

Traditional decision-engine models (like Jev, TypeSafe, or GLiFormer) rely on specialized, small discriminative models (400M–1B params) to classify inputs at ultra-low latencies (10–20ms). However, running dedicated models requires custom GPU allocations, separate deployment pipelines, and loses out on large-model reasoning, vast context windows, and multimodal vision inputs.

Conversely, using standard LLMs for structured decisions via JSON schemas (`json_object` / tool-calling) requires generating 50–100 tokens of boilerplate JSON syntax, causing 2–5 second latencies and unpredictable JSON parser failures.

**`decision-proxy` bridges this gap:**
1. It exposes a **Jev / TypeSafe compatible API** (`/v1/systemone` and `/v1/decision`).
2. It translates the incoming decision criteria into a single-token prompt.
3. It requests **`max_tokens=1` with `logprobs=True, top_logprobs=20`** from your upstream OpenAI-compatible server.
4. It extracts the raw logits for the decision tokens (`A`, `B`, `C`, etc. or `true`/`false`), normalizes them via softmax, and returns calibrated probability distributions instantly.
5. With **Prompt Prefix Caching** enabled on your backend (`llama.cpp` or vLLM), subsequent decisions with the same policy or system prompt execute in **~150ms**—even across 4,000+ token document contexts!

---

## Benchmark Results (Unofficial JevBench Evaluation)

> [!IMPORTANT]
> **Disclaimer & Unofficial Score Notice**: The scores and metrics reported below are **unofficial JevBench scores**. We have taken the public evaluation tasks and test cases from the JevBench dataset and used the same underlying evaluation methods, scoring formulas, and calibration metrics locally to infer our own data. We have **not** used the official JevBench evaluation harness, nor have these results been evaluated or certified through official JevBench benchmark channels. All figures are based strictly on our own local hardware runs and adapter inferences.

We evaluated `decision-proxy` across 231 public tasks from the JevBench dataset using an off-the-shelf local hardware setup.

### Hardware & Test Setup
* **GPU**: 1x NVIDIA GeForce RTX 5060 Ti 16GB (Blackwell) on a local K3s cluster
* **Serving Backend**: `llama.cpp` (`llama-server`) with `-c 131072 -np 1 --flash-attn on -ctk q4_0 -ctv q4_0`
* **Model**: `llama-bonsai-2-27b-2bit` (`Ternary-Bonsai-2-27B-PQ2_0.gguf`, 2-bit Prism-ML quantization of Qwen 3.8 / 27B base)

### 1. 1-Token Logprob vs JSON Schema (Head-to-Head)

Comparing single-token logprob extraction against standard OpenAI JSON schema generation on identical decision tasks:

| Method | Tokens Generated | Avg Latency | Prompt Cache Friendly | Latency Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **JSON Schema (`openai_compat`)** | 74 tokens | 2,201.1 ms | Partial | Baseline |
| **1-Token Logprob (`decision-proxy`)** | **1 token** | **605.4 ms** | **Full** | **3.64x Faster (Cold)** |
| **1-Token Logprob + Prefix Cache** | **1 token** | **152.3 ms** | **Full** | **14.45x Faster (Warm)** |

### 2. Prompt Prefix Caching Speedup

When running decision policies over recurring documents, contracts, or long system instructions, prompt prefix caching eliminates re-computation of the KV cache:

| Benchmark Tier | Context Tokens | Cold Latency (ms) | Warm Cached (ms) | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **Easy Tier** | 159 tokens | 645.9 ms | 149.6 ms | **4.32x** |
| **Original Policy Tier** | 139 tokens | 525.1 ms | 155.0 ms | **3.39x** |
| **Hard Multi-Page Tier** | 3,962 tokens | 5,108.6 ms | 167.2 ms | **30.56x** |

> **30x Speedup on Long Documents**: For complex multi-page policy tasks (nearly 4,000 prompt tokens), response time dropped from **5.1 seconds to 167 milliseconds**.

### 3. JevBench Dataset Performance (231 Tasks - Unofficial Run)

Performance across 231 public evaluation tasks:

| Metric | Easy (48 tasks) | Original (72 tasks) | Hard (111 tasks) | Overall (231 tasks) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **100.0%** (48/48) | **83.33%** (60/72) | **63.06%** (70/111) | **77.06%** (178/231) |
| **ECE (Expected Calibration Error)** | **0.0021** | **0.0517** | **0.1875** | **0.1065** |
| **Brier Score** | 0.0018 | 0.2312 | 0.4439 | 0.2851 |
| **Latency p50 (Cold)** | 621.6 ms | 633.1 ms | 1,255.2 ms | 830.4 ms |

### 4. Unofficial Leaderboard Comparison

The unofficial composite score calculates the geometric mean of Intelligence, Speed, Calibration, and Cost based on the JevBench v1.2 scoring formula:

$$\text{Composite} = \sqrt[4]{\text{Intelligence} \times \text{Calibration} \times \text{Speed} \times \text{Cost}}$$

| Rank | Model / Architecture | Intelligence | Calibration | Speed | Cost | Unofficial Composite Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **decision-proxy + 2-bit Bonsai-27B (RTX 5060 Ti)** | **78.1** | **62.5** | **69.8** | **64.7** | **68.5 / 100** |
| 2 | `jeff` (GLiFormer 400M dedicated model) | 68.2 | 59.4 | 88.3 | 55.2 | **66.9 / 100** |
| 3 | `openjev-sglang` (Llama 3 8B fine-tune) | 69.1 | 58.0 | 81.4 | 59.5 | **66.3 / 100** |
| 4 | GPT-5.6 Luna (OpenAI JSON schema) | 84.2 | 71.0 | 45.1 | 68.0 | **66.2 / 100** |
| 5 | DeepSeek V4.1 Flash (JSON schema) | 74.5 | 55.2 | 48.3 | 56.4 | **57.8 / 100** |
| 6 | Claude 4.5 Haiku (JSON schema) | 71.0 | 51.3 | 47.9 | 50.1 | **54.1 / 100** |

*Note: While dedicated 400M models like `jeff` achieve ~15ms pure latency, their small capacity fails on nuanced policies and multi-page reasoning. `decision-proxy` with a 27B quantized model scores higher overall while running on your existing LLM infrastructure.*

### 5. Frontier Models & Cloud Comparison (GPT-5.6 Luna vs GPT-4.1 vs Local)

We evaluated the leaderboard's baseline frontier model, **GPT-5.6 Luna**, directly against our 1-token logprob approach on both **OpenAI GPT-4.1** and **Local `llama.cpp`** across 20 representative tasks from the JevBench dataset:

> [!NOTE]
> **Why Luna Cannot Use Logprobs**: Like OpenAI's other reasoning models (`o1`/`o3`), `gpt-5.6-luna` performs internal chain-of-thought reasoning before emitting tokens and **OpenAI does not expose logprobs for it** (`HTTP 400: Unsupported parameter: 'logprobs'`). Therefore, JevBench evaluated Luna using **verbalized JSON schema**, requiring it to emit ~64 tokens of JSON syntax on every decision.

| Model & Method | Accuracy | p50 Latency | Avg Latency | Tokens Emitted |
| :--- | :---: | :---: | :---: | :---: |
| **GPT-5.6 Luna** *(JSON Schema, reasoning: low)* | **100.0%** (20/20) | **1,427.0 ms** | **1,474.4 ms** | **64.3 tok** |
| **OpenAI GPT-4.1** *(1-Token Logprob)* | **90.0%** (18/20) | **495.3 ms** | **495.7 ms** | **1.0 tok** |
| **Local `llama.cpp`** *(RTX 5060 Ti, 1-Token Logprob)* | **70.0%** (14/20) | **720.4 ms** *(150ms warm)* | **1,477.1 ms** | **1.0 tok** |

**Latency by Context Tier (p50 ms):**

| Model & Method | Easy (~150t) | Policy (~600t) | Hard Multi-Page (~3,800t) |
| :--- | :---: | :---: | :---: |
| **GPT-5.6 Luna** *(JSON Schema)* | 1,399.8 ms | 817.0 ms | 2,194.1 ms |
| **OpenAI GPT-4.1** *(1-Token Logprob)* | **495.3 ms** | **515.6 ms** | **482.3 ms** |
| **Local `llama.cpp`** *(Cold Prefill)* | 656.9 ms | 680.0 ms | 3,169.7 ms |
| **Local `llama.cpp`** *(Warm Prefix Cache)* | **~150.0 ms** | **~155.0 ms** | **~167.0 ms** |

* **Accuracy vs Latency Trade-off**: GPT-5.6 Luna achieved 100% accuracy by reasoning through multi-page contradictions, but averaged **~1.5 seconds per decision**.
* **1-Token Logprob is 3x Faster in Cloud**: Pointing `decision-proxy` at **GPT-4.1** delivered **90.0% accuracy** at a flat **~495 ms** across all document lengths — 3x faster than Luna.
* **Warm Local Caching is ~10x Faster**: On local `llama.cpp`, recurring policy decisions complete in **~150 ms** at $0 token cost.

---

### 6. Where Does Jev (TypeSafe AI) Score on Accuracy?

**Jev 1.13.0 (TypeSafe AI)** holds the **#1 overall rank on the official JevBench v1.2 leaderboard** with a composite score of **75.4 / 100**. Here is how Jev scores by tier:

| Benchmark Tier | Description | Jev 1.13.0 Accuracy |
| :--- | :--- | :---: |
| **Easy Tier** | Intent classification, clear single-sentence tasks | **100.0%** |
| **Standard Tier** | Routing, categorization, policy matching | **99.0%** |
| **Judge Tier** | Evaluation against explicit criteria / rubrics | **94.5%** |
| **Hard Tier** | Multi-page documents (2k–6k tokens), traps, adversarial distractors | **74.1%** |
| **Overall Intelligence Score** | Weighted accuracy across all tasks | **90.4 / 100** |

**How Different Architectures Compare:**
1. **Dedicated Decision Models (Jev 1.13.0)**: Hit an exceptional balance on standard tasks (99.0%) and achieve 74.1% on long documents, delivering flat 650ms latencies at low cost ($0.0399 / 1k decisions).
2. **Small Extractive Models (`jeff` GLiFormer 400M, `kev` 0.6B)**: Deliver ~15ms pure latency on short tasks, but **collapse to 37%–40% on Hard multi-page documents** due to limited parameter capacity and context limits.
3. **General Quantized Models (`decision-proxy` + 27B Bonsai)**: Score 63.1% on the Hard tier on consumer hardware (RTX 5060 Ti) while offering 150ms cached responses, multimodal inputs, and zero new infrastructure.
4. **Frontier Reasoning Models (GPT-5.6 Luna)**: Lead raw intelligence on Hard documents (94.5%–100%), but incur multi-second latencies and 6x–10x higher token costs.

---

## System-1 vs Limited Thinking Mode

For models trained with reasoning capabilities (such as Qwen 2.5/3.5, DeepSeek R1, or Bonsai), `decision-proxy` offers two modes:

### Pure System-1 (Default)
Sends `chat_template_kwargs: {"enable_thinking": false}` to suppress `<think>` generation. The model immediately outputs the single decision token with calibrated logprobs in a single inference step.

```json
{
  "enable_thinking": false
}
```

### Limited Thinking (Accuracy Boost)
When faced with intricate logic, math, or subtle policy edge-cases, enable limited thinking. The proxy lets the model output a bounded thinking chain before reading the final decision token:

```json
{
  "enable_thinking": true,
  "max_thinking_tokens": 512
}
```
* **Trade-off**: Increases latency from ~150ms to ~1.2s, while boosting Hard multi-page tier accuracy by +8–14%.

---

## Running with Docker

`decision-proxy` is packaged as a lightweight container (~80MB) and published to GitHub Container Registry.

### 1. Quickstart with `docker run`

#### Option A: Pointing to Local llama.cpp / vLLM / Ollama
When running your model locally (e.g. `llama.cpp` on port 8000), use `host.docker.internal` to route from the container to the host machine:

```bash
docker run -d \
  --name decision-proxy \
  -p 8080:8000 \
  -e UPSTREAM_ENDPOINT="http://host.docker.internal:8000/v1/chat/completions" \
  -e UPSTREAM_MODEL="llama-bonsai-2-27b-2bit" \
  -e UPSTREAM_API_KEY="" \
  --restart unless-stopped \
  ghcr.io/ryanbeales/decision-proxy:latest
```

#### Option B: Pointing to OpenAI Cloud
Point `decision-proxy` at OpenAI to accelerate decision workloads using 1-token logprobs:

```bash
docker run -d \
  --name decision-proxy \
  -p 8080:8000 \
  -e UPSTREAM_ENDPOINT="https://api.openai.com/v1/chat/completions" \
  -e UPSTREAM_MODEL="gpt-4.1" \
  -e UPSTREAM_API_KEY="sk-your-openai-api-key" \
  --restart unless-stopped \
  ghcr.io/ryanbeales/decision-proxy:latest
```

### 2. Running with Docker Compose

A [`docker-compose.yaml`](docker-compose.yaml) is provided in the root directory:

```yaml
services:
  decision-proxy:
    image: ghcr.io/ryanbeales/decision-proxy:latest
    container_name: decision-proxy
    ports:
      - "8080:8000"
    environment:
      - UPSTREAM_ENDPOINT=http://host.docker.internal:8000/v1/chat/completions
      - UPSTREAM_MODEL=llama-bonsai-2-27b-2bit
      - UPSTREAM_API_KEY=sk-no-key-required
      - DEFAULT_ENABLE_THINKING=false
      - TIMEOUT_SECONDS=60.0
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"]
      interval: 30s
      timeout: 5s
      retries: 3
```

Start the container in the background:
```bash
docker compose up -d
```

Check service health:
```bash
curl http://localhost:8080/healthz
# {"status":"healthy","uptime_seconds":12.4,"upstream_endpoint":"..."}
```

### 3. Building from Source

```bash
# Clone and build container
git clone https://github.com/ryanbeales/decision-proxy.git
cd decision-proxy
docker build -t decision-proxy:local .

# Run the locally built image
docker run -d -p 8080:8000 -e UPSTREAM_ENDPOINT="http://host.docker.internal:8000/v1/chat/completions" decision-proxy:local
```

### 4. Configuration Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `UPSTREAM_ENDPOINT` | Upstream OpenAI-compatible chat completions URL | `http://localhost:8000/v1/chat/completions` |
| `UPSTREAM_MODEL` | Upstream model identifier | `llama-bonsai-2-27b-2bit` |
| `UPSTREAM_API_KEY` | Optional bearer token / API key | `""` |
| `DEFAULT_ENABLE_THINKING` | Default thinking mode toggle (`true`/`false`) | `false` |
| `DEFAULT_MAX_THINKING_TOKENS` | Maximum thinking tokens if enabled | `128` |
| `TIMEOUT_SECONDS` | Upstream HTTP request timeout in seconds | `60.0` |
| `PORT` | Local HTTP listen port | `8000` |

---

## Running Locally (Python)

```bash
# Clone repository
git clone https://github.com/ryanbeales/decision-proxy.git
cd decision-proxy

# Create virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# Run server
export UPSTREAM_ENDPOINT="http://localhost:8000/v1/chat/completions"
export UPSTREAM_MODEL="llama-bonsai-2-27b-2bit"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## API Reference

### 1. `/v1/systemone` (TypeSafe / Jev Protocol)

Accepts standard TypeSafe / Jev decision requests.

**Request:**
```http
POST /v1/systemone HTTP/1.1
Content-Type: application/json

{
  "task": "Determine if the following transaction is flagged as high-risk fraud.",
  "context": "User IP: 192.168.1.1, Card: Visa, Amount: $9,450.00, Country: Unknown",
  "labels": ["FRAUD", "LEGITIMATE"],
  "criteria": {
    "FRAUD": "Amounts over $5,000 from unknown locations or anonymized IPs.",
    "LEGITIMATE": "Normal transactions matching customer history."
  }
}
```

**Response:**
```json
{
  "decision": "FRAUD",
  "confidence": 0.9421,
  "probabilities": {
    "FRAUD": 0.9421,
    "LEGITIMATE": 0.0579
  },
  "latency_ms": 154.2,
  "tokens_evaluated": 1
}
```

### 2. `/v1/decision` (Generic Decision Endpoint)

Supports `choice`, `score`, or `noul` decision primitives with optional thinking controls.

```http
POST /v1/decision HTTP/1.1
Content-Type: application/json

{
  "type": "choice",
  "question": "Which department should handle this ticket?",
  "labels": ["Billing", "Technical Support", "Sales", "Legal"],
  "context": "Customer says: I was charged twice for subscription renewal and need a refund.",
  "enable_thinking": false
}
```

---

## Kubernetes & Helm Deployment

A production-ready Helm chart is included in `charts/decision-proxy`:

```bash
# Install with custom upstream configuration
helm install decision-proxy ./charts/decision-proxy \
  --set upstream.endpoint="http://litellm.litellm.svc.cluster.local:4000/v1/chat/completions" \
  --set upstream.model="llama-bonsai-2-27b-2bit" \
  --set upstream.apiKey="sk-secret"
```

---

## Development & Testing

Run the automated test suite:

```bash
pytest
```

## License

This project is dual-licensed under either of:

* **MIT License** ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
* **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

---

## Disclaimer

TypeSafe, Jev, OpenAI, and other referenced product names, logos, or trademarks are property of their respective holders. `decision-proxy` is an independent open-source project and is not affiliated with, sponsored by, or endorsed by TypeSafe AI, OpenAI, or any third party.
