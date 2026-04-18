# Module 3: Online API Server

This module shows the other side of vLLM: a persistent HTTP server that exposes an OpenAI-compatible API. Where Module 2 processed everything in one shot, this approach lets you send individual or concurrent requests to a running server.

## Starting the server

```bash
make serve
```

The server binds to `0.0.0.0:8000` and serves the OpenAI chat completions API. Health checks are at `/health`.

## Single request

```bash
make client
```

Sends one PII-detection request and prints the result. Good for verifying the server is working before running the full evaluation.

## Concurrent requests

```bash
make client-concurrent
```

Fires all 150 evaluation prompts (50 records x 3 heuristics) concurrently with controlled parallelism. The script uses `asyncio` + `httpx` to manage in-flight requests, and prints throughput metrics you can compare directly with Module 2's batch results.

## When to use which approach

| Approach | Best for | Tradeoff |
|----------|----------|----------|
| **Offline batch** (Module 2) | Large datasets, one-time jobs | Higher throughput, requires loading the full model per run |
| **Online server** (Module 3) | Real-time pipelines, streaming input | Lower latency per request, HTTP overhead at scale |
