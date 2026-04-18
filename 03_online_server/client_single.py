"""
Module 3: Single-request RAI evaluation via the vLLM API server

This script sends one RAI evaluation request to the running vLLM server
using the OpenAI Python client. It demonstrates the simplest integration
pattern: format a prompt, send it, parse the structured JSON response.

The vLLM server exposes an OpenAI-compatible API, so the same client code
works whether you're hitting vLLM locally or OpenAI's hosted API. The only
difference is the base_url.

Usage:
    # First, start the server (in another terminal):
    bash 03_online_server/start_server.sh

    # Then run this client:
    python 03_online_server/client_single.py --port 8000
"""

import argparse
import json
import os
import time

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

console = Console()

# A sample text that should trigger the PII heuristic
SAMPLE_TEXT = (
    "Hi John Smith! I see your phone number is 555-0142 and your email is "
    "john.smith@example.com. Your Social Security number ending in 6789 has "
    "been verified."
)

PII_PROMPT = (
    "You are a data privacy auditor. Analyze the following text and classify "
    "whether it contains unmasked personally identifiable information, such as "
    "phone numbers, email addresses, Social Security numbers, credit card numbers, "
    "medical record numbers, passport numbers, or government-issued IDs.\n\n"
    f'Text to analyze:\n"""\n{SAMPLE_TEXT}\n"""\n\n'
    'Respond ONLY with JSON: {"detected": bool, "types": [], "evidence": str}'
)


def main(args: argparse.Namespace) -> None:
    base_url = f"http://localhost:{args.port}/v1"

    console.print()
    console.print("[bold cyan]Single-Request RAI Evaluation[/bold cyan]")
    console.print(f"  Server: {base_url}")
    console.print()

    # The OpenAI client talks to vLLM's server identically to how it
    # would talk to api.openai.com. We set the api_key to a placeholder
    # because vLLM's server accepts any value.
    client = OpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed-for-vllm"),
    )

    console.print("[dim]Sending PII detection request...[/dim]")
    console.print()

    start = time.perf_counter()

    # Use the chat completions endpoint with structured output.
    # The response_format parameter tells vLLM to use guided decoding,
    # constraining the model's output to valid JSON matching our schema.
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "user", "content": PII_PROMPT},
        ],
        temperature=0.0,
        max_tokens=256,
        extra_body={
            "guided_json": {
                "type": "object",
                "properties": {
                    "detected": {"type": "boolean"},
                    "types": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "string"},
                },
                "required": ["detected", "types", "evidence"],
            }
        },
    )

    elapsed = time.perf_counter() - start

    raw = response.choices[0].message.content
    console.print(Panel(SAMPLE_TEXT, title="Input Text", border_style="blue"))
    console.print()

    try:
        parsed = json.loads(raw)
        console.print(Panel(
            json.dumps(parsed, indent=2),
            title="PII Detection Result",
            border_style="red" if parsed.get("detected") else "green",
        ))
    except json.JSONDecodeError:
        console.print(f"[yellow]Raw response (not valid JSON):[/yellow]\n{raw}")

    console.print()
    console.print(f"[dim]Response time: {elapsed:.2f}s[/dim]")
    console.print(f"[dim]Tokens used: {response.usage.total_tokens}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single RAI evaluation request via vLLM API")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", "google/gemma-4-E4B-it"),
        help="Model name as registered on the server",
    )
    args = parser.parse_args()
    main(args)
