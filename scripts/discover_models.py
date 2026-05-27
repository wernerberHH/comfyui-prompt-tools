#!/usr/bin/env python3
"""Discover available models per provider and print a YAML snippet.

Usage:
    python scripts/discover_models.py                    # all providers
    python scripts/discover_models.py --provider openai  # one provider

Output is a YAML block compatible with `config/endpoints.yaml`. Paste
the block into your `endpoints.yaml` to populate the model dropdowns.

API keys are read from env vars or `config/api_keys.yaml` — see
`config/api_keys.yaml.example` for the resolution order.

Providers without a configured API key are skipped silently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comfyui_prompt_tools.engines import (  # noqa: E402
    PROVIDERS,
    PROVIDER_CHOICES,
)
from comfyui_prompt_tools.engines.model_discovery import (  # noqa: E402
    DiscoveryError,
    discover_models,
)


def _format_yaml_block(results: dict[str, tuple[str, list[str]]]) -> str:
    """Format the per-provider results as a YAML snippet."""
    lines = ["engines:"]
    for provider, (url, models) in results.items():
        lines.append(f"  {provider}:")
        lines.append(f"    - url: \"{url}\"")
        if models:
            lines.append("      models:")
            for m in models:
                lines.append(f"        - \"{m}\"")
        else:
            lines.append("      models: []")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        help="Discover only the given provider (default: all)",
    )
    args = parser.parse_args()

    targets = [args.provider] if args.provider else PROVIDER_CHOICES
    results: dict[str, tuple[str, list[str]]] = {}

    print(f"Discovering models for: {', '.join(targets)}", file=sys.stderr)
    print(file=sys.stderr)

    for provider in targets:
        spec = PROVIDERS[provider]
        try:
            models = discover_models(provider)
            results[provider] = (spec["default_url"], models)
            tag = f"{len(models)} models" if models else "skipped"
            print(f"  {provider:<8}  {tag}", file=sys.stderr)
        except DiscoveryError as exc:
            print(f"  {provider:<8}  FAILED: {exc}", file=sys.stderr)

    if not results:
        print("\nNo results.", file=sys.stderr)
        return 1

    print(file=sys.stderr)
    print("--- YAML snippet for endpoints.yaml ---", file=sys.stderr)
    print(_format_yaml_block(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
