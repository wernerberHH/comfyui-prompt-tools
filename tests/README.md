# Tests

Light test coverage for the pure-Python helper modules.

## Running

```bash
pip install -r requirements-test.txt
python3 -m pytest tests/unit/ -q
```

## Scope

The current test suite covers only the pure-logic modules:

- `prompts.py` — System-prompt loader (file resolution, placeholder substitution)
- `random_pools.py` — Pool-pick helpers (delimiter handling, age ranges)
- `post_processing.py` — LLM output cleanup (preamble stripping)

Out of scope here (planned for a fuller test suite — see Roadmap):

- `engines/ollama_client.py` — needs a running Ollama or HTTP mock
- `nodes/prompt_helper.py` — full integration test with both engine and prompts
- `nodes/text_mux.py` — trivial; covered transitively

See Roadmap entry "v0.4.0 — Tests & CI" in Contexta (`7ad4f856`) for the full
test suite plan.
