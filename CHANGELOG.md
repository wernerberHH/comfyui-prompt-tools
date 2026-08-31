# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Two new PromptComposer output styles for Pony Diffusion XL:
  `Pony photoreal` (photoreal merges) and `Pony anime/illustrious`
  (Illustrious XL anime models). Both emit tag-based prompts anchored on
  the `score_*` quality tags; they differ in the appended style tags and
  in the `source_anime` tag.

## [1.1.4] — 2026-06-16

### Added

- New PromptHelper mode `ltx2_video` for LTX-2.3 audio-video generation:
  expands a short motion intent into a single flowing prompt covering
  scene, explicit motion, camera behaviour, and synchronized audio
  (ambient tone, foley, optional speech and music).
- New PromptComposer style `composer_ltx2` producing the same LTX-2.3
  prompt format at the end of a specialist pipeline.

## [1.1.3] — 2026-06-04

### Changed

- System-prompt wording reviewed for a general audience — the shared
  output rules now read as neutral formatting guidance.
- Model-family routing is now configurable without editing code. The
  built-in defaults (`qwen3-vl`, `qwen3`, `gemma`, `llama`) stay in
  code; site-specific mappings live in a new gitignored
  `config/model_families.yaml` and are prepended to the defaults, so a
  custom entry wins over (or extends) them. A missing file, missing
  `pyyaml`, or malformed YAML degrades gracefully to the built-ins.

### Added

- `config/model_families.yaml.example` documenting the model-family
  mapping schema, plus loader unit tests covering custom-pattern
  loading, prepend precedence, graceful degradation, and cache refresh.

## [1.1.2] — 2026-06-02

### Changed

- Package description rewritten to be more motivating and informative on
  the Comfy Registry node page: now leads with a pitch, names the five
  supported backends (Ollama, vLLM, OpenAI, Claude via OpenRouter,
  Gemini) and highlights the four node families.

### Added

- Discovery keywords: `flux`, `prompt-generation`, `image-to-prompt`,
  `vision-language-model`.

## [1.1.1] — 2026-05-28

### Added

- Project icon (`docs/ComfyUIToolsLogo.png`, 400×400 PNG) referenced
  from `pyproject.toml` `[tool.comfy].Icon` so it displays on the
  Comfy Registry node page.

## [1.1.0] — 2026-05-27

First public release. The v1.0 internal tag was cleaned up for public
distribution: internal documentation and any references to the author's
private infrastructure were removed, and the system prompt loader was
refactored so user customisations survive `git pull`.

### Added

- `system_prompts/*.txt.example` fallback in the prompt loader. The
  committed templates now ship as `.txt.example` and the loader
  resolves each prompt as `<name>.txt` (user-editable, gitignored)
  with a fallback to `<name>.txt.example` (shipped). This mirrors the
  existing `config/endpoints.yaml` / `endpoints.yaml.example` pattern
  and means a `git pull` no longer clobbers locally edited prompts.
- Smoke tests against the shipped `endpoints.yaml.example`: ensures it
  parses cleanly, exposes localhost URLs for ollama and vllm, and
  declares the cloud-provider sections so users can fill in API keys
  without restructuring the file.
- 13 unit tests covering the new `.txt` / `.txt.example` resolution and
  override cascade.

### Changed

- All committed system prompts renamed from `*.txt` to `*.txt.example`.
  Existing installations are unaffected — the loader picks up either
  form automatically. New installations get the example as a starting
  point and can copy it to `*.txt` to customise.
- `config/endpoints.yaml.example` rewritten to a generic single-host
  setup (localhost ollama, localhost vllm, three cloud providers with
  empty model lists for the in-app discovery flow). The internal
  multi-host topology comments were removed.
- `DEFAULT_VLLM_MODEL` is now an empty string. There is no longer a
  hard-coded default model name; users pick the model in the node UI
  and save it with their workflow.

### Removed

- Internal documentation that was never meant to ship publicly:
  `CLAUDE.md`, `CLAUDE-CODE-START-HERE.md`, `docs/copyright-checklist.md`,
  `docs/deployment.md`, `docs/memory_strategy.md`, `docs/roadmap.md`,
  `docs/v0.4-refactor-plan.md`, `docs/v0.5-per-model-prompts-plan.md`.
- All shipped per-model system-prompt override files. The override
  cascade itself is unchanged — users can drop their own
  `<mode>.<family>.txt` into `system_prompts/` to activate it.

### Fixed

- Several inline references to the author's private network (IPs in
  test fixtures, a docstring, the CHANGELOG, and the endpoints example
  file) replaced with `localhost`/generic placeholders.

## [1.0.0] — 2026-05-23

First stable release. No functional changes since 0.10.4 — this tag
marks the package as production-ready and adds the documentation,
licensing, and example material needed for a public release.

### Added

- `DISCLAIMER.md`: as-is, no warranty, third-party Terms-of-Service
  responsibility, user owns generated content and provider costs,
  modifications at own risk.
- `examples/` directory with three ready-to-load ComfyUI workflows
  covering the typical use patterns:
    - `z_image_turbo_txt2img_demo.json` — minimal text-to-image with a
      single PromptHelper.
    - `flux2_fashion_v4_demo.json` — two-reference outfit transfer
      using VisionPromptHelper in "Outfit Transfer" mode plus FLUX.2
      `ReferenceLatent` for identity + garment consistency.
    - `video_v1_wan22_i2v_demo.json` — full Wan 2.2 image-to-video
      pipeline exercising all four custom nodes (VisionPromptHelper,
      PromptHelper, PromptComposer, TextMux).
- `docs/screenshots/`: Extensions submenu, PromptTools settings tab,
  and a full Wan 2.2 pipeline overview, all embedded in the README.
- README "Supported LLM providers" matrix listing Ollama, vLLM,
  OpenAI, and Gemini as stable, with OpenRouter (Claude) flagged as
  experimental / untested by the author.
- README "Configuring providers" section explaining the split between
  the Settings UI (recommended for API keys — values land in
  gitignored `api_keys.yaml` with `0600` perms) and `endpoints.yaml`
  (recommended for URLs + model autocomplete).
- README + composer workflow guide: explicit rule of thumb for when
  to wire a helper into `user_instruction` (dominant scene/motion
  intent) versus one of the five `PromptComposer` input slots
  (raw aspect material).
- `pyproject.toml`: author email, search keywords, PyPI classifiers
  (Development Status :: 5 - Production/Stable), Issues + Changelog
  URLs, and an optional `[test]` dependency group.

### Changed

- Status: pre-release → **stable / v1.0**.

### Tests

340 passing (unchanged from 0.10.4).


## [0.10.4] — 2026-05-21

Branch `fix/v0.10.4-gemini-discovery-and-filters`. One discovery bug
fix for Gemini, broader chat-model filters, and two defensive fixes
in the Settings UI / OpenAI client.

### Fixed

- **Gemini model discovery** returned 401. The provider spec combined
  the OpenAI-compatible base URL with the native `/v1beta/models`
  discovery path and sent the API key as a Bearer token, but Google's
  native endpoint rejects Bearer auth (it expects `?key=` or
  `x-goog-api-key`). Discovery now hits the OpenAI-compatible
  `/v1beta/openai/models`, consistent with the chat path. The dead
  `gemini` parser branch was removed.
- **Test All Connections** no longer paints providers green when no
  API key is configured. `discover_models()` silently returns `[]` in
  that case, which the UI used to misread as success. The test now
  short-circuits to `ok=False` with a clear "no API key configured"
  message before discovery is even attempted.
- **OpenAIChatEngine.chat()** no longer crashes with
  `TypeError: 'NoneType' is not subscriptable` when the upstream
  response has missing or `null` choices/message/content. The new
  `_extract_content()` helper raises a meaningful `OpenAIError` for
  each malformed-response case (including the toolproxy bug that
  returns `{"choices": null}` when vLLM emits a 400).

### Changed

- **OpenAI chat-model blacklist** broadened: `codex` and `sora` are
  now substring matches (catch `gpt-5-codex`, `codex-mini-latest`,
  `sora-2`, `sora-2-pro`, future variants).
- **Shared blacklist:** `^text-embedding` is now `embedding`
  (substring), so it also catches `gemini-embedding-*` without a
  duplicate per-provider entry.
- **Gemini-specific blacklist** added: `nano-banana` (Flash Image
  codename) and `deep-research` (specialised research mode).
  `preview` is deliberately not filtered — Google ships active models
  as `*-preview`, so filtering would drop the newest releases.

### Tests

323 → 340 (+17 new, -1 obsolete, 1 renamed).


## [0.10.3] — 2026-05-20

Branch `fix/v0.10.3-discovery-filter-prefix-toast`. Three quality-of-life
fixes for the Settings UI.

### Added

- **Per-provider chat-model filter** in discovery. OpenAI's `/v1/models`
  returns ~100 entries, most of which are not chat-capable. A regex
  blacklist now drops TTS, transcription, audio, realtime, image gen,
  search previews, embeddings, dall-e, whisper, legacy completion
  (babbage/davinci), and gpt-3.5. claude/gemini are not filtered —
  their catalogues are already curated upstream. ollama/vllm are not
  filtered — users curate their own local model sets.
- **`[engine] model` prefix** on every entry in the model dropdown.
  Makes the source backend obvious at a glance. The prefix is purely a
  UI affordance — `_resolve_engine` strips it back out before the
  value reaches the engine client. Bare model names without a prefix
  still work for backward-compat with existing workflows.

### Fixed

- **Save-toast spam on page refresh.** ComfyUI V2 fires `onChange` for
  every persisted setting at page load when it restores values from
  localStorage. That triggered our `scheduleSave` on every refresh,
  producing a wall of "Saved" toasts the user did not initiate. A
  last-known-value cache now suppresses the initial-load echo: the
  first onChange per `(provider, field)` key records the value and
  returns without saving; subsequent calls only fire on real changes.

### Caveat

The engine field is independent of the model prefix. If you pick
`engine=ollama` together with a `[openai]`-prefixed model, the prefix
is silently stripped and the request is routed to ollama (where the
model does not exist and the request fails at runtime). True
prefix-driven auto-routing is a v0.11+ topic.

### Tests

12 new tests (8 filter scenarios, 4 prefix-strip scenarios) for a
total of 323 passing.

## ## [0.10.2] — 2026-05-20

Branch `fix/v0.10.2-top-menu-commands`. Theme: discoverability fix for
the Settings UI actions.

### Fixed

- **Test/Discover actions now visible in the top menu.** In v0.10.1 the
  two actions (`Test All Connections`, `Discover Models`) lived only in
  the command palette, which is not easily reachable in many browsers
  (Ctrl+Shift+P is captured by the print dialog) and not discoverable
  for users who don't know it exists.

  Adds a `menuCommands` registration that surfaces both actions under
  **Extensions → Prompt Tools** in the top menu bar. Backend, settings
  layout, and command palette entries are unchanged.

## ## [0.10.1] — 2026-05-20

Branch `fix/v0.10.1-settings-v2-compat`. Theme: Settings UI rework for
ComfyUI V2 frontend compatibility.

### Fixed

- **`type: "button"` rendered as a text input** in the ComfyUI V2
  frontend (Nodes 2.0). The previous v0.10.0 Settings entry was
  registered as a button that should open a modal dialog — V2 silently
  fell back to a text input, leaving users with no way to reach the
  settings UI. The button-and-modal architecture is replaced by
  inline settings entries.

### Changed

- **Settings layout reworked**: instead of one button that opened a
  modal, the Settings dialog now shows **10 entries** directly under
  the *PromptTools* category — one URL field and one API key field per
  provider (ollama, vllm, openai, claude, gemini). Each entry has its
  own onChange handler and saves to the backend immediately
  (debounced 500ms).
- **Actions moved to the command palette**:
  - `Prompt Tools: Test All Connections`
  - `Prompt Tools: Discover Models`
  Both surface results as toast notifications. Reachable from the
  command menu in V2 and from the keyboard shortcut bar.
- **Removed**: per-provider Test buttons and the discovery status panel
  (both required the modal). Test All gives the same information in
  one go via toast.

### Notes

- API keys live in two places: the browser's setting store
  (localStorage) and `~/.config/comfyui-prompt-tools/api_keys.yaml` on
  the server (chmod 600). Anyone with access to either can read them.
- After upgrading, you may want to re-enter your API keys via the new
  Settings layout to ensure they sync to the yaml file on the server.

## [0.10.0] — 2026-05-19

Branch `feat/v0.10-settings-ui`. Theme: Settings UI for API keys and
model discovery — the user-facing layer on top of v0.9.0's backend.

### Added

- **Settings dialog** opened from a ComfyUI Settings entry
  *"Prompt Tools — API Keys & Discovery"*. One row per provider with
  a configured/missing badge. Cloud providers (openai, claude, gemini)
  show an API-key input (password-masked with a 👁 toggle), a URL
  override input (placeholder shows the default URL), and a *Test*
  button that runs a live discovery call against the saved key. Local
  providers (ollama, vllm) show only the URL field.
- **Global *Save* button**: writes ``config/api_keys.yaml`` with
  ``chmod 600``. Empty fields remove the entry, so a user can clear a
  key without editing YAML by hand. The key is sent only on save; the
  dialog never displays a stored key.
- **Global *Discover Models* button**: triggers discovery for all
  providers and merges the results into ``config/endpoints.yaml``.
  The previous file is copied to ``endpoints.yaml.backup`` first.
  User-set URLs in ``endpoints.yaml`` are preserved — only the
  ``models`` lists are touched.
- **Four HTTP routes** registered on the ComfyUI PromptServer
  (registration is no-op when running pytest without ComfyUI):

  - ``GET  /comfyui-prompt-tools/status`` — per-provider configured
    flag, never returns the key itself
  - ``POST /comfyui-prompt-tools/save``   — write keys + URL overrides
  - ``POST /comfyui-prompt-tools/test``   — validate one provider
  - ``POST /comfyui-prompt-tools/discover`` — run discovery + merge

- **``WEB_DIRECTORY`` in ``__init__.py``** so ComfyUI serves the JS
  extension from ``web/comfyui-prompt-tools.js``.

### UI compatibility

The dialog is a plain HTML overlay rendered via ``document.body``
rather than a ComfyUI Settings-panel embed. It uses CSS variables
(``--comfy-menu-bg``, ``--input-text``) to inherit the active theme.
Works in both classic ComfyUI and the new Nodes 2.0 UI.

### Tests

22 new tests for the four route helpers (status / save / test /
discover) — total 311 passing. The JS layer is exercised manually
because integration testing requires a live ComfyUI server.

### Notes

- After running discovery, ComfyUI needs to be reloaded (page refresh
  or container restart) before the model dropdowns pick up the new
  list. The dialog's status line mentions this.
- The 401 / 403 error messages from v0.9.0 surface through the *Test*
  button as an alert, so the user gets immediate, actionable feedback
  without having to read logs.

## [0.9.0] — 2026-05-19

Branch `feat/v0.9-multi-provider-core`. Theme: multi-provider engine
backend (Backend-Core for Phase 3a; the Settings UI follows in
v0.10.0).

### Added

- **Five engine choices instead of two**: `ollama`, `vllm`, `openai`,
  `claude`, `gemini`. The dropdown now carries a tooltip explaining
  each choice and the required API-key env var.
- **API key resolver** with a 4-step priority chain:
  1. Custom env var `COMFYUI_PROMPT_TOOLS_API_KEY_<PROVIDER>`
  2. Standard env var (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`)
  3. Project-local `config/api_keys.yaml` (gitignored)
  4. User-global `~/.config/comfyui-prompt-tools/api_keys.yaml`
- **Authorization header** sent by `OpenAIChatEngine` when an
  `api_key` is set. Local unauthenticated vLLM keeps working
  (header is omitted when key is `None`).
- **Actionable error messages**: 401 → "Authentication failed, check
  API key". 403 → "API key valid but lacks permission for this model".
- **Model discovery** module (`engines/model_discovery.py`):
  `discover_models(provider)` hits `/v1/models` (or `/api/tags` for
  Ollama) and returns a sorted, deduplicated model list. Three
  response parsers handle openai-style, ollama-style, and gemini-style
  payloads (Gemini strips the `models/` prefix). Providers without a
  configured API key are skipped silently.
- **CLI script** `scripts/discover_models.py`: runs discovery on
  demand and prints a YAML snippet ready to paste into
  `config/endpoints.yaml`. The Settings UI in v0.10.0 will add a
  one-click "Discover Models" button.
- **URL override**: `api_keys.yaml` can also carry an optional `url`
  per provider, in case a provider changes their endpoint.
- **Config examples**: `config/api_keys.yaml.example` documents the
  schema, file permissions (600), and resolution order.
  `config/endpoints.yaml.example` gains openai/claude/gemini sections
  with empty model lists (filled by discovery).

### Changed

- `ENGINE_CHOICES` is now derived from `PROVIDER_CHOICES`. Existing
  workflows with `engine: 'vllm'` continue to work unchanged.
- `_resolve_engine` collapses vllm/openai/claude/gemini into a single
  `OpenAIChatEngine` code path — they differ only in default URL and
  API-key source. Ollama remains a separate code path.

### Tests

64 new tests across 4 files; total 289 passing.

### Notes

- Claude is wired via OpenRouter's OpenAI-compatible API. A native
  Anthropic engine remains a candidate for v1.1 if there is demand.
- The `endpoints.yaml.example` ships with empty model lists for the
  three Cloud providers. Run `scripts/discover_models.py` once to
  populate them, or wait for v0.10.0's "Discover Models" button.

## [0.8.0] — 2026-05-19

Branch `feat/v0.8-composer-lora-keywords`. Theme: LoRA-trigger keyword
field on PromptComposer for stable LoRA activation through the
prompt pipeline.

### Added

- **`lora_keywords` input on `PromptComposer`** — optional STRING
  field, comma-separated. Carries LoRA trigger tokens
  (e.g. `ohwx_man, <lora:cinematic_v2:0.8>, slow motion`) through the
  composer so they survive into the final prompt verbatim.

  Mechanism is a hybrid of "weave" and "post-check":

  1. When the field is non-empty, a *MANDATORY TOKENS* directive is
     appended to the system prompt instructing the LLM to keep the
     tokens verbatim and weave them naturally where contextually
     fitting.
  2. After the LLM call, `_ensure_verbatim_tokens` scans the output
     and appends any missing tokens as a comma-separated suffix.
     This is the safety net — even if the LLM paraphrases or drops a
     trigger, the downstream `CLIPTextEncode` never loses the LoRA
     activation.
  3. `debug_info` reports `LoRA tokens: <count> (<appended> appended)`
     so users can see at a glance whether the safety net kicked in.

  Pass-through semantics: LoRA syntax (`<lora:name:weight>`,
  `(token:1.2)`) is preserved unchanged. Token order is preserved.

  The field defaults to an empty string and is optional — existing
  workflows are not affected.

### Helpers (internal API)

- `_parse_lora_keywords(raw)` — comma-split, whitespace-strip,
  empty-drop, order-preserving.
- `_build_lora_keywords_directive(keywords)` — renders the
  system-prompt suffix; empty input returns empty string for
  unconditional concatenation.
- `_ensure_verbatim_tokens(output, required)` — post-check + append;
  returns `(final_output, missing_tokens)`.

### Tests

28 new tests covering helper functions and end-to-end integration
through `compose()`. Total: 225 passing.

## [0.7.0] — 2026-05-19

Branch `chore/v0.7-cleanup-deprecations`. Theme: deprecation warnings
cleanup ahead of the v1.0 release.

### Fixed

- **Pillow `mode` parameter removed** from `Image.fromarray` in
  `image_io.py`. The `mode` parameter is deprecated and will be removed
  in Pillow 13 (2026-10-15). For 3-channel uint8 NumPy arrays, Pillow
  auto-detects the RGB mode, so the explicit `mode="RGB"` was redundant.

### Known Issues

- **pytest-asyncio cosmetic warning**: If `pytest-asyncio` is installed
  globally on the host, running `pytest -W default` may emit a
  `PytestDeprecationWarning` about an unset
  `asyncio_default_fixture_loop_scope` configuration option. This
  project does not use pytest-asyncio (no declared dependency, no
  `@pytest.mark.asyncio` markers, no `async def test_*`). The warning
  is raised in the plugin's `pytest_configure` hook before
  pyproject.toml-based filters (incl. `filterwarnings`, `-W ignore`,
  `-p no:asyncio`) can take effect, so it cannot be suppressed via
  project config. It is invisible in standard `pytest`/`pytest -q`
  output and harmless. Workaround if needed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/`.

## [0.6.0] — 2026-05-19

Branch `feat/wan22-vision-prompt`. Theme: image-to-video prompt
pipeline support — a holistic image-description vision mode plus a
Wan 2.2 motion composer style. Together they slot the existing
PromptHelper / VisionPromptHelper / PromptComposer trio into a
clean video-prompt workflow without touching any node code.

### Added

- **Vision mode: `Describe Picture`** — holistic single-pass image
  description (4-6 sentences covering subject, clothing, pose,
  setting, lighting, and framing) intended as the upstream half of a
  video-prompt workflow. Sits at the top of the Describe-modes
  block in `VISION_MODE_TO_FILE` (`vision_describe_picture`).
- **Composer output style: `Wan 2.2 motion`** — new style in
  `PromptComposer.OUTPUT_STYLES` that fuses image-description
  snippets and a motion intent into a Wan 2.2 I2V prompt (60-100
  words, active voice, explicit camera, no negation). Mapped to
  `composer_wan22` in `_STYLE_TO_FILE`.

Both additions follow the standard mode-registration pattern from
`docs/adding-a-mode.md` — registry entry + template file + tests,
no node changes. The per-model override cascade applies automatically
(drop a `vision_describe_picture.<family>.txt` or
`composer_wan22.<family>.txt` to specialise per LLM family).

### Notes

- Targets Wan 2.2 native ComfyUI I2V implementation. Verified
  defaults: ModelSamplingSD3 shift 5.0 for both stages, euler/simple
  sampler.
- Lightning LoRA mode (4 steps / cfg 1.0) and Quality mode (20 steps
  / cfg 3.5) are workflow-level concerns, not prompt-level — both
  consume the same composer output.

## [0.5.0] — 2026-05-12

Branch `feat/per-model-system-prompts`. Theme: per-model
system-prompt variants via an override cascade, plus a new Z-Image
Text-to-Image mode and a model-family example entry in
`endpoints.yaml.example`. (The internal v0.5 work brief was retired
when the repository went public.)

### Added

- **Per-model override cascade** — `get_system_prompt(mode,
  model_name=...)` first probes `system_prompts/<mode>.<family>.txt`
  before falling back to the default. The shared `render_template`
  helper means `get_vision_system_prompt` and
  `_load_composer_system_prompt` honour the same cascade and the
  same `{shared_rules}` substitution. Families are detected from a
  module-level `MODEL_FAMILY_PATTERNS` list (case-sensitive,
  first-match-wins).
- **Five per-model overrides** for narrative and creative-variation
  modes:
  - FLUX Kontext (Couple Scene)
  - Qwen Image Edit (Couple Scene)
  - Random Character (Pony)
  - Random Character (Z-Image)
  - Z-Image Text-to-Image
- **New mode: Z-Image Text-to-Image** — plain text-to-image
  variant for Z-Image Turbo. Sits between FLUX Text-to-Image and
  SDXL Photorealistic in the dropdown. Mode count: 9 → 10.
- **Node wiring** — PromptHelper, VisionPromptHelper, and
  PromptComposer now pass the user-selected model to their
  loaders so the cascade actually activates on real LLM calls.
  Custom-system-prompt bypass paths are unchanged.
- **Example model entry in `endpoints.yaml.example`** — listed under
  `localhost:11434` so a fresh install surfaces it in the dropdown
  without further edits (the IP-based duplicate was removed from the
  example file). Header expanded with a network-topology section
  (same-host vs separate-hosts) and a pointer to the override cascade.
- **`docs/system-prompt-overrides.md`** — explains the cascade,
  family detection, naming convention, style guide for writing an
  override, and a worked example.
- **`docs/adding-a-mode.md`** — end-user step-by-step for adding a
  new PromptHelper mode (label, basename, template, registry,
  optional overrides, unit test).
- **Network-topology section in `docs/deployment.md`** —
  Scenario A (same host, with the Docker-bridge gotcha) and
  Scenario B (separate hosts) with ASCII diagrams.

### Changed

- `README.md` — added a "Per-model system prompts" section and a
  Z-Image Text-to-Image row to the mode table.

### Tests

- 185 unit tests, up from 126 at v0.4.0. New coverage:
  - `MODEL_FAMILY_PATTERNS` detection (11 known + 6 unknown
    parametrised cases, pattern-order verification)
  - Cascade behaviour in prompts.py, vision_prompts.py, and the
    composer loader (override wins, default fallback on no-override
    / unknown family / None, shared-rules substitution in
    overrides, KeyError preservation)
  - Node-level model-name propagation in all three nodes
    (PromptHelper, VisionPromptHelper, PromptComposer) and the
    custom-prompt bypass
  - Shipped per-model overrides activate via the cascade and differ
    from defaults; skipped modes fall through identically
  - Anchor-checks for Pony quality tags and Qwen three-image
    tokens to guard the format contracts
  - Smoke test against the real `endpoints.yaml.example` to ensure
    the example model stays listed on both Ollama URLs

### Non-changes

- Defaults for qwen3-vl, qwen3, gemma, and llama families are
  unchanged in v0.5 — only one family ships overrides.
- No vision-mode overrides in v0.5.
- No public-release rebrand to VisualPromptKit — still v1.0
  preparation.

## [0.4.0] — 2026-05-11

Branch `refactor/v0.4-base-and-pipeline`. Theme: shared engine abstraction
+ pipeline-of-specialists architecture. See `docs/v0.4-refactor-plan.md`
for the full plan and `docs/prompt-pipeline-pattern.md` for the design
rationale.

### Added
- **`BasePromptNode`** — shared base class with engine selector (Ollama /
  vLLM), URL/model resolution, and a unified `_call_engine()` helper.
  Subclassed by all helper nodes.
- **`PromptComposer`** node — fuses 1–5 description snippets and a user
  instruction into one final prompt via LLM call. Three output styles:
  `FLUX.2 natural language`, `SDXL tag-based`, `Z-Image compact`. Empty
  inputs are silently dropped before the call.
- **VisionPromptHelper expansion** — from 1 mode to 14 modes, split into
  edit-modes (Outfit Transfer, Hair Change, Body Reshape, Background
  Change, Pose Change, Combined Edit) and describe-modes (Face, Hair,
  Body, Pose, Outfit, Background, Lighting, Composition).
- **`config_loader.py` + `config/endpoints.yaml.example`** — optional
  YAML config for URL / model autocomplete dropdowns. Falls back to
  `.example` if no user config exists; degrades to plain text inputs if
  pyyaml is absent or the file is malformed.
- **`docs/composer-workflow-guide.md`** — worked example of a
  pipeline-of-specialists workflow.
- **`docs/prompt-pipeline-pattern.md`** — design rationale for splitting
  description, editing, and composition into separate nodes.
- **CHANGELOG.md** (this file).

### Changed
- `PromptHelper` — now subclasses `BasePromptNode`. Supports vLLM in
  addition to Ollama via the new `engine` dropdown. Old `ollama_url`
  field is gone — workflows that wired it need to be rebuilt once.
- `VisionPromptHelper` — now subclasses `BasePromptNode`. Inputs renamed
  to `image_1` / `image_2` (was `person_image` / `outfit_image`).
  `backend` / `api_url` fields replaced by the new engine selector.

### Dependencies
- **Added**: `pyyaml>=6.0` for `config/endpoints.yaml` parsing. Optional
  in practice — the loader gracefully degrades to empty config if the
  import fails.

### Tests
- 120 unit tests (up from 23), 85% line coverage.
- All new modules >80% coverage per Standards §4a acceptance.
- Engine HTTP layers patched via `tests/conftest.py` fixtures; no
  network calls.

### Breaking changes
- Workflow JSONs that referenced the old field names (`ollama_url`,
  `backend`, `api_url`, `person_image`, `outfit_image`) need to be
  rebuilt once. Agreed-upon trade-off to make v1.0 a non-event migration.

### Pre-release fix
- `BasePromptNode._build_engine_inputs()` originally hardcoded the
  autocomplete source to ``"ollama"``, so vLLM URLs and models never
  surfaced in the dropdowns. ComfyUI's ``INPUT_TYPES`` is static and
  cannot react to the user's engine choice, so the fix flattens every
  URL across every entry in ``ENGINE_CHOICES`` into one Combo, and every
  model across every (engine, url) pair into another (sorted,
  deduplicated). The user picks a matching ``engine`` + ``base_url`` +
  ``model`` themselves — mismatched combinations surface as a connection
  error at run time. Falls back to plain STRING inputs when the config
  is empty.
- **Edit-mode identity markers** — Bernd's manual Outfit-Transfer tests
  showed that purely referential identity anchors ("the same woman from
  image 1, maintaining her exact facial features") lose identity in the
  generated image. All six edit-mode system prompts
  (`vision_outfit_transfer`, `vision_hair_change`, `vision_body_reshape`,
  `vision_background_change`, `vision_pose_change`,
  `vision_combined_edit`) now instruct the vision LLM to extract
  DESCRIPTIVE markers from image 1 — ethnicity, age, eye shape and
  colour, skin tone, freckles/moles/scars, hair (when not the changed
  aspect), glasses, facial hair, tattoos, piercings, body build (when
  not the changed aspect) — and weave them into the opening sentence.
  Each prompt also carries a USER INTENT OVERRIDES block that tells the
  LLM to drop or modify markers via positive phrasing only ("remove
  glasses" → omit the marker, not "no glasses"). Word-count targets
  raised to 80–130 for the single-aspect modes and 120–180 for
  combined_edit to fit the descriptive identity anchor.

## [0.3.1] — 2026-05-05

- Display-name cleanup (drop emoji).

## [0.3.0] — 2026-05-04

- New `VisionPromptHelper` node (Outfit Transfer mode) with FLUX.2
  Multi-Reference prompt template.
- New `OpenAIChatEngine` for vLLM backend.
- New `image_io` helper for tensor → base64 encoding.
- Initial test suite: 23 unit tests for `prompts.py`, `random_pools.py`,
  `post_processing.py`.

## [0.2.0 and earlier]

Initial migration of `PromptHelper` and `TextMux` from the monolithic
`comfyui-prompt-helper` repo. See `git log` for details.
