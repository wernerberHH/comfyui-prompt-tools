// ComfyUI Prompt Tools — Settings UI (V1 + V2 compatible)
//
// Registers 10 text-input settings (URL + API key per provider) directly
// in the ComfyUI Settings dialog under category "PromptTools", plus
// 2 commands ("Test All Connections" and "Discover Models") that
// surface results as toasts.
//
// Security note: API keys live in the browser's setting store
// (localStorage) and in ~/.config/comfyui-prompt-tools/api_keys.yaml on
// the server. Anyone with access to either can read them.

import { app } from "../../scripts/app.js";

const PROVIDERS = ["ollama", "vllm", "openai", "claude", "gemini"];

// Display labels for the settings dialog
const PROVIDER_LABEL = {
    ollama: "Ollama",
    vllm:   "vLLM (local OpenAI-compatible)",
    openai: "OpenAI (ChatGPT)",
    claude: "Claude (via OpenRouter)",
    gemini: "Google Gemini",
};

// Default URLs — kept in sync with comfyui_prompt_tools/engines/provider_defaults.py
const PROVIDER_DEFAULT_URL = {
    ollama: "http://localhost:11434",
    vllm:   "http://localhost:8000/v1",
    openai: "https://api.openai.com/v1",
    claude: "https://openrouter.ai/api/v1",
    gemini: "https://generativelanguage.googleapis.com/v1beta/openai/",
};

const API = {
    status:   "/comfyui-prompt-tools/status",
    save:     "/comfyui-prompt-tools/save",
    test:     "/comfyui-prompt-tools/test",
    discover: "/comfyui-prompt-tools/discover",
};

// ---------------- HTTP helpers -------------------------------------------

async function apiGet(path) {
    const r = await fetch(path, { method: "GET" });
    if (!r.ok) throw new Error(`GET ${path}: HTTP ${r.status}`);
    return r.json();
}

async function apiPost(path, body) {
    const r = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error(`POST ${path}: HTTP ${r.status}`);
    return r.json();
}

// ---------------- Toast helper (V2 + V1 fallback) ------------------------

function toast(severity, summary, detail) {
    // V2 frontend exposes app.extensionManager.toast
    const tm = app?.extensionManager?.toast;
    if (tm && typeof tm.add === "function") {
        tm.add({ severity, summary, detail, life: 6000 });
        return;
    }
    // V1 fallback — plain console + alert for errors
    const line = `[Prompt Tools] ${summary}${detail ? ": " + detail : ""}`;
    if (severity === "error") {
        console.error(line);
        alert(line);
    } else {
        console.log(line);
    }
}

// ---------------- Save (debounced per field) -----------------------------
//
// Each setting onChange fires save() with just that provider + field.
// Backend `/save` accepts partial updates and merges into the yaml.
// Debounce so rapid typing doesn't spam the server.

const _saveTimers = {};

// v0.10.3: ComfyUI V2 fires onChange for every setting at page-load when
// it restores persisted values. That triggered our scheduleSave on every
// refresh, producing a wall of "Saved" toasts the user did not initiate.
// We use a last-known-value cache: the first call per key just records
// the value and returns (it is the initial-load echo). Subsequent calls
// only fire when the value differs from the last cached one.
const _lastKnown = {};

function scheduleSave(provider, field, value) {
    const key = `${provider}.${field}`;

    // Initial-load echo — record and bail without saving.
    if (!(key in _lastKnown)) {
        _lastKnown[key] = value;
        return;
    }
    // No real change — also bail.
    if (_lastKnown[key] === value) return;
    _lastKnown[key] = value;

    clearTimeout(_saveTimers[key]);
    _saveTimers[key] = setTimeout(async () => {
        try {
            const body = { [provider]: { [field]: value } };
            const r = await apiPost(API.save, body);
            if (r.ok) {
                toast("success", "Saved", `${PROVIDER_LABEL[provider]} ${field}`);
            } else {
                toast("error", "Save failed", r.error || "unknown");
            }
        } catch (e) {
            toast("error", "Save failed", String(e));
        }
    }, 500);
}

// ---------------- Commands -----------------------------------------------

async function testAllConnections() {
    toast("info", "Testing connections", "running…");
    const results = [];
    for (const p of PROVIDERS) {
        try {
            const r = await apiPost(API.test, { provider: p });
            results.push(
                `${PROVIDER_LABEL[p]}: ${r.ok ? "OK" : "✗ " + (r.error || "failed")}`
            );
        } catch (e) {
            results.push(`${PROVIDER_LABEL[p]}: ✗ ${e}`);
        }
    }
    toast("info", "Test results", results.join("\n"));
}

async function discoverModels() {
    toast("info", "Discovering models", "this may take a few seconds…");
    try {
        const r = await apiPost(API.discover, {});
        if (!r.ok) {
            toast("error", "Discovery failed", r.error || "unknown");
            return;
        }
        const lines = Object.entries(r.results || {}).map(
            ([p, info]) => `${PROVIDER_LABEL[p]}: ${info.count} models`
        );
        const skipped = (r.skipped || []).map(
            ([p, reason]) => `${PROVIDER_LABEL[p]}: skipped (${reason})`
        );
        toast(
            "success",
            "Discovery complete",
            [...lines, ...skipped].join("\n") || "no providers reachable",
        );
    } catch (e) {
        toast("error", "Discovery failed", String(e));
    }
}

// ---------------- Settings builder ---------------------------------------

function buildSettings() {
    const settings = [];

    for (const provider of PROVIDERS) {
        // URL field
        settings.push({
            id:           `PromptTools.${provider}.URL`,
            category:     ["PromptTools", PROVIDER_LABEL[provider], "URL"],
            name:         `${PROVIDER_LABEL[provider]} — URL`,
            type:         "text",
            defaultValue: PROVIDER_DEFAULT_URL[provider],
            tooltip:      `Base URL for the ${provider} backend. Leave at default unless the provider changed their endpoint.`,
            onChange:     (newVal) => scheduleSave(provider, "url", newVal || ""),
        });

        // API key field
        settings.push({
            id:           `PromptTools.${provider}.ApiKey`,
            category:     ["PromptTools", PROVIDER_LABEL[provider], "API Key"],
            name:         `${PROVIDER_LABEL[provider]} — API Key`,
            type:         "text",
            defaultValue: "",
            tooltip:      `API key for ${provider}. Leave empty if not needed (e.g. local Ollama). Saved to ~/.config/comfyui-prompt-tools/api_keys.yaml on the server.`,
            onChange:     (newVal) => scheduleSave(provider, "api_key", newVal || ""),
        });
    }

    return settings;
}

// ---------------- Extension registration ---------------------------------

app.registerExtension({
    name:     "comfyui-prompt-tools.settings",
    settings: buildSettings(),
    commands: [
        {
            id:       "PromptTools.TestAllConnections",
            label:    "Prompt Tools: Test All Connections",
            icon:     "pi pi-bolt",
            function: testAllConnections,
        },
        {
            id:       "PromptTools.DiscoverModels",
            label:    "Prompt Tools: Discover Models",
            icon:     "pi pi-search",
            function: discoverModels,
        },
    ],
    // v0.10.2: Surface both actions in the top menu under
    // "Extensions → Prompt Tools" so users do not need to know about
    // the command palette. Command palette entries remain registered
    // via the commands: [...] block above for power users.
    menuCommands: [
        {
            path: ["Extensions", "Prompt Tools"],
            commands: [
                "PromptTools.TestAllConnections",
                "PromptTools.DiscoverModels",
            ],
        },
    ],
    async setup() {
        // Best-effort status check — log so users can see in DevTools
        // whether the backend routes are reachable.
        try {
            const status = await apiGet(API.status);
            const summary = Object.entries(status)
                .map(([p, s]) => `${p}=${s.configured ? "✓" : "·"}`)
                .join(" ");
            console.log(`[Prompt Tools] Backend OK. Providers: ${summary}`);
        } catch (e) {
            console.warn(`[Prompt Tools] Backend status check failed: ${e}`);
        }
    },
});
