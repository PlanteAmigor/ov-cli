# Configure OpenCode with ov-cli local model

[OpenCode](https://opencode.ai) is an open-source AI coding assistant CLI that supports connecting to local models via OpenAI-compatible APIs.

## Prerequisites

ov-cli server must be running:

```bash
./ov-cli server --model ./model-ov --device GPU.1
```

Verify the API is accessible:

```bash
curl http://localhost:8080/v1/models
# Should return a list of models
```

## Configuration

### 1. Edit config file

`~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ov-cli": {                            // provider ID, can be any string
      "npm": "@ai-sdk/openai-compatible",  // OpenAI-compatible API adapter
      "name": "ov-cli (local)",            // display name in UI
      "options": {
        "baseURL": "http://localhost:8080/v1",
        "apiKey": "dummy"                  // any value works for local server
      },
      "models": {
        "4B-ov-int4": {                    // model ID, must match /v1/models response
          "name": "Qwen3.5-4B (local)"
        }
      }
    }
  }
}
```

> Get the model ID by running `curl http://localhost:8080/v1/models`.

### 2. Start OpenCode

```bash
PATH="$HOME/.opencode/bin:$PATH" opencode
```

### 3. Select model

In the TUI, press `/` and enter:

```
/models
```

Select `ov-cli (local)` → `Qwen3.5-4B (local)`.

## Switching models

Replace the model ID and name under `models` with the corresponding model from your ov-cli server.
