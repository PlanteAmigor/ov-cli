# ov-cli

**[中文](README.md) | [English](README_EN.md)**

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-≥3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/OpenVINO-≥2026.2-purple" alt="OpenVINO">
</p>

> I found the official OpenVINO tools a bit cumbersome for daily LLM experiments, so I built ov-cli as a lightweight alternative. With the help of AI coding tools, I turned my workflow needs into a simple CLI — setup, convert, chat — all in one place.

**OpenVINO LLM CLI Tool** — Lightweight, offline LLM inference on CPU & GPU.

> 💡 **Switch to Chinese UI**: Prefix any command with `--lang zh`, e.g. `./ov-cli --lang zh chat --model ./model-ov`

Built on Optimum Intel + OpenVINO GenAI. Features: model conversion (7 quantization formats), interactive chat (streaming, translation, VLM), OpenAI-compatible API server.

## Quick Start

```bash
# 1. One-command environment setup
./ov-cli setup
eval "$(./ov-cli venv)"

# 2. Convert model (HuggingFace → OpenVINO IR)
./ov-cli convert --model ./Qwen3/2B --format int8

# 3. Chat terminal
./ov-cli chat --model ./Qwen3/2B-ov

# 4. API server
./ov-cli server --model ./Qwen3/2B-ov
```

## How to Upgrade

```bash
# Pull latest code
git pull

# Running any command will auto-detect version changes:
# ⚠ Version changed (0.0.0 → 0.1.0), run:
#    ./ov-cli setup --fix

# Fix mode upgrades deps + reapplies patches in seconds
./ov-cli setup --fix
```

Fix mode (`setup --fix`) only upgrades package versions and reapplies patches — no redundant downloads.

**ZIP users**: Download the latest source, extract and overwrite your old directory, then run `./ov-cli setup --fix`.

## Commands

### `setup` — Create Environment

Creates a Python venv, installs all dependencies (openvino-genai, optimum-intel, transformers, torch, etc.),
auto-detects `optimum-intel-main/` source in project root to skip GitHub download,
applies Gemma-4 shared KV layer patch automatically.

```bash
./ov-cli setup                          # default ./.venv (interactive mode selection)
./ov-cli setup --venv ./my-venv         # custom path
./ov-cli setup --optimum-dir ./optimum-intel-main
./ov-cli setup --fix                    # fix mode (no rebuild, upgrade + repatch)
```

**Mode selection** (interactive):
1. **Simple mode** — pip install only. `--reasoning off` has no effect on thinking models.
2. **Full mode** — compiles modified GenAI from source to enable thinking budget
   (logit-level `</think>` forcing).

**Version detection**: After `git pull`, running any command will auto-detect version
changes and suggest `./ov-cli setup --fix` for a quick fix.

**Fix mode** (`--fix`): Skips venv recreation, only upgrades dependencies and
reapplies patches. Takes seconds — ideal for version updates or patch fixes.

### `venv` — Enter Virtual Environment

```bash
eval "$(./ov-cli venv)"
eval "$(./ov-cli venv --venv ./my-venv)"
```

### `convert` — Model Conversion

Exports HuggingFace models to OpenVINO IR via Optimum Intel, auto-detecting task type.

```bash
./ov-cli convert --model ./Qwen3/2B --format int8     # output to ./model-ov
./ov-cli convert --model ./Qwen3/2B --format int4 -o ./custom-path
```

**Quantization formats** (7):

| Format | Size (vs fp32) | Notes |
|--------|:-------------:|-------|
| `fp32` | 100% | Lossless |
| `fp16` | ~50% | Half precision |
| `int8` | ~25% | 8-bit |
| `int4` | ~12.5% | 4-bit |
| `mxfp4` | ~12.5% | MX float 4-bit |
| `nf4` | ~12.5% | Normal float 4-bit |
| `cb4` | ~12.5% | 4-bit (double) |

**INT4 mixed precision**:

```bash
./ov-cli convert --model ./Hy-MT2/1.8B --format int4 --ratio 0.8 --group-size 128
```

| Param | Default | Description |
|-------|---------|-------------|
| `--ratio` | 1.0 | INT4 ratio (0-1), lower = more INT8 |
| `--group-size` | 128 | Quantization group size |

### `chat` — Chat Terminal

Interactive terminal. Auto-detects model format (GenAI / Optimum), supports streaming, multi-turn, images.

```bash
# Chat mode
./ov-cli chat --model ./Qwen3/2B-ov                               # GenAI format
./ov-cli chat --model ./gemma-4-E2B-it-ov                          # Optimum format (Gemma-4)
./ov-cli chat --model ./model-ov --temp 0.9 --max-tokens 2048

# Reasoning control (GenAI format only)
./ov-cli chat --model ./Qwen3.5/0.8B-ov --reasoning off            # filter <think> blocks
./ov-cli chat --model ./Qwen3.6/35B-A3B-ov --reasoning off         # force </think> (full mode)

# Translate mode
./ov-cli chat --model ./Hy-MT2-1.8B-ov --mode translate

# VLM image support
./ov-cli chat --model ./model-vlm-ov --image ./photo.jpg

# English UI (default)
./ov-cli --lang en chat --model ./model-ov

# Once mode (single output, auto-exit)
./ov-cli chat --model ./model-ov --mode once --prompt "Hello"
./ov-cli chat --model ./model-ov --mode once --file ./doc.pdf --prompt "summarize" --output ./outputs/
```

**In-chat commands** (chat mode only):

| Command | Description |
|---------|-------------|
| `//img PATH1 [PATH2 ...]` | Load image(s) (VLM) |
| `//pdf PATH` | Load PDF (auto-convert to images, max 24 pages) |
| `//txt PATH1 [PATH2 ...]` | Load text file(s) |
| `/file` | List loaded files |
| `/temp N` | Set temperature (0-2) |
| `/system TEXT` | Set system prompt |
| `/clear [ids]` | Clear context or specific files by ID |
| `/help` | Help |
| `/exit` | Exit |

**Once mode** (`--mode once`):

| Option | Description |
|--------|-------------|
| `--prompt TEXT` | Input text (supports `\n` newlines) |
| `--file PATH` | Upload file(s), auto-detect type (PDF/image/text) |
| `--output PATH` | Save result as .md file (auto-name or explicit path) |

### `server` — API Server

Starts an OpenAI-compatible HTTP API server.

```bash
./ov-cli server --model ./Qwen3/8B-ov                              # default port 8080
./ov-cli server --model ./model-ov --port 8081 --host 0.0.0.0
./ov-cli server --model ./model-ov --device CPU                     # force CPU
```

**API Endpoints**:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/models` | List models + capabilities |
| `POST` | `/v1/chat/completions` | Chat completion (stream + non-stream, multi-image) |
| `POST` | `/v1/chat/completions/control` | Stop generation |
| `GET` | `/props` | Server properties |
| `GET` | `/health` | Health check |
| `POST` | `/token` | Count tokens |

**curl examples**:

```bash
# Text chat
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"8B-ov","messages":[{"role":"user","content":"Hello"}],"stream":false,"max_tokens":100}' \
  | python3 -m json.tool

# Streaming
curl -s -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"8B-ov","messages":[{"role":"user","content":"Count 1 2 3"}],"stream":true,"max_tokens":50}'

# Image inference
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "$(cat <<EOF
{"model":"8B-ov","messages":[{"role":"user","content":[
  {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,$(base64 -w0 /path/to/photo.jpg)"}},
  {"type":"text","text":"What color?"}
]}],"stream":false,"max_tokens":50}
EOF
)" | python3 -m json.tool
```

### `benchmark` — Benchmark

```bash
./ov-cli benchmark --model ./Qwen3.5/0.8B-ov
./ov-cli benchmark --model ./Qwen3.6/35B-A3B-ov --reasoning off
```

## Model Support

### Inference Formats

| Format | Loader | Feature |
|--------|--------|---------|
| **GenAI** | `LLMPipeline` / `VLMPipeline` | Standard export, `openvino_config.json` |
| **Optimum** | `OVModelForVisualCausalLM` + `AutoProcessor` | Gemma-4 etc., has `openvino_text_embeddings_per_layer_model.xml` |

### Verified Models

| Model | Format | Text | Image | Translate | Notes |
|-------|--------|:----:|:-----:|:---------:|-------|
| **Hy-MT2 1.8B** | GenAI | | | ✅ | Translation model, all 4 precisions |
| **Gemma-4 E2B** | Optimum | ✅ | ✅ | | INT4, needs `kv_shared_layer` patch |
| **Qwen3-VL 8B** | GenAI | ✅ | ✅ | | Pre-converted, 1 of 3 supported VLM |
| **Qwen3.6 35B-A3B** | GenAI | ✅ | ✅ | | MoE, pre-converted |
| **Qwen3.5 0.8B** | GenAI | ✅ | ❌ | | Small model VLM unsupported |
| **Qwen3 2B** | GenAI | ✅ | ❌ | | Vision encoder reshape bug |

> **VLM note**: Among Qwen models, GenAI `VLMPipeline` only supports vision for **Qwen3-VL 8B**, **Qwen3.6 35B-A3B**, **Qwen3.5 35B-A3B**. Small models (0.8B, 2B) have vision encoder issues. Optimum format models (Gemma-4) may not be affected.

### Pre-converted Models (Recommended)

OpenVINO provides a large collection of pre-converted models on HuggingFace and ModelScope.
**Skip the conversion step** — download and run directly:

- [HuggingFace OpenVINO Models](https://huggingface.co/OpenVINO)
- [ModelScope OpenVINO Models](https://www.modelscope.cn/organization/OpenVINO?tab=model)

### Manual Conversion

`./ov-cli convert` supports the following architectures (verified to export successfully):

| Architecture | Notes |
|------|------|
| Qwen3 / Qwen3.5 / Qwen3.6 | Includes MoE variants |
| Hy-MT2 | Multi-language translation model |
| Llama / Mistral / DeepSeek / Phi / Gemma | Standard transformers architectures |

All standard transformers architectures should work as long as `optimum-cli` can export them.

### Notes

- **Gemma-4**: Export needs `model_patcher.py` patch (`kv_shared_layer_index` → `layer_type`), applied by `setup` automatically.
- **Ctrl+C**: Interrupt during generation may take 20-200ms (one token time).
- **`--reasoning off`**: Inherent thinking models (Qwen3.6 etc.) cannot be stopped by prompt tricks.
  ov-cli inserts a `ThinkingBudgetTransform` into the LogitProcessor chain to force `</think>`.
  Requires `setup` **full mode** (compiled GenAI).
  Simple mode `--reasoning off` only filters `<think>` blocks from output, but cannot prevent the model from reasoning.
- **Pre-converted models**: Available at [ModelScope OpenVINO](https://www.modelscope.cn/organization/OpenVINO) or [HuggingFace OpenVINO](https://huggingface.co/OpenVINO).

## Performance

Tested on: Intel Arc Pro 130T/140T (Arrow Lake-P) GPU | openvino-genai 2026.2 | 3 warmup rounds

| Model | Quant | 32 1st | 32 2nd | 32 tok/s | 1024 1st | 1024 2nd | 1024 tok/s |
|:-----|:----:|:------:|:------:|:--------:|:---------:|:---------:|:----------:|
| **Qwen3.5/0.8B** | int8 | 297ms | 19ms | 54.9 | 660ms | 20ms | 51.8 |
| **Hy-MT2/1.8B** | int4 | 267ms | 25ms | 40.6 | 710ms | 24ms | 38.2 |
| **Qwen3/2B** | int8 | 262ms | 33ms | 30.7 | 771ms | 35ms | 27.8 |
| **Qwen3/8B** | int4 AWQ | 402ms | 79ms | 12.9 | 2161ms | 82ms | 12.1 |
| **Gemma-4 E2B** | int4 | 342ms | 77ms | 14.2 | 1732ms | 196ms | 10.8 |
| **Qwen3.6/35B** (reasoning on) | int4/8 | 1069ms | 88ms | 11.8 | 4518ms | 87ms | 11.6 |
| **Qwen3.6/35B** (reasoning off) | int4/8 | 1070ms | 92ms | 11.2 | 4571ms | 94ms | 10.9 |

> tok/s based on encoded text. Chinese ~1.8 chars/subword.

## Project Structure

```
ov-cli/
├── ov-cli                   # Entry script (auto-discovers .venv)
├── pyproject.toml
├── README.md / README_EN.md
│
├── ov_cli/
│   ├── __init__.py          # Package info + i18n
│   ├── __main__.py          # python -m ov_cli entry
│   ├── cli.py               # CLI parser + dispatcher + setup
│   ├── chat.py              # Chat/translate terminal (GenAI + Optimum)
│   ├── convert.py           # Model conversion (7 formats)
│   ├── server.py            # FastAPI OpenAI-compatible server
│   └── benchmark.py         # Performance benchmark
│
└── openvino.genai-2026.2.0.0-optimization/  # Modified GenAI source (full mode)
```

## Dependencies

- Python >= 3.10
- OpenVINO >= 2026.2, openvino-genai
- Optimum Intel >= 1.27.0 (GitHub source)
- transformers >= 5.9, torch, torchvision
- GPU: Intel integrated / Arc (auto-detected)
- CPU: Any x86-64

## Related Links

- [OpenVINO Documentation](https://docs.openvino.ai/)
- [OpenVINO GitHub](https://github.com/openvinotoolkit/openvino)
- [OpenVINO Toolkit Repositories](https://github.com/orgs/openvinotoolkit/repositories?type=all)


