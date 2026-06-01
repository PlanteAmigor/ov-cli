# ov-cli

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-≥3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/OpenVINO-≥2026.2-purple" alt="OpenVINO">
  <img src="https://img.shields.io/badge/platform-Linux%20|%20Windows-lightgrey" alt="Platform">
  <img src="https://img.shields.io/github/stars/PlanteAmigor/ov-cli?style=flat&label=stars" alt="Stars">
</p>

**[中文](README.md) | English**

**OpenVINO LLM CLI Tool** — Lightweight, offline, runs on CPU & GPU.

Built on Optimum Intel + OpenVINO GenAI. Supports model conversion (FP32/FP16/INT8/INT4), interactive chat (streaming), and translation.

## Quick Start

```bash
# 1. One-command environment setup
./ov-cli setup
eval "$(./ov-cli venv)"

# 2. Convert model
./ov-cli convert --model ./Qwen3/2B --format int8

# 3. Chat
./ov-cli chat --model ./Qwen3/2B-ov
```

## Commands

### `setup` — Create Environment

Creates a Python venv and installs all dependencies (`openvino-genai`, `optimum-intel`, `transformers 5.9`, `torch`, etc.),
then applies the Gemma-4 shared KV layer patch.

```bash
./ov-cli setup                          # default ./.venv (interactive mode selection)
./ov-cli setup --venv ./my-venv         # custom path
./ov-cli setup --optimum-dir ./optimum-intel-main

# Full mode compiles openvino-genai from source to enable reasoning budget
# (logit-level `</think>` forcing). **Linux only**.
```

**Mode selection** (interactive): `setup` prompts to choose:
1. **Simple mode** — pip install only, for regular use
2. **Full mode** — compiles modified GenAI from source to enable `--reasoning off` for thinking models (Qwen3.6 etc.)

### `venv` — Enter Virtual Environment

Prints the activate command for the venv created by `setup`:

```bash
eval "$(./ov-cli venv)"
eval "$(./ov-cli venv --venv ./my-venv)"
```

### `convert` — Model Conversion

Exports HuggingFace models to OpenVINO IR using Optimum Intel, auto-detecting task type.

```bash
./ov-cli convert --model ./Qwen3/2B --format int8
./ov-cli convert --model ./Qwen3/2B --format int4 -o ./Qwen3/2B-ov-int4
./ov-cli convert --model ./model --format fp16
```

**Quantization formats**:

| Format | Size (vs fp32) | Notes |
|--------|---------------|-------|
| `fp32` | 100% | Lossless, largest |
| `fp16` | ~50% | Half precision, nearly lossless |
| `int8` | ~25% | 8-bit, nearly lossless |
| `int4` | ~12.5% | 4-bit, some accuracy loss |

**Advanced options**:

```bash
# INT4 mixed precision (80% INT4 + 20% INT8)
./ov-cli convert --model ./Hy-MT2/1.8B --format int4 --ratio 0.8 --group-size 128
```

| Param | Default | Description |
|-------|---------|-------------|
| `--ratio` | 1.0 | INT4 ratio (0-1), lower = more INT8 |
| `--group-size` | 128 | Quantization group size |

### `chat` — Chat Terminal

Loads an OpenVINO model and starts an interactive terminal. Auto-detects model format, supports streaming, multi-turn, images (VLM).

```bash
# Chat mode (auto-detect format)
./ov-cli chat --model ./Qwen3/2B-ov                               # GenAI format
./ov-cli chat --model ./gemma-4-E2B-it-ov                          # Optimum format (Gemma-4)
./ov-cli chat --model ./model-ov --temp 0.9 --max-tokens 2048

# Reasoning control (GenAI format only)
./ov-cli chat --model ./Qwen3/2B-ov --reasoning on                 # thinking on (default)
./ov-cli chat --model ./Qwen3.5/0.8B-ov --reasoning off            # thinking off
./ov-cli chat --model ./Qwen3.6/35B-A3B-ov --reasoning off         # needs custom GenAI build

# Translate mode (Hy-MT2 etc.)
./ov-cli chat --model ./Hy-MT2-1.8B-ov --mode translate

# VLM image support
./ov-cli chat --model ./model-vlm-ov --image ./photo.jpg

# English UI
./ov-cli --lang en chat --model ./model-ov
```

**In-chat commands**:

| Command | Description |
|---------|-------------|
| `//img PATH` | Load image (VLM) |
| `/temp 0.7` | Temperature (0-2) |
| `/system ...` | System prompt |
| `/clear` | Clear context |
| `/help` | Help |
| `/exit` | Exit |

**Translate mode commands**:

| Command | Description |
|---------|-------------|
| `//en text` | Force translate to English |
| `//zh text` | Force translate to Chinese |
| Direct input | Auto-detect language direction |

> **Note**: Translate mode accepts one line per input. Paste multi-line text in separate chunks.

### ✅ Two Inference Formats

| Format | Loader | Models | Feature |
|--------|--------|--------|---------|
| **GenAI** | `LLMPipeline` / `VLMPipeline` | Standard optimum-cli exports | `openvino_config.json`, no per-layer models |
| **Optimum** | `OVModelForVisualCausalLM` | Gemma-4 VLM | Has `openvino_text_embeddings_per_layer_model.xml` |

### ✅ Verified

| Model | Format | Text | Image | Translate | Notes |
|-------|--------|:----:|:-----:|:---------:|-------|
| **Hy-MT2 1.8B** | GenAI | | | ✅ | `--mode translate`, all 4 precisions |
| **Gemma-4 E2B** | Optimum | ✅ | ✅ | | INT4, needs `kv_shared_layer` patch |
| **Qwen3-VL 8B** | GenAI | ✅ | ✅ | | Official pre-converted |
| **Qwen3.5 0.8B** | GenAI | ✅ | ❌ | | Vision encoder bug |
| **Qwen3.6 35B-A3B** | GenAI | ✅ | ✅ | | MoE, mixed precision |
| **Qwen3 2B** | GenAI | ✅ | ❌ | | Self-converted vision bug |

### ⚠️ Notes

- **Gemma-4**: Export needs `model_patcher.py` patch (`kv_shared_layer_index` → `layer_type`), `setup` applies it automatically
- **Ctrl+C latency**: Interrupt may take 20-200ms (one token time). `^C` may appear in output
- **`--reasoning off` for thinking models**: Qwen3.6 etc. cannot truly disable reasoning via prompt. `ov-cli` patches OpenVINO GenAI (`ThinkingBudgetTransform`) for logit-level `</think>` forcing, similar to llama.cpp's reasoning budget. Use `setup` full mode (option 2) to auto-build. **Linux only.**
- Pre-converted OpenVINO models: [ModelScope OpenVINO](https://www.modelscope.cn/organization/OpenVINO)

## Project Structure

```
ov-cli/
├── ov-cli                   # Shell entry script
├── ov-cli.bat               # Windows entry script
├── pyproject.toml
├── README.md / README_EN.md
│
├── ov_cli/
│   ├── __init__.py          # Package info + i18n
│   ├── __main__.py          # python -m ov_cli entry
│   ├── cli.py               # CLI parser + dispatcher + setup patches
│   ├── chat.py              # Chat/translate terminal (GenAI + Optimum)
│   └── convert.py           # Model conversion
│
├── openvino.genai-2026.2.0.0-optimization/  # Modified GenAI source (full mode)
│
├── model/                   # Model files
│
└── optimum-intel-main/      # Optimum Intel source (optional)
```

## Performance

Tested on: Intel Arc Pro 130T/140T (Arrow Lake-P) GPU | openvino-genai 2026.2 | 3 warmup rounds

| Model | Quant | 32 1st | 32 2nd | 32 tok/s | 1024 1st | 1024 2nd | 1024 tok/s | RSS |
|:-----|:----:|:------:|:------:|:--------:|:---------:|:---------:|:----------:|:---:|
| **Qwen3.5/0.8B-ov** | int8 | 297ms | 19ms | **54.9** | 660ms | 20ms | **51.8** | 826MB |
| **Hy-MT2/1.8B-ov** | int4 | 267ms | 25ms | 40.6 | 710ms | 24ms | 38.2 | 916MB |
| **Qwen3/2B-ov** | int8 | 262ms | 33ms | 30.7 | 771ms | 35ms | 27.8 | 1207MB |
| **Qwen3/8B-ov** | int4 AWQ | 402ms | 79ms | 12.9 | 2161ms | 82ms | 12.1 | 2010MB |
| **Gemma-4-E2B-ov** | int4 | 342ms | 77ms | 14.2 | 1732ms | 196ms | 10.8 | 8278MB |
| **Qwen3.6/35B-A3B** (reasoning on) | int4/8 | 1069ms | 88ms | 11.8 | 4518ms | 87ms | 11.6 | 1013MB |
| **Qwen3.6/35B-A3B** (reasoning off) | int4/8 | 1070ms | 92ms\* | 11.2 | 4571ms | 94ms\* | 10.9 | 1015MB |

> \* reasoning off: 2nd latency includes minimal overhead from forced `</think>` (~1 thinking token). Throughput (tok/s) unaffected.

## Dependencies

- Python >= 3.10
- OpenVINO >= 2026.2, openvino-genai
- Optimum Intel >= 1.27.0 (GitHub source)
- transformers >= 5.9, torch, torchvision
- GPU: Intel integrated / Arc (auto-detect)
- CPU: Any x86-64

## Windows Support

| Item | Notes |
|------|-------|
| **Entry** | Use `ov-cli.bat` or `python -m ov_cli` |
| **setup** | Auto-detects Windows paths (`Scripts\` vs `bin/`) |
| **Multi-line input** | Falls back to single-line |
| **benchmark RSS** | Not collected (only Unix) |
| **Not supported** | Bash entry script `ov-cli` |

```bash
# Windows usage
ov-cli.bat setup
ov-cli.bat chat --model ./model-ov
# or
python -m ov_cli setup
python -m ov_cli chat --model ./model-ov
```
