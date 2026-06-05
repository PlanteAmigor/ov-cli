# ov-cli

**[中文](README.md) | [English](README_EN.md)**

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-≥3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/OpenVINO-≥2026.2-purple" alt="OpenVINO">
</p>

> 我发现官方的 OpenVINO 工具在日常 LLM 实验中较为繁琐，因此我开发了 ov-cli 作为轻量级的替代方案。借助 AI 编码工具，我将我的工作流需求转化为简单的 CLI 操作——设置、转换、聊天——所有操作都可以在同一个地方完成。

**OpenVINO LLM 命令行工具** — 轻量、离线、CPU/GPU 皆可运行。

> 💡 **切换英文界面**：所有命令前加 `--lang en`，例如 `./ov-cli --lang en chat --model ./model-ov`

基于 Optimum Intel + OpenVINO GenAI 推理引擎。支持模型转换（7 种量化格式）、交互式聊天（流式/翻译/图片）、OpenAI 兼容 API 服务。

## 快速开始

```bash
# 1. 一键创建环境（自动安装所有依赖）
./ov-cli setup
eval "$(./ov-cli venv)"

# 2. 转换模型（HuggingFace → OpenVINO IR）
./ov-cli convert --model ./Qwen3/2B --format int8

# 3. 聊天终端
./ov-cli chat --model ./Qwen3/2B-ov

# 4. 文生图
./ov-cli generate --model ./FLUX/ov-int4

# 5. API 服务
./ov-cli server --model ./Qwen3/2B-ov
```

## 如何升级

```bash
# 拉取最新代码
git pull

# 运行任意命令时会自动检测版本变化并提示：
# ⚠ 检测到版本变化 (0.0.0 → 0.1.0)，建议运行:
#    ./ov-cli setup --fix

# 使用修复模式快速升级（不重建 venv，数秒完成）
./ov-cli setup --fix
```

升级模式 (`setup --fix`) 仅升级依赖版本和重打补丁，不下重复的包。

**ZIP 用户**：下载最新源码解压后，覆盖旧目录，然后执行 `./ov-cli setup --fix` 即可。

## 命令

### `setup` — 创建环境

创建 Python 虚拟环境，安装所有依赖（openvino-genai、optimum-intel、transformers、torch 等），
自动检测项目根目录下的 `optimum-intel-main/` 源码目录以跳过 GitHub 下载，
安装完成后自动应用 Gemma-4 共享 KV 层补丁。

```bash
./ov-cli setup                          # 默认 ./.venv（交互选择安装模式）
./ov-cli setup --venv ./my-venv         # 指定路径
./ov-cli setup --optimum-dir ./optimum-intel-main  # 指定 optimum 源码
./ov-cli setup --fix                    # 修复环境（不重建，仅升级+重打补丁）
```

**模式选择**（交互式，由 `setup` 提示）：
1. **简易模式** — pip 安装，日常使用。`--reasoning off` 对思考型模型无效。
2. **完整模式** — 从源码编译 OpenVINO GenAI 以启用 thinking budget 功能
   （logit 级别的 `</think>` 强制结束思考）。

**版本检测**：`git pull` 更新代码后运行任意命令会自动检测版本变化，
提示执行 `./ov-cli setup --fix` 快速修复。

**修复模式** (`--fix`)：不重建虚拟环境，仅升级依赖、重打补丁，
适用于版本更新或补丁修复，数秒完成。

### `venv` — 进入虚拟环境

```bash
eval "$(./ov-cli venv)"
eval "$(./ov-cli venv --venv ./my-venv)"
```

### `convert` — 模型转换

使用 Optimum Intel 将 HuggingFace 模型导出为 OpenVINO IR，自动推断 task 类型。

```bash
./ov-cli convert --model ./Qwen3/2B --format int8     # 默认输出到 ./model-ov
./ov-cli convert --model ./Qwen3/2B --format int4 -o ./custom-path
```

**量化格式**（7 种）：

| 格式 | 体积 (相对 fp32) | 说明 |
|------|:-------------:|------|
| `fp32` | 100% | 无损 |
| `fp16` | ~50% | 半精度，几乎无损 |
| `int8` | ~25% | 8-bit，几乎无损 |
| `int4` | ~12.5% | 4-bit，有精度损失 |
| `mxfp4` | ~12.5% | MX 浮点 4-bit |
| `nf4` | ~12.5% | 正态分布 4-bit |
| `cb4` | ~12.5% | 双峰 4-bit |

**INT4 混合精度**：

```bash
./ov-cli convert --model ./Hy-MT2/1.8B --format int4 --ratio 0.8 --group-size 128
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--ratio` | 1.0 | INT4 比例 (0-1)，越低 INT8 越多 |
| `--group-size` | 128 | 量化分组大小 |

### `chat` — 聊天终端

交互式终端。自动检测模型格式（GenAI / Optimum），支持流式输出、多轮对话、图片。

```bash
# 聊天模式
./ov-cli chat --model ./Qwen3/2B-ov                               # GenAI 格式
./ov-cli chat --model ./gemma-4-E2B-it-ov                          # Optimum 格式
./ov-cli chat --model ./model-ov --temp 0.9 --max-tokens 2048

# 推理控制（仅 GenAI 格式）
./ov-cli chat --model ./Qwen3.5/0.8B-ov --reasoning off            # 关闭思考（过滤<think>块）
./ov-cli chat --model ./Qwen3.6/35B-A3B-ov --reasoning off         # 关闭思考（需完整模式）

# 翻译模式
./ov-cli chat --model ./Hy-MT2-1.8B-ov --mode translate

# VLM 图片支持
./ov-cli chat --model ./model-vlm-ov --image ./photo.jpg

# 英文界面
./ov-cli --lang en chat --model ./model-ov

# 单次输出模式（输出完自动退出）
./ov-cli chat --model ./model-ov --mode once --prompt "Hello"
./ov-cli chat --model ./model-ov --mode once --file ./doc.pdf --prompt "总结" --output ./outputs/
./ov-cli chat --model ./model-ov --mode once --prompt "Hello" --json          # JSON 格式输出
```

**终端内命令**（仅 chat 模式）：

| 命令 | 说明 |
|------|------|
| `//img PATH1 [PATH2 ...]` | 加载图片（支持多文件，VLM） |
| `//pdf PATH` | 加载 PDF（自动转图片，最多 24 页，[限页原因](https://github.com/openvinotoolkit/openvino/issues/36260)） |
| `//txt PATH1 [PATH2 ...]` | 加载文本文件（支持多文件） |
| `/file` | 查看已加载文件列表 |
| `/temp N` | 设置温度 (0-2) |
| `/system TEXT` | 设置系统提示词 |
| `/clear [ids]` | 清空全部上下文或指定文件 ID |
| `/help` | 帮助 |
| `/exit` | 退出 |

**单次输出模式**（`--mode once`）：

| 参数 | 说明 |
|------|------|
| `--prompt TEXT` | 文字输入（支持 `\n` 换行） |
| `--file PATH` | 上传文件（可多次，支持 PDF/图片/文本） |
| `--output PATH` | 保存结果为 .md 文件（自动命名或指定路径） |

**翻译模式**：自动检测语言方向；`//en 文本` 强制译英，`//zh 文本` 强制译中。

### `server` — API 服务

启动 OpenAI 兼容的 HTTP API 服务器。

```bash
./ov-cli server --model ./Qwen3/8B-ov                              # 默认端口 8080
./ov-cli server --model ./model-ov --port 8081 --host 0.0.0.0
./ov-cli server --model ./model-ov --device CPU                     # 指定 CPU
```

**API 端点**：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/models` | 模型列表 + 能力（视觉/文字） |
| `POST` | `/v1/chat/completions` | 聊天补全（流式 + 非流式，支持多图） |
| `POST` | `/v1/chat/completions/control` | 停止生成 |
| `GET` | `/props` | 服务器属性 |
| `GET` | `/health` | 健康检查 |
| `POST` | `/token` | Token 计数 |

**curl 示例**：

```bash
# 文字聊天
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"8B-ov","messages":[{"role":"user","content":"你好"}],"stream":false,"max_tokens":100}' \
  | python3 -m json.tool

# 流式输出
curl -s -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"8B-ov","messages":[{"role":"user","content":"数数 1 2 3"}],"stream":true,"max_tokens":50}'

# 图片推理
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "$(cat <<EOF
{"model":"8B-ov","messages":[{"role":"user","content":[
  {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,$(base64 -w0 /path/to/photo.jpg)"}},
  {"type":"text","text":"什么颜色？"}
]}],"stream":false,"max_tokens":50}
EOF
)" | python3 -m json.tool
```

### `benchmark` — 性能测试

```bash
./ov-cli benchmark --model ./Qwen3.5/0.8B-ov
./ov-cli benchmark --model ./Qwen3.6/35B-A3B-ov --reasoning off
```

### `whisper` — 语音转文字

使用 OpenVINO GenAI WhisperPipeline 转录音频，支持交互式和单次模式。

```bash
# 交互式
./ov-cli whisper --model ./whisper/ov-large

# 单次（输出完自动退出）
./ov-cli whisper --model ./whisper/ov-large --mode once --file speech.mp3 -o output.txt
./ov-cli whisper --model ./whisper/ov-large --mode once --file speech.mp3 --json   # JSON 格式输出
```

**注意：** Whisper 根据音频中的停顿和语调添加标点符号。
TTS（文字转语音）生成的音频语速均匀、缺少自然停顿，转录结果可能不含句号逗号等标点，属正常现象。

### `generate` — 文生图

使用 OpenVINO GenAI Text2ImagePipeline 生成图片，支持交互式和单次模式。

```bash
# 交互式（多轮生图）
./ov-cli generate --model ./FLUX/ov-int4

# 单次（输出完自动退出）
./ov-cli generate --model ./FLUX/ov-int4 --mode once --prompt "cat" -o cat.png
./ov-cli generate --model ./FLUX/ov-int4 --mode once --prompt "cat" --json       # JSON 格式输出
```

**终端内命令**（仅交互模式）：

| 命令 | 说明 |
|------|------|
| `/size W H` | 设置分辨率 (默认 512x512) |
| `/steps N` | 推理步数 (默认 4) |
| `/guidance F` | guidance scale (默认 0.0) |
| `/seed [N]` | 设置/重置随机种子 |
| `/save DIR` | 设置输出目录 |
| `/history` | 查看已生成的图片 |
| `/help` | 帮助 |
| `/exit` | 退出 |

## 外部集成

ov-cli 支持通过 `--mode once` 和 `--json` 被其他项目调用，日志走 stderr，stdout 只输出纯净结果。

### 支持外部调用的命令

| 命令 | once 模式 | `--json` | stdout 输出 |
|:----|:---------:|:--------:|:------------|
| `chat` | `--mode once --prompt TEXT [--file ...]` | ✅ | 回复文本 / `{"text":"...","time":n}` |
| `whisper` | `--mode once --file audio.mp3` | ✅ | 转录文本 / `{"text":"...","time":n,"duration":n}` |
| `generate` | `--mode once --prompt "cat" [-o output.png]` | ✅ | 图片路径 / `{"path":"...","time":n}` |

### 推荐方式

```bash
# Shell 脚本：捕获纯文本结果
text=$(/path/to/ov-cli whisper -m ./model --mode once -f speech.mp3 2>/dev/null)

# Shell 脚本：捕获 JSON
json=$(/path/to/ov-cli whisper -m ./model --mode once -f speech.mp3 --json 2>/dev/null)
```

```python
# Python 子进程调用
import subprocess, json

result = subprocess.run([
    "/path/to/ov-cli", "whisper",
    "--model", "./model",
    "--mode", "once",
    "--file", "speech.mp3",
    "--json"
], capture_output=True, text=True)

if result.returncode == 0:
    data = json.loads(result.stdout)
    print(data["text"])  # 转录结果
```

> 💡 `2>/dev/null` 可省略日志，只保留 stdout 的结果。不加则终端同时显示日志和结果。

## 模型支持

### 两种推理格式

| 格式 | 加载方式 | 特征 |
|------|---------|------|
| **GenAI** | `LLMPipeline` / `VLMPipeline` | 标准 optimum-cli 导出，`openvino_config.json` |
| **Optimum** | `OVModelForVisualCausalLM` + `AutoProcessor` | Gemma-4 等，有 `openvino_text_embeddings_per_layer_model.xml` |

### 实测验证

| 模型 | 格式 | 文字 | 图片 | 翻译 | 说明 |
|------|------|:----:|:----:|:----:|------|
| **Hy-MT2 1.8B** | GenAI | | | ✅ | 翻译模型，4 种精度全通过 |
| **Gemma-4 E2B** | Optimum | ✅ | ✅ | | INT4，需 `kv_shared_layer` 补丁 |
| **Qwen3-VL 8B** | GenAI | ✅ | ✅ | | 官方预转换，OpenVINO 3 个受支持 VLM 之一 |
| **Qwen3.6 35B-A3B** | GenAI | ✅ | ✅ | | MoE，官方预转换 |
| **Qwen3.5 0.8B** | GenAI | ✅ | ❌ | | 小模型 VLM 不支持 |
| **Qwen3 2B** | GenAI | ✅ | ❌ | | 视觉编码器 reshape 有 bug |

> **VLM 说明**：GenAI 格式的 `VLMPipeline` 对 Qwen 系列只支持 **Qwen3-VL 8B**、**Qwen3.6 35B-A3B**、**Qwen3.5 35B-A3B** 三个模型的视觉能力。小模型（0.8B、2B）视觉编码器在 OpenVINO 上无法正常工作。Optimum 格式（Gemma-4）可能不受此限制。

### 预转换模型（推荐）

OpenVINO 官方已在 HuggingFace 和 ModelScope 提供了大量预转换模型，
**省去自己转换的麻烦**，下载即用：

- [HuggingFace OpenVINO 模型库](https://huggingface.co/OpenVINO)
- [ModelScope OpenVINO 模型库](https://www.modelscope.cn/organization/OpenVINO?tab=model)

### 自行转换

`./ov-cli convert` 支持以下架构（已验证可成功导出）：

| 架构 | 说明 |
|------|------|
| Qwen3 / Qwen3.5 / Qwen3.6 | 含 MoE 变体 |
| Hy-MT2 | 多语言翻译模型 |
| Llama / Mistral / DeepSeek / Phi / Gemma | 标准 transformers 架构 |

理论上支持所有 transformers 标准架构，只需 `optimum-cli` 能成功导出即可推理。

> **语音模型**：当前 `convert` 不支持导出 Whisper 等语音模型。
> 请下载官方预转换模型：
> - [HuggingFace Speech-to-Text 合集](https://huggingface.co/collections/OpenVINO/speech-to-text)
> - [ModelScope Speech-to-Text 合集](https://www.modelscope.cn/collections/Speech-to-Text-b9ab5c24c32649)
>
> **文生图模型**：当前 `convert` 不支持转换文生图模型（如 FLUX、SD3.5）。
> 请下载官方预转换模型：
> - [HuggingFace Image Generation 合集](https://huggingface.co/collections/OpenVINO/image-generation)
> - [ModelScope 文生图合集](https://www.modelscope.cn/collections/Image-Generation-eb38cde2fa3d46)

### 注意事项

- **Gemma-4**：导出需修改 `model_patcher.py` 中 `kv_shared_layer_index` → `layer_type`，`setup` 自动打补丁。
- **Ctrl+C 中断**：生成期间按 Ctrl+C 可中断，但需等待当前 token 生成完毕（约 20-200ms）。
- **`--reasoning off`**：Qwen3.6 等天生思考模型无法通过 prompt 技巧阻止推理。
  解决方案：ov-cli 在 LogitProcessor 中插入 `ThinkingBudgetTransform`，
  预算耗尽后强制输出 `</think>`。需 `setup` **完整模式**编译修改版 GenAI。
  简易模式下 `--reasoning off` 仅过滤输出中的 `<think>` 块，但无法阻止推理。
- **预转换模型**：可在 [ModelScope OpenVINO 组织](https://www.modelscope.cn/organization/OpenVINO) 或 [HuggingFace OpenVINO](https://huggingface.co/OpenVINO) 查找。

## 性能参考

测试环境: Intel Arc Pro 130T/140T (Arrow Lake-P) GPU | openvino-genai 2026.2 | 3 轮预热

| 模型 | 量化 | 32 1st | 32 2nd | 32 tok/s | 1024 1st | 1024 2nd | 1024 tok/s |
|:-----|:----:|:------:|:------:|:--------:|:---------:|:---------:|:----------:|
| **Qwen3.5/0.8B** | int8 | 297ms | 19ms | 54.9 | 660ms | 20ms | 51.8 |
| **Hy-MT2/1.8B** | int4 | 267ms | 25ms | 40.6 | 710ms | 24ms | 38.2 |
| **Qwen3/2B** | int8 | 262ms | 33ms | 30.7 | 771ms | 35ms | 27.8 |
| **Qwen3/8B** | int4 AWQ | 402ms | 79ms | 12.9 | 2161ms | 82ms | 12.1 |
| **Gemma-4 E2B** | int4 | 342ms | 77ms | 14.2 | 1732ms | 196ms | 10.8 |
| **Qwen3.6/35B** (思考开) | int4 | 1069ms | 88ms | 11.8 | 4518ms | 87ms | 11.6 |
| **Qwen3.6/35B** (思考关) | int4 | 1070ms | 92ms | 11.2 | 4571ms | 94ms | 10.9 |

> tok/s 基于生成文本的编码结果。中文约 1.8 字符/subword。

## 项目结构

```
ov-cli/
├── ov-cli                   # 入口脚本（自动发现 .venv）
├── pyproject.toml
├── README.md / README_EN.md
│
├── ov_cli/
│   ├── __init__.py          # 包信息 + i18n
│   ├── __main__.py          # python -m ov_cli 入口
│   ├── cli.py               # CLI 参数解析 + 命令分发 + setup
│   ├── chat.py              # 聊天/翻译终端（GenAI + Optimum）
│   ├── convert.py           # 模型转换（7 种量化）
│   ├── generate.py          # 文生图终端
│   ├── server.py            # FastAPI OpenAI 兼容服务
│   └── benchmark.py         # 性能测试
│
└── openvino.genai-2026.2.0.0-optimization/  # 修改版 GenAI 源码（setup 完整模式用）
```

## 依赖

- Python >= 3.10
- OpenVINO >= 2026.2, openvino-genai
- Optimum Intel >= 1.27.0（GitHub 源码）
- transformers >= 5.9, torch, torchvision
- GPU: Intel 集成显卡 / Arc 独显（自动检测）
- CPU: 任意 x86-64

### WSL2 支持

WSL2 下使用 Intel GPU 需额外安装 runtime：

```bash
sudo apt install intel-level-zero-gpu libze1
```

安装后 `./ov-cli` 会自动检测 GPU 可用性，若缺少 runtime 会给出提示。

## 相关链接

- [OpenVINO 文档](https://docs.openvino.ai/)
- [OpenVINO GitHub](https://github.com/openvinotoolkit/openvino)
- [OpenVINO Toolkit 仓库](https://github.com/orgs/openvinotoolkit/repositories?type=all)