<p align="center">
  <img src="ov-cli.png" alt="ov-cli logo" width="600">
</p>

<p align="center"><b><a href="README.md">中文</a> | <a href="README_EN.md">English</a></b></p>

> 我发现官方的 OpenVINO 工具在日常 LLM 实验中较为繁琐，因此我开发了 ov-cli 作为轻量级的替代方案。
> 代码几乎全部由 **DeepSeek V4 Flash** 自动生成，我只负责引导和方向把控。

**OpenVINO LLM 命令行工具** — 轻量、离线、CPU/GPU 皆可运行。

> 💡 **切换英文界面**：所有命令前加 `--lang en`，例如 `./ov-cli --lang en chat --model ./model-ov`

基于 Optimum Intel + OpenVINO GenAI 推理引擎。支持模型转换（7 种量化格式）、交互式聊天（流式/翻译/图片）、OpenAI 兼容 API 服务。

## 快速开始

```bash
# 1. 一键创建环境（自动安装所有依赖）
./ov-cli setup
source .venv/bin/activate

# 2. 转换模型（HuggingFace → OpenVINO IR）
./ov-cli convert --model ./Qwen3/2B --format int8

# 3. 聊天终端
./ov-cli chat --model ./Qwen3/2B-ov

# 4. 文生图
./ov-cli image --model ./FLUX/ov-int4

# 5. TTS 语音合成
./ov-cli tts --model ./0.6B-CV-ov --prompt 你好 --speaker Vivian

# 6. API 服务
./ov-cli server --model ./Qwen3/2B-ov

# 7. Web 界面
./ov-cli ui --model ./Qwen3/2B-ov

# 8. MCP 协议
./ov-cli mcp --model ./Qwen3/2B-ov

# 9. 管道模式（批量/外部调用）
printf '你好\n再见' | ./ov-cli chat --model ./model-ov --mode pipe
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

创建 Python 虚拟环境，安装依赖。支持按需安装，只装你需要的模块。

```bash
# 按需安装（只装需要的模块）
./ov-cli setup --with chat                          # 只装聊天终端
./ov-cli setup --with chat,image,asr                # 聊天 + 文生图 + 语音识别
./ov-cli setup --with all                           # 全装

# 指定 venv 路径
./ov-cli setup --venv ./my-venv --with chat,convert

# 修复模式（不重建，仅升级已安装的模块）
./ov-cli setup --fix
```

> 系统依赖：`setup` 会自动检测 `sox`、`libsndfile1` 等系统包是否安装，缺失时给出安装提示。

**可选模块：**（按需安装，只装你需要的包，不装多余的依赖）

| 模块 | 说明 | 额外依赖 |
|------|------|---------|
| `chat` | 聊天/翻译终端 | PyMuPDF, soxr |
| `image` | 文生图 | — |
| `asr` | 语音识别 | soundfile, scipy, qwen-asr |
| `tts` | 语音合成 | soundfile, sox, qwen-tts |
| `ui` | Gradio Web 界面 | gradio |
| `mcp` | MCP 协议服务器 | — |
| `server` | API 服务器 | fastapi, uvicorn |
| `convert` | 模型转换 | torch, torchvision, optimum-intel（约 3GB，耗时 5-10 分钟） |

**模式选择**（仅装 `chat` 时提示）：
1. **简易模式** — pip 安装，日常使用。`--reasoning off` 对思考型模型无效。
2. **完整模式** — 从源码编译 OpenVINO GenAI 以启用 thinking budget 功能
   （logit 级别的 `</think>` 强制结束思考）。

**修复模式** (`--fix`)：不重建虚拟环境，仅升级已安装的模块、重打补丁，
数秒完成。

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

# 管道模式（模型常驻，批量处理）
printf '问题1\n问题2\n' | ./ov-cli chat --model ./model-ov --mode pipe
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

### `ui` — Web 界面

启动 Gradio Web UI，自动检测模型类型（chat/tts/asr/image），提供对应的可视化界面。

```bash
# 聊天界面
./ov-cli ui --model ./Qwen3/2B-ov
./ov-cli ui --model ./Qwen3/8B-ov --port 7861                      # 指定端口
./ov-cli ui --model ./model-vlm-ov --share                          # 公开链接
./ov-cli ui --model ./deepseek/7B-ov --reasoning off                # 关闭思考

# TTS / ASR / 文生图界面
./ov-cli ui --model ./0.6B-CV-ov                                    # TTS
./ov-cli ui --model ./Qwen3-ASR-0.6B-ov                             # ASR
./ov-cli ui --model ./FLUX/ov-int4                                  # 文生图
```

**聊天界面功能**：
- 流式输出、多轮对话
- 图片上传（VLM 模型）
- 对话历史保存 / 加载 / 删除
- `Ctrl+C` 安全退出

### `mcp` — MCP 协议服务器

启动 MCP (Model Context Protocol) 服务器，通过 stdin/stdout JSON-RPC 暴露 LLM 工具。
可被 VS Code Copilot (agent 模式)、Cursor、Claude Desktop 等支持 MCP 的 AI 编程工具调用。

```bash
./ov-cli mcp --model ./Qwen3/2B-ov
./ov-cli mcp --model ./deepseek/7B-ov
```

**暴露的工具：**

| 工具 | 说明 |
|------|------|
| `chat` | 向本地 LLM 发送提示并获取回复 |
| `chat_stream` | 流式聊天，逐块返回文本 |

**VS Code 配置**（`.vscode/mcp.json`）例如（将路径替换为你的实际路径）：
```json
{
  "servers": {
    "ov-cli": {
      "command": "/run/media/amigor/Project/ov-cli/.venv/bin/ov-cli",
      "args": ["mcp", "--model", "/run/media/amigor/Project/ov-cli/model/deepseek/7B-ov"],
      "type": "stdio",
      "description": "本地 LLM 推理（聊天、翻译、问答）"
    }
  }
}
```

**其他平台**（Cursor → `.cursor/mcp.json`，Claude Desktop → `claude_desktop_config.json`），格式基本一致。

### `tts` — 语音合成

使用 OpenVINO Qwen3-TTS 生成语音，仅支持单次。

**CustomVoice**（预设声音）：

```bash
./ov-cli tts --model ./0.6B-CV-ov --prompt "今天天气真好" --speaker Vivian
./ov-cli tts --model ./0.6B-CV-ov --prompt "Hello" --speaker vivian --lang english
./ov-cli tts --model ./0.6B-CV-ov --prompt "你好" --speaker Vivian --instruct "温柔地" -o voice.wav

# 管道模式（模型常驻，批量合成）
printf '你好\n再见\n' | ./ov-cli tts --model ./0.6B-CV-ov --mode pipe --speaker Vivian
```

**Base（声音克隆）**（需参考音频）：

```bash
./ov-cli tts --model ./0.6B-ov --prompt "你好" --ref-audio ref.mp3 -o voice.wav
./ov-cli tts --model ./0.6B-ov --prompt "Hello" --ref-audio ref.mp3 --lang english
```

### `asr` — 语音转文字

自动识别 Whisper / Qwen3-ASR。**推荐 Qwen3-ASR**（自动加标点、语种识别、支持 52 种语言和方言）。

**Qwen3-ASR**（推荐）：

```bash
# 交互式
./ov-cli asr --model ./Qwen3-ASR-0.6B-ov

# 单次（输出完自动退出）
./ov-cli asr --model ./Qwen3-ASR-0.6B-ov --mode once --file speech.mp3
./ov-cli asr --model ./Qwen3-ASR-0.6B-ov --mode once --file speech.mp3 --json

# 管道模式（模型常驻，批量转录）
printf '/path/to/a.wav\n/path/to/b.wav\n' | ./ov-cli asr --model ./Qwen3-ASR-0.6B-ov --mode pipe
```

**Whisper**：

```bash
./ov-cli asr --model ./whisper/ov-large
./ov-cli asr --model ./whisper/ov-large --mode once --file speech.mp3
```

> Qwen3-ASR 基于语义理解自动添加标点，Whisper 依赖音频停顿加标点。TTS 生成的匀速音频在 Whisper 下可能缺少标点。
TTS（文字转语音）生成的音频语速均匀、缺少自然停顿，转录结果可能不含句号逗号等标点，属正常现象。

### `image` — 文生图

使用 OpenVINO GenAI Text2ImagePipeline 生成图片，支持交互式和单次。

```bash
# 交互式（多轮生图）
./ov-cli image --model ./FLUX/ov-int4

# 单次（输出完自动退出）
./ov-cli image --model ./FLUX/ov-int4 --mode once --prompt "cat" -o cat.png
./ov-cli image --model ./FLUX/ov-int4 --mode once --prompt "cat" --seed 42 -o cat.png  # 固定种子
./ov-cli image --model ./FLUX/ov-int4 --mode once --prompt "cat" --json       # JSON 格式输出

# 管道模式（模型常驻，批量生图）
printf 'a cat\na dog\n' | ./ov-cli image --model ./FLUX/ov-int4 --mode pipe
```

**终端内命令**（仅文生图交互模式）：

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

ov-cli 支持两种外部调用方式：`--mode once`（单次）和 `--mode pipe`（管道）。

### 方式一：`--mode once` — 单次推理

每次调用独立加载模型，适合低频使用。日志走 stderr，stdout 只输出纯净结果。

| 命令 | once 模式 | `--json` | stdout 输出 |
|:----|:---------:|:--------:|:------------|
| `chat` | `--mode once --prompt TEXT [--file ...]` | ✅ | 回复文本 / `{"text":"...","time":n}` |
| `asr` | `--mode once --file audio.mp3` | ✅ | 转录文本 / `{"text":"...","time":n,"duration":n}` |
| `image` | `--mode once --prompt "cat" [-o output.png]` | ✅ | 图片路径 / `{"path":"...","time":n}` |
| `tts` | `--prompt TEXT` | ✅ | 音频路径 / `{"path":"...","time":n,"duration":n}` |

### 方式二：`--mode pipe` — 管道模式（推荐）

模型常驻内存，stdin 输入、stdout 输出 JSON，适合高频批量调用。

```bash
# ASR 批量转录
printf '/path/to/a.wav\n/path/to/b.wav\n' | ov-cli asr --model ./model --mode pipe
# → {"text":"...","time":0.5}
# → {"text":"...","time":0.4}

# TTS 批量合成
printf '你好\n再见\n' | ov-cli tts --model ./model --mode pipe --speaker Vivian
# → {"path":"outputs/pipe_1.wav","text":"你好","time":2.1}

# Chat 批量问答
printf '1+1=?\n什么是Python?\n' | ov-cli chat --model ./model --mode pipe
# → {"text":"1+1=2","time":3.2}

# Image 批量生图
printf 'a cat\na dog\n' | ov-cli image --model ./FLUX-ov --mode pipe
# → {"path":"outputs/pipe_cat.png","time":10.2}
```

### Python 调用示例

```bash
# Shell 脚本：捕获纯文本结果
text=$(/path/to/ov-cli asr -m ./model --mode once -f speech.mp3 2>/dev/null)

# Shell 脚本：捕获 JSON
json=$(/path/to/ov-cli asr -m ./model --mode once -f speech.mp3 --json 2>/dev/null)
```

```python
# Python 子进程调用（--mode once 单次）
import subprocess, json

result = subprocess.run([
    "/path/to/ov-cli", "asr",
    "--model", "./model",
    "--mode", "once",
    "--file", "speech.mp3",
    "--json"
], capture_output=True, text=True)

if result.returncode == 0:
    data = json.loads(result.stdout)
    print(data["text"])  # 转录结果

# Python 子进程调用（--mode pipe 常驻）
proc = subprocess.Popen(
    ["/path/to/ov-cli", "asr", "--model", "./model", "--mode", "pipe"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
)
for audio in ["a.wav", "b.wav"]:
    proc.stdin.write(audio + "\n")
    proc.stdin.flush()
    data = json.loads(proc.stdout.readline())
    print(data["text"])  # 每次 ~0.5s
proc.stdin.close()
```

> 💡 `2>/dev/null` 可省略日志，只保留 stdout 的结果。不加则终端同时显示日志和结果。

## 模型支持

### 推理格式

| 格式 | 加载方式 | 特征 |
|------|---------|------|
| **GenAI** | `LLMPipeline` / `VLMPipeline` | 标准 optimum-cli 导出，`openvino_config.json` |
| **Optimum** | `OVModelForVisualCausalLM` + `AutoProcessor` | Gemma-4 等，有 `openvino_text_embeddings_per_layer_model.xml` |

### 大语言模型

#### 实测验证

| 模型 | 格式 | 文字 | 图片 | 翻译 | 说明 |
|------|------|:----:|:----:|:----:|------|
| **Hy-MT2 1.8B** | GenAI | | | ✅ | 翻译模型，4 种精度全通过 |
| **Gemma-4 E2B** | Optimum | ✅ | ✅ | | INT4，需 `kv_shared_layer` 补丁 |
| **Qwen3-VL 8B** | GenAI | ✅ | ✅ | | 官方预转换 |
| **Qwen3.6 35B-A3B** | GenAI | ✅ | ✅ | | MoE，官方预转换 |
| **Qwen3.5 0.8B** | GenAI | ✅ | ❌ | | 小模型 VLM 不支持 |
| **Qwen3 2B** | GenAI | ✅ | ❌ | | 视觉编码器 reshape 有 bug |

> **VLM 说明**：GenAI `VLMPipeline` 只支持 **Qwen3-VL 8B**、**Qwen3.6 35B-A3B**、**Qwen3.5 35B-A3B** 的视觉能力。小模型（0.8B、2B）视觉编码器在 OpenVINO 上无法正常工作。

#### 自行转换

`./ov-cli convert` 支持以下架构（已验证）：

| 架构 | 说明 |
|------|------|
| Qwen3 / Qwen3.5 / Qwen3.6 | 含 MoE 变体 |
| Hy-MT2 | 多语言翻译模型 |
| Llama / Mistral / DeepSeek / Phi / Gemma | 标准 transformers 架构 |

### 语音模型

#### TTS（文字转语音）

**推荐 Qwen3-TTS**（效果最佳、功能最全）：

| 方案 | 类型 | 特点 | 命令 |
|:----|:----|:-----|:-----|
| **Qwen3-TTS** ⭐ | 自定义 OV | 预设声音 / 声音克隆 / 10 种语言 / 情感控制 | `ov-cli convert --model ./Qwen3-TTS-0.6B-CV --output ./0.6B-CV-ov` |
| **SpeechT5** | GenAI Pipeline | 轻量（600M），CPU 实时，英文 | 下载预转换模型 |

**Qwen3-TTS**（推荐）：

支持两种模型类型，自动检测：

| 类型 | 特点 | 转换命令 |
|:----|:-----|:---------|
| **CustomVoice** | 9 种预设声音，不需参考音频 | `ov-cli convert --model ./Qwen3-TTS-0.6B-CV --output ./0.6B-CV-ov` |
| **Base** | 声音克隆，需提供参考音频 | `ov-cli convert --model ./Qwen3-TTS-0.6B --output ./0.6B-ov` |

转换时自动安装 `qwen-tts` 依赖，完成后恢复。

> TTS/ASR 推理时自动切换 transformers 版本（运行时临时降级到 4.x 兼容版本，退出后自动恢复）。

```bash
# CustomVoice — 预设声音
ov-cli tts --model ./0.6B-CV-ov --prompt "你好" --speaker Vivian

# Base — 声音克隆（需参考音频）
ov-cli tts --model ./0.6B-ov --prompt "你好" --ref-audio ref.mp3
```

#### ASR（语音转文字）

支持两种方案，**推荐 Qwen3-ASR**（自动加标点、语种识别）。

| 方案 | 类型 | 特点 |
|:----|:----|:-----|
| **Qwen3-ASR** ⭐ | 自定义 OV | 语义加标点 / 52 种语言方言 / 语种识别 |
| **Whisper** | GenAI Pipeline | 轻量，交互式流畅 |

**Qwen3-ASR** 转换：
```bash
ov-cli convert --model ./Qwen3-ASR-0.6B --output ./Qwen3-ASR-0.6B-ov
```

**Whisper** 请下载官方预转换模型：
- [HuggingFace Speech-to-Text 合集](https://huggingface.co/collections/OpenVINO/speech-to-text)
- [ModelScope Speech-to-Text 合集](https://www.modelscope.cn/collections/Speech-to-Text-b9ab5c24c32649)

### 图像模型

#### 文生图（Text-to-Image）

如 FLUX、SD3.5。当前 `convert` 不支持转换文生图模型，请下载官方预转换模型：
- [HuggingFace Image Generation 合集](https://huggingface.co/collections/OpenVINO/image-generation)
- [ModelScope 文生图合集](https://www.modelscope.cn/collections/Image-Generation-eb38cde2fa3d46)

### 预转换模型来源

OpenVINO 官方提供了大量预转换模型，**下载即用**：

- [HuggingFace OpenVINO 模型库](https://huggingface.co/OpenVINO)
- [ModelScope OpenVINO 模型库](https://www.modelscope.cn/organization/OpenVINO?tab=model)

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
| **DeepSeek-R1-7B** | int4 | 344ms | 65ms | 16.0 | 1816ms | 67ms | 15.5 |
| **Qwen3/8B** | int4 AWQ | 402ms | 79ms | 12.9 | 2161ms | 82ms | 12.1 |
| **Qwen3/14B** | int4 | 506ms | 388ms | 8.0 | 3045ms | 268ms | 7.6 |
| **Gemma-4 E2B** | int4 | 342ms | 77ms | 14.2 | 1732ms | 196ms | 10.8 |
| **Gemma-4 31B** | int4 | 1541ms | 281ms | 3.6 | 9243ms | 370ms | 3.3 |
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
│   ├── image.py             # 文生图终端
│   ├── tts.py               # TTS 语音合成终端
│   ├── asr.py               # 语音转文字终端
│   ├── server.py            # FastAPI OpenAI 兼容服务
│   ├── mcp.py               # MCP 协议服务器
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

## Contributors

- **PlanteAmigor** — 引导和方向把控
- **DeepSeek V4 Flash** — 代码生成

## 相关链接

- [OpenVINO 文档](https://docs.openvino.ai/)
- [OpenVINO GitHub](https://github.com/openvinotoolkit/openvino)
- [OpenVINO Toolkit 仓库](https://github.com/orgs/openvinotoolkit/repositories?type=all)