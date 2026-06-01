# ov-cli

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-≥3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/OpenVINO-≥2026.2-purple" alt="OpenVINO">
  <img src="https://img.shields.io/badge/platform-Linux%20|%20Windows-lightgrey" alt="Platform">
  <img src="https://img.shields.io/github/stars/PlanteAmigor/ov-cli?style=flat&label=stars" alt="Stars">
</p>

**OpenVINO LLM 命令行工具** — 轻量、离线、CPU/GPU 皆可运行。

基于 Optimum Intel + OpenVINO GenAI。支持模型转换（FP32/FP16/INT8/INT4）、交互式聊天（流式输出）、翻译。

## 快速开始

```bash
# 1. 一键创建环境（自动安装所有依赖 + 应用 Gemma-4 补丁）
./ov-cli setup
eval "$(./ov-cli venv)"          # 进入虚拟环境

# 2. 转换模型
./ov-cli convert --model ./Qwen3/2B --format int8

# 3. 聊天
./ov-cli chat --model ./Qwen3/2B-ov
```

## 命令

### `setup` — 创建环境

创建 Python 虚拟环境并安装所有依赖（`openvino-genai`、`optimum-intel`、`transformers 5.9`、`torch` 等）。
安装完成后自动应用 Gemma-4 共享 KV 层补丁。

```bash
./ov-cli setup                          # 默认 ./.venv（交互选择安装模式）
./ov-cli setup --venv ./my-venv         # 指定路径
./ov-cli setup --optimum-dir ./optimum-intel-main  # 指定 optimum 源码

完整模式下，setup 会自动从源码编译 openvino-genai 以启用 thinking budget 功能
（实现 logit 级别的 `</think>` 强制结束思考），仅 **Linux** 支持。
```

### `venv` — 进入虚拟环境

打印虚拟环境的 activate 路径，配合 `eval` 使用：

```bash
eval "$(./ov-cli venv)"                 # 一键激活
eval "$(./ov-cli venv --venv ./my-venv)"
```

### `convert` — 模型转换

使用 Optimum Intel 官方工具将 HuggingFace 模型导出为 OpenVINO IR，自动推断 task 类型。

```bash
# 基本用法
./ov-cli convert --model ./Qwen3/2B --format int8
./ov-cli convert --model ./Qwen3/2B --format int4 -o ./Qwen3/2B-ov-int4
./ov-cli convert --model ./model --format fp16       # 半精度
```

**量化格式**：

| 格式 | 体积 (相对 fp32) | 说明 |
|------|-----------------|------|
| `fp32` | 100% | 无损，体积最大 |
| `fp16` | ~50% | 半精度，几乎无损 |
| `int8` | ~25% | 8-bit，几乎无损 |
| `int4` | ~12.5% | 4-bit，有精度损失 |

**高级参数**：

```bash
# INT4 混合精度（80% INT4 + 20% INT8）
./ov-cli convert --model ./Hy-MT2/1.8B --format int4 --ratio 0.8 --group-size 128
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--ratio` | 1.0 | INT4 比例 (0-1)，越小 INT8 越多 |
| `--group-size` | 128 | 量化分组大小 |

### `chat` — 聊天终端

加载 OpenVINO 模型并启动交互终端。自动识别三种模型格式，支持流式输出、多轮对话、图片（VLM）。

```bash
# 聊天模式（自动检测格式）
./ov-cli chat --model ./Qwen3/2B-ov                               # GenAI 格式
./ov-cli chat --model ./gemma-4-E2B-it-ov                          # Optimum 格式（Gemma-4 等）
./ov-cli chat --model ./model-ov --temp 0.9 --max-tokens 2048

# 控制思考/推理（仅 GenAI 格式）
./ov-cli chat --model ./Qwen3/2B-ov --reasoning on                 # 开启思考（默认）
./ov-cli chat --model ./Qwen3.5/0.8B-ov --reasoning off            # 关闭思考（Qwen3.5 等有效）
./ov-cli chat --model ./Qwen3.6/35B-A3B-ov --reasoning off         # 关闭思考（需编译修改版 GenAI，见下文）

# 翻译模式（Hy-MT2 等翻译模型）
./ov-cli chat --model ./Hy-MT2-1.8B-ov --mode translate

# VLM 图片支持
./ov-cli chat --model ./model-vlm-ov --image ./photo.jpg

# 英文界面
./ov-cli --lang en chat --model ./model-ov
```

**终端内命令**：

| 命令 | 说明 |
|------|------|
| `//img PATH` | 加载/切换图片（VLM 模型） |
| `/temp 0.7` | 设置温度 (0-2) |
| `/system ...` | 设置系统提示词 |
| `/clear` | 清空对话上下文 |
| `/help` | 帮助 |
| `/exit` | 退出 |

**翻译模式命令**：

| 命令 | 说明 |
|------|------|
| `//en 文本` | 强制翻译为英语 |
| `//zh 文本` | 强制翻译为中文 |
| 直接输入 | 自动检测语言方向 |

## 支持模型

### ✅ 两种推理格式

| 格式 | 加载方式 | 适用模型 | 特征 |
|------|---------|---------|------|
| **GenAI** | `LLMPipeline` / `VLMPipeline` | 常规 optimum-cli 导出模型 | `openvino_config.json`，无逐层输入 |
| **Optimum** | `OVModelForVisualCausalLM` | Gemma-4 等 VLM | 有 `openvino_text_embeddings_per_layer_model.xml` |

### ✅ 实测验证（✓ = 实际跑通过）

| 模型 | 格式 | 文字 | 图片 | 翻译 | 说明 |
|------|------|:----:|:----:|:----:|------|
| **Hy-MT2 1.8B** | GenAI | | | ✅ | `--mode translate`，FP32/FP16/INT8/INT4 四种精度全通过 |
| **Gemma-4 E2B** | Optimum | ✅ | ✅ | | INT4 导出+推理，需 `kv_shared_layer` 补丁 |
| **Qwen3-VL 8B** | GenAI | ✅ | ✅ | | 官方预转换版，ModelScope 下载 |
| **Qwen3.5 0.8B** | GenAI | ✅ | ❌ | | INT8，视觉编码器有 bug |
| **Qwen3.6 35B-A3B** | GenAI | ✅ | ✅ | | MoE，混合精度，官方预转换 |
| **Qwen3 2B** | GenAI | ✅ | ❌ | | 文字正常，自转视觉编码器 reshape 有 bug |

### 📌 适用范围

理论上支持所有 transformers 标准架构（Llama、Mistral、DeepSeek、Phi、Gemma 等），只需 `optimum-cli` 能成功导出即可推理。

### ⚠️ 注意事项

- **Gemma-4**：导出需修改 `model_patcher.py` 中 `kv_shared_layer_index` → `layer_type`，`setup` 命令会自动打补丁
- **Qwen3-VL 小模型**：自转 2B 视觉编码器导出有 bug（`aten::view/Reshape` 形状不匹配）；Qwen3.5 0.8B 视觉编码器相同问题。官方预转换 8B 和 35B-A3B 正常
- **Ctrl+C 中断延迟**：生成期间按 Ctrl+C 可中断，但最坏情况下需等待当前 token 生成完毕（约 20-200ms 不等），无法达到像 llama.cpp 的即时中断。中断时 `^C` 字符可能出现在输出中
- **`--reasoning off` 对思考型模型**：Qwen3.6 等天生思考的模型通过 prompt 无法真正禁用推理。`ov-cli` 通过修改 OpenVINO GenAI 源码（`ThinkingBudgetTransform`）实现了 logit 级别的 `</think>` 强制结束思考，效果类似 llama.cpp 的 reasoning budget sampler。
  - 使用 `setup` 完整模式（交互选 2）自动编译安装修改版 GenAI，仅 **Linux** 支持
  - 简易模式（默认）下 `--reasoning off` 仅过滤显示 `<think>...</think>` 内容，不节省时间
- 预转换 OpenVINO 模型可在 [ModelScope OpenVINO 组织](https://www.modelscope.cn/organization/OpenVINO) 查找

## 项目结构

```
ov-cli/
├── ov-cli                   # Shell 入口脚本（自动发现 .venv）
├── pyproject.toml
├── README.md
├── GEMMA4-DEVLOG.md         # Gemma-4 适配开发日志
├── KV-CACHE-DEVLOG.md       # KV Cache 探索开发日志
│
├── ov_cli/
│   ├── __init__.py          # 包信息 + i18n
│   ├── __main__.py          # python -m ov_cli 入口
│   ├── cli.py               # CLI 参数解析 + 命令分发 + setup 自动打补丁
│   ├── chat.py              # 聊天/翻译终端（GenAI + Optimum 双格式）
│   └── convert.py           # optimum-cli 模型转换
│
├── openvino.genai-2026.2.0.0-optimization/  # 修改版 GenAI 源码（setup 完整模式用）
│
├── model/                   # 模型文件（用户自行下载/转换）
│   ├── Qwen3/
│   ├── Qwen3.5/
│   ├── gemma/
│   └── Hy-MT2/
│
└── optimum-intel-main/      # optimum-intel 源码（可选）
```

## 性能参考

测试环境: Intel Arc Pro 130T/140T (Arrow Lake-P) GPU | openvino-genai 2026.2 | 3 轮 "你好" 预热后

| 模型 | 量化 | 32 1st | 32 2nd | 32 tok/s | 1024 1st | 1024 2nd | 1024 tok/s | RSS |
|:-----|:----:|:------:|:------:|:--------:|:---------:|:---------:|:----------:|:---:|
| **Qwen3.5/0.8B-ov** | int8 | 297ms | 19ms | **54.9** | 660ms | 20ms | **51.8** | 826MB |
| **Qwen3.5/0.8B-ov** | int8 | 308ms | 19ms | **56.0** | 630ms | 19ms | **51.7** | 812MB |
| **Hy-MT2/1.8B-ov** | int4 | 267ms | 25ms | 40.6 | 710ms | 24ms | 38.2 | 916MB |
| **Hy-MT2/INT8** | int8 | 232ms | 25ms | 34.4 | 646ms | 33ms | 32.2 | 1033MB |
| **Qwen3/2B-ov** | int8 | 262ms | 33ms | 30.7 | 771ms | 35ms | 27.8 | 1207MB |
| **Qwen3/8B-ov** | int4 AWQ | 402ms | 79ms | 12.9 | 2161ms | 82ms | 12.1 | 2010MB |
| **Gemma-4-E2B-ov-test** | int4 | 342ms | 77ms | 14.2 | 1732ms | 196ms | 10.8 | 8278MB |
| **Qwen3.6/35B-A3B-ov** | int4/8 mix | 1069ms | 88ms | 11.8 | 4518ms | 87ms | 11.6 | 1013MB |

> tok/s 对应 BPE subword token，中文约 1.8 字符/subword。

## 工作流

```
HuggingFace 模型
      │
      ▼
optimum-cli export openvino  ─── 自动推断 task + 量化
      │
      ├──→ GenAI 格式 ──→ LLMPipeline / VLMPipeline
      │                     ├── chat  ── 流式、多轮对话、图片（VLM）
      │                     └── translate ── 自动检测方向、33 种语言
      │
      └──→ Optimum 格式 ──→ OVModelForVisualCausalLM
                              └── chat  ── 流式、多轮对话、图片（Gemma-4）
```

## 依赖

- Python >= 3.10
- OpenVINO >= 2026.2, openvino-genai
- Optimum Intel >= 1.27.0（GitHub 源码）
- transformers >= 5.9, torch, torchvision
- 支持 GPU: Intel 集成显卡 / Arc 独显（自动检测，GPU 优先）
- 支持 CPU: 任意 x86-64 处理器

## Windows 支持

Windows 上基本可用，以下注意事项：

| 项目 | 说明 |
|------|------|
| **入口** | 使用 `ov-cli.bat` 替代 `./ov-cli`；或 `python -m ov_cli` |
| **setup** | 自动检测 Windows 路径（`Scripts\` 而非 `bin/`） |
| **多行输入** | 退化为单行输入（`select` 在 Windows 不支持） |
| **benchmark RSS** | 暂不采集（`resource.getrusage` 仅 Unix） |
| **不支持** | shell 入口脚本 `ov-cli`（bash 语法） |

```bash
# Windows 用法
ov-cli.bat setup
ov-cli.bat chat --model ./model-ov

# 或直接 Python
python -m ov_cli setup
python -m ov_cli chat --model ./model-ov
```
