# OpenCode 配置 ov-cli 本地模型

[OpenCode](https://opencode.ai) 是一款开源的 AI 编码助手 CLI，支持通过 OpenAI 兼容 API 连接本地模型。

## 前提

已启动 ov-cli server：

```bash
./ov-cli server --model ./model-ov --device GPU.1
```

确认 API 可用：

```bash
curl http://localhost:8080/v1/models
# 应返回模型列表
```

## 配置步骤

### 1. 编辑配置文件

`~/.config/opencode/opencode.jsonc`：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ov-cli": {                            // 提供商 ID，可自定义
      "npm": "@ai-sdk/openai-compatible",  // OpenAI 兼容 API 适配器
      "name": "ov-cli (local)",            // 界面显示名称
      "options": {
        "baseURL": "http://localhost:8080/v1",
        "apiKey": "dummy"                  // 本地服务任意值即可
      },
      "models": {
        "4B-ov-int4": {                    // 模型 ID，需与 /v1/models 返回一致
          "name": "Qwen3.5-4B (local)"
        }
      }
    }
  }
}
```

> 模型 ID 可通过 `curl http://localhost:8080/v1/models` 查看。

### 2. 启动 OpenCode

```bash
PATH="$HOME/.opencode/bin:$PATH" opencode
```

### 3. 选择模型

在 TUI 中按 `/` 输入命令：

```
/models
```

选择 `ov-cli (local)` → `Qwen3.5-4B (local)`。

## 多模型配置

如果切换模型，将 `models` 下的 ID 和名称替换为对应的模型名即可。
