# Thinking Budget 开发日志

## 问题：为什么 `--reasoning off` 在 Qwen3.6 上无效？

### 背景

Qwen3.6 是**天生思考模型**——无论 prompt 怎么写，它都会输出 `<think>...</think>` 内容的推理过程。这与 Qwen3.5 不同：

| 模型 | `--reasoning off` 效果 |
|------|----------------------|
| Qwen3.5 0.8B | ✅ 真正不思考（不加 `<think>` 标签即可） |
| Qwen3.6 35B-A3B | ❌ 总是思考（模型被训练为必须推理） |

### 为什么 prompt 方法不行？

我们尝试了所有 prompt 层面的方案：

1. **`enable_thinking=False`** → chat template 输出 `<think>\n\n</think>\n\n`（空思考块），但模型无视
2. **不加 `<think>` 标签** → 模型自创 `"Here's a thinking process:"` 照样推理
3. **System prompt 抑制** → `"不要思考，直接回答"` → 模型在思考中分析这条指令，然后无视
4. **高惩罚参数** → `presence_penalty=2.0` 等 → 模型输出崩溃（乱码、多语言混合）

**根本原因**：Qwen3.6 训练数据中所有回答都包含推理。模型**不知道如何不推理**——它必须先生成推理才能回答问题。

### 为什么 streamer 过滤是假不思考

我们可以用 streamer 过滤掉 `<think>...</think>` 内容不显示，但：
- 模型**仍然消耗时间和 token** 去推理（19s 生成 630 chars 思考内容）
- 只有显示被隐藏，计算开销一样
- 用户感知到延迟，体验差

---

## 真正解决方案：logit 级别强制

### llama.cpp 的做法

llama.cpp 有一个 **Reasoning Budget Sampler**，工作原理：

```
1. 模型计算所有 token 的概率
   → [猫:0.1, 思考:0.3, </think>:0.01, ...]
2. ↓ sampler 检测到思考预算用完
3. ↓ 把其他所有 token 的 logit 设为 -∞
   → [猫:-∞, 思考:-∞, </think>:0.01, ...]
4. ↓ 模型只能选 </think>
```

这相当于在方向盘上直接控制——无论模型多想思考，**物理上不让它选择思考 token**。

### OpenVINO GenAI 的采样架构

OV GenAI 的采样是一系列 **ILogitTransformer** 的链式调用：

```
GenerationConfig
    ↓
LogitProcessor (logit_processor.hpp)
    └── m_logit_transformers: vector<ILogitTransformer>
        ├── [0] EOSPenaltyTransform
        ├── [1] RepetitionPenaltyTransform
        ├── [2] PresencePenaltyTransform
        ├── [3] TemperatureLogitTransform
        ├── [4] TopKFilter
        └── [5] TopPFilter
                    ↓
            apply(logits)
                    ↓
            sample() → 从修改后的 logits 选 token
```

每个 `ILogitTransformer::apply(Logits& logits)` 接收一个 `Logits` 结构体：

```cpp
struct Logits {
    float * m_data;   // logit 值数组，索引即 token ID
    size_t m_size;    // 词表大小
};
```

`m_data[i]` 中，**索引 i 就是 token ID**，值是 logit（浮点数）。把某个位置的 logit 设为 `-∞ +1 禁止模型选择该 token。

这正是 llama.cpp 的做法——我们的 GenAI 已有相同的扩展点，只是没人实现过。

---

## 实现过程

### 第 1 步：添加 GenerationConfig 字段

**文件**: `src/cpp/include/openvino/genai/generation_config.hpp`

在 `GenerationConfig` 类中添加三个字段：

```cpp
int64_t reasoning_budget_tokens = -1;   // -1 = 禁用，N = 最大思考 token 数
int64_t thinking_start_token_id = -1;   // <think> 的 token ID（如 Qwen3.6 的 248068）
int64_t thinking_end_token_id = -1;     // </think> 的 token ID（如 Qwen3.6 的 248069）
```

默认值 `-1` 表示"不启用"，向后兼容所有现有模型。

### 第 2 步：添加 JSON/AnyMap 序列化

**文件**: `src/cpp/src/generation_config.cpp`

让新增字段支持从 JSON 和 `AnyMap` 读取，这样 `pipe.generate(prompt, generation_config=cfg)` 传参能正确解析。

### 第 3 步：添加 Python 绑定

**文件**: `src/python/py_generation_config.cpp`

```cpp
.def_readwrite("reasoning_budget_tokens", &GenerationConfig::reasoning_budget_tokens)
.def_readwrite("thinking_start_token_id", &GenerationConfig::thinking_start_token_id)
.def_readwrite("thinking_end_token_id", &GenerationConfig::thinking_end_token_id)
```

这样 Python 端可以：
```python
cfg = ov_genai.GenerationConfig()
cfg.reasoning_budget_tokens = 0
cfg.thinking_start_token_id = 248068
cfg.thinking_end_token_id = 248069
```

### 第 4 步：创建 ThinkingBudgetTransform

**文件**: `src/cpp/src/sampling/logit_transformers.hpp`

这是核心——一个新的 `ILogitTransformer` 实现：

```cpp
class ThinkingBudgetTransform : public ILogitTransformer {
    enum State { IDLE, COUNTING, FORCING, DONE };

    int64_t m_budget;    // 最大思考 token 数
    int64_t m_start_id;  // <think> token ID
    int64_t m_end_id;    // </think> token ID
    State m_state;
    size_t m_count;      // 已生成的思考 token 数
```

**状态机**：

```
构造 → COUNTING (count=0)
  │
  ├── apply() 被调用
  │     ├── state == COUNTING && count >= budget → FORCING
  │     └── state == FORCING → 所有 logit 置 -∞，只保留 end_id
  │
  └── accept_token(token_id) 被调用
        ├── state == COUNTING && token == end_id → DONE
        ├── state == COUNTING → count++
        └── state == FORCING && token == end_id → DONE
```

**关键设计决策**：构造函数直接设 `m_state = COUNTING`。

原因是 Qwen 的 chat template 在 prompt 末尾已经包含了 `<think>` token（token 248068）。这个 token 是 prompt 的一部分，不是生成的。而 `accept_token` 只对**生成**的 token 调用，不对 prompt token 调用。如果从 IDLE 开始，永远不会看到 `<think>`，预算永远不会激活。

所以直接从 COUNTING 开始，`<think>` 之后的第一个生成 token 就开始计数。

**强制逻辑**（在 `apply()` 中）：

```cpp
void apply(Logits& logits) override {
    if (m_state != FORCING) return;
    
    for (size_t i = 0; i < logits.m_size; ++i) {
        if (static_cast<int64_t>(i) != m_end_id) {
            logits.m_data[i] = -std::numeric_limits<float>::infinity();
        }
    }
}
```

⚠️ **这里有一个我踩过的坑**：`m_data[i]` 的索引 `i` 就是 **token ID**，值是该 token 的 **logit 分数**。不要用 `m_data[i]` 和 `m_end_id` 比较（logit vs token ID），而是用索引 `i` 和 `m_end_id` 比较。

```
// ❌ 错误：logit 值和 token ID 是两个不同维度
if (logits.m_data[i] != static_cast<float>(m_end_id))

// ✅ 正确：索引 i 就是 token ID
if (static_cast<int64_t>(i) != m_end_id)
```

### 第 5 步：接入 LogitProcessor

**文件**: `src/cpp/src/sampling/logit_processor.hpp`

在构造函数中添加条件判断：

```cpp
if (sampling_params.reasoning_budget_tokens >= 0 &&
    sampling_params.thinking_start_token_id >= 0 &&
    sampling_params.thinking_end_token_id >= 0) {
    m_thinking_budget = std::make_shared<ThinkingBudgetTransform>(
        sampling_params.reasoning_budget_tokens,
        sampling_params.thinking_start_token_id,
        sampling_params.thinking_end_token_id);
    m_logit_transformers.push_back(m_thinking_budget);
}
```

还需在 `register_new_generated_token()` 中调用 `accept_token`：

```cpp
if (m_thinking_budget) {
    m_thinking_budget->accept_token(new_token_id);
}
```

### 第 6 步：适配 VLM Pipeline

VLM pipeline 的 prompt 以 embedding 形式传入（而非 token IDs），导致 `sequence_group->get_prompt_ids()` 返回空向量。因此不能在 LogitProcessor 构造函数中用 prompt tokens 初始化 budget。

这就是为什么我们让 budget 从 COUNTING 状态开始——不依赖 prompt tokens。

### 第 7 步：处理编译依赖

**问题**：`ENABLE_GGUF=OFF` 时编译失败，因为部分 tokenizer 文件无条件引用了 GGUF 类型。

**修复**：调整了多个文件的条件编译：
- `gguf_tokenizer.hpp`：将 `load_shared_object`、`get_symbol`、`is_gguf_model` 移出 `#ifdef ENABLE_GGUF`
- `tokenizer_impl.cpp`：用 `#ifdef ENABLE_GGUF` 包裹 GGUF 专用代码块
- `add_second_input_pass.hpp/.cpp`：整个文件用 `#ifdef ENABLE_GGUF` 包裹
- `CMakeLists.txt`：将 `gguf_tokenizer.cpp` 纳入编译（但排除纯 GGUF 的 `.cpp` 文件）

---

## 最终效果

### 对比

| 指标 | 之前（streamer 过滤） | 之后（budget=0） |
|------|---------------------|-----------------|
| 输出 | 630 chars 思考 + 回答 | 28 chars 直接回答 |
| 耗时 | ~19s | ~2.5s |
| tok/s | 含大量推理 token | 纯回答 token |
| 体验 | 等待很久才有回复 | 几乎即时 |

### 性能提升明细

对于 Qwen3.6 35B-A3B 模型 + "你好" 测试：

```
之前: 19.0s / 630 chars / 33.2 ch/s  (含 600+ 思考 chars)
之后:  2.5s /  28 chars / 11.2 ch/s  (纯回答，无思考)
速度: 7.6x 提升
```

### budget=0 的含义

`reasoning_budget_tokens = 0` 表示"0 个额外的思考 token"。因为起始状态是 COUNTING（count=0），`apply()` 首次被调用时 `count >= budget`（0 >= 0）立即成立，进入 FORCING 状态，强制输出 `</think>`。

实际上模型会输出 1 个 token（"Here"）后才进入 FORCING——因为 `apply()` 先检查状态（COUNTING 不满足 → passthrough），然后采样出 "Here"，再 `accept_token("Here")` 使 count 变为 1，触发 FORCING。下一个 `apply()` 调用时强制 `</think>`。

---

## 后续优化方向

1. **`setup` 命令自动编译**：让 `./ov-cli setup` 检测到开源 genai 源码后自动编译并安装
2. **非 Qwen 模型的支持**：对于 `</think>` 标签不同的模型，自动检测 token ID
3. **Optimum 路径**：Optimum 格式（OVModelForVisualCausalLM）仍需 streamer 过滤
4. **上游 PR**：将此功能提交到 [openvino.genai](https://github.com/openvinotoolkit/openvino.genai) 仓库
