# openvino-genai-thinking

Community patched [openvino-genai](https://github.com/openvinotoolkit/openvino.genai) with `ThinkingBudgetTransform`.

When `reasoning_budget_tokens=0` is set, it **actually forces the model to stop thinking** — no more ignored empty `<think></think>` prefixes.

## Usage

```python
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("./model-ov", "GPU")
cfg = ov_genai.GenerationConfig()
cfg.max_new_tokens = 512
cfg.reasoning_budget_tokens = 0  # ← now works
cfg.thinking_start_token_id = 248068  # <think> token id
cfg.thinking_end_token_id = 248069   # </think> token id

result = pipe.generate("Hello", generation_config=cfg)
print(result)
```

## What's patched

- Added `ThinkingBudgetTransform` LogitProcessor in `logit_transformers.hpp`
- Added `reasoning_budget_tokens`, `thinking_start_token_id`, `thinking_end_token_id` fields in `GenerationConfig`
- Registered the transformer in `logit_processor.hpp`

## Compared to official

| Feature | Official | Patched |
|---------|----------|---------|
| `reasoning_budget_tokens` | ❌ Not declared | ✅ Works |
| INT4 empty-think | ❌ Ignored | ✅ Enforced via logit transformer |

## Note

This is a community patch, not an official Intel release. It will be retired once the official openvino-genai fixes the issue.
