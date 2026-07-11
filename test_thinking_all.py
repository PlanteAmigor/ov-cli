#!/usr/bin/env python3
"""测试所有 LLM/VLM 模型的 thinking suppression"""

import openvino_genai
import time

MODEL_DIR = '/home/amigor/ov-cli/model'

THINKING_MODELS = [
    ('Qwen3/14B-ov',              False, 'Qwen3-14B-LLM'),
    ('Qwen3/30B-A3B-ov-int4',     False, 'Qwen3-30B-MoE-LLM'),
    ('deepseek/7B-ov',            False, 'DeepSeek-7B-LLM'),
    ('Qwen3/2B-ov-int4',          True,  'Qwen3-2B-VLM'),
    ('Qwen3/8B-ov-int4-v2',       True,  'Qwen3-8B-VLM'),
    ('Qwen3.5/0.8B-ov-int4-v2',   True,  'Qwen3.5-0.8B-VLM'),
    ('Qwen3.5/4B-ov-int4',        True,  'Qwen3.5-4B-VLM'),
    ('Qwen3.6/27B-ov-int4',       True,  'Qwen3.6-27B-VLM'),
    ('gemma/E2B-ov',              True,  'Gemma-E2B-VLM'),
]

def get_think_token_ids(tok):
    try:
        enc = tok.encode('<think>', add_special_tokens=False)
        ids = list(enc.input_ids.data)
        think_id = int(ids[0][0]) if ids and hasattr(ids[0], '__len__') else int(ids[0])
    except Exception:
        think_id = -1
    try:
        enc = tok.encode('</think>', add_special_tokens=False)
        ids = list(enc.input_ids.data)
        nothink_id = int(ids[0][0]) if ids and hasattr(ids[0], '__len__') else int(ids[0])
    except Exception:
        nothink_id = -1
    return think_id, nothink_id

def test_model(rel_path, is_vlm, name):
    print()
    print('=' * 60)
    print(name)
    print('=' * 60)

    try:
        t0 = time.time()
        Pipeline = openvino_genai.VLMPipeline if is_vlm else openvino_genai.LLMPipeline
        pipe = Pipeline(MODEL_DIR + '/' + rel_path, 'CPU')
        tok = pipe.get_tokenizer()
        print('  加载: %.1fs' % (time.time() - t0))
    except Exception as e:
        print('  FAIL load:', e)
        return

    think_id, nothink_id = get_think_token_ids(tok)
    print('  <think>=%d  </think>=%d' % (think_id, nothink_id))

    # 默认
    t0 = time.time()
    r = pipe.generate('Hello', max_new_tokens=48)
    out1 = str(r)
    h1 = '<think>' in out1
    print('  默认: <think>=%s | %.1fs | %s...' % (h1, time.time() - t0, out1[:120]))

    # enable_thinking=False
    t0 = time.time()
    cfg = openvino_genai.GenerationConfig()
    cfg.enable_thinking = False
    cfg.thinking_start_token_id = think_id
    cfg.thinking_end_token_id = nothink_id
    cfg.max_new_tokens = 48
    r = pipe.generate('Hello', generation_config=cfg)
    out2 = str(r)
    h2 = '<think>' in out2
    prefix = '</think>' if out2.startswith('</think>') else out2[:20]
    print('  关闭: <think>=%s | 开头=%s | %.1fs | %s...' % (h2, prefix, time.time() - t0, out2[:150]))

    # budget=5
    t0 = time.time()
    cfg = openvino_genai.GenerationConfig()
    cfg.reasoning_budget_tokens = 5
    cfg.thinking_start_token_id = think_id
    cfg.thinking_end_token_id = nothink_id
    cfg.max_new_tokens = 48
    r = pipe.generate('What is 2+2?', generation_config=cfg)
    out3 = str(r)
    h3 = '<think>' in out3
    print('  budget=5: <think>=%s | %.1fs | %s...' % (h3, time.time() - t0, out3[:150]))

    del pipe

def main():
    for rel_path, is_vlm, name in THINKING_MODELS:
        test_model(rel_path, is_vlm, name)

if __name__ == '__main__':
    main()
