"""
ov-cli benchmark: 模型性能基准测试。
"""

import os, sys, time, json
import resource


def _make_prompt(target_tokens):
    """生成约 target_tokens 个 token 的文本。中文约 1.8 字符/token。"""
    ch = "你好，今天天气真不错。让我们一起探索人工智能的奥秘吧！"
    base = "请用中文回答以下问题：" + " ".join([ch] * (target_tokens // 8 + 1))
    return base


def _measure_rss():
    """返回当前 RSS (MB)。Windows 返回 0。"""
    import sys as _sys
    if _sys.platform == "win32":
        return 0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def _run_genai_bench(pipe, input_size, model_path):
    """GenAI 格式基准测试。"""
    prompt = _make_prompt(input_size)
    test_prompt = prompt + "请详细解释这段话的含义。"

    # 预热
    pipe.generate("你好", max_new_tokens=10)

    # 正式测试
    t0 = time.perf_counter()
    out = pipe.generate(test_prompt, max_new_tokens=128)
    total = time.perf_counter() - t0

    # 从 output 提取指标
    # GenAI 的 generate 返回完整文本，无法直接分离首 token 延迟
    # 改用 streamer 方式测量
    return _run_genai_bench_detailed(pipe, test_prompt)


def _run_genai_bench_detailed(pipe, prompt):
    """用 streamer 精确测量各指标。"""
    import openvino_genai as ov_genai

    rss_before = _measure_rss()

    # 计时：首 token / 第二 token / 总时间
    first_token_time = None
    second_token_time = None
    last_token_time = None
    pieces = 0
    all_text = []

    def streamer(t):
        nonlocal first_token_time, second_token_time, last_token_time, pieces
        now = time.perf_counter()
        all_text.append(t)
        pieces += 1
        if first_token_time is None:
            first_token_time = now
        elif second_token_time is None:
            second_token_time = now
        last_token_time = now
        return False

    cfg = ov_genai.GenerationConfig(max_new_tokens=128)
    t_start = time.perf_counter()
    if isinstance(pipe, ov_genai.VLMPipeline):
        pipe.generate(prompt, images=[], generation_config=cfg, streamer=streamer)
    else:
        pipe.generate(prompt, cfg, streamer)
    t_end = time.perf_counter()

    rss_after = _measure_rss()
    max_rss = max(rss_before, rss_after)

    total_time = t_end - t_start
    first_latency = (first_token_time - t_start) * 1000 if first_token_time else 0
    second_latency = (second_token_time - first_token_time) * 1000 if second_token_time else 0

    # 实际 token 数：用 tokenizer 编码输出文本
    full_text = "".join(all_text)
    tok_out = pipe.get_tokenizer().encode(full_text)
    actual_tokens = tok_out.input_ids.shape[-1]

    # tok/s = 实际 token / (总时间 - 首 token 延迟)
    gen_time = t_end - (first_token_time or t_end)
    second_tps = actual_tokens / gen_time if gen_time > 0 else 0

    return {
        "first_latency": first_latency,
        "second_latency": second_latency,
        "max_rss": max_rss,
        "second_tps": second_tps,
        "total_tokens": actual_tokens,
        "total_time": total_time,
    }


def _run_optimum_bench(model, processor, input_size):
    """Optimum 格式基准测试。"""
    from transformers import TextIteratorStreamer
    from threading import Thread
    import torch

    prompt = _make_prompt(input_size)
    test_prompt = prompt + "请详细解释这段话的含义。"

    # 预热
    msgs = [{"role": "user", "content": [{"type": "text", "text": "你好"}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt")
    model.generate(**inputs, max_new_tokens=10)

    rss_before = _measure_rss()

    # 正式测试
    msgs = [{"role": "user", "content": [{"type": "text", "text": test_prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt")

    first_token_time = None
    second_token_time = None
    all_text = []

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=128, do_sample=False, streamer=streamer)

    t_start = time.perf_counter()
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for t in streamer:
        now = time.perf_counter()
        if t:
            all_text.append(t)
            if first_token_time is None:
                first_token_time = now
            elif second_token_time is None:
                second_token_time = now
    thread.join()
    t_end = time.perf_counter()

    rss_after = _measure_rss()
    max_rss = max(rss_before, rss_after)

    total_time = t_end - t_start
    first_latency = (first_token_time - t_start) * 1000 if first_token_time else 0
    second_latency = (second_token_time - first_token_time) * 1000 if second_token_time else 0

    # 实际 token 数
    full_text = "".join(all_text)
    actual_tokens = len(processor.tokenizer.encode(full_text))

    gen_time = t_end - (first_token_time or t_end)
    second_tps = actual_tokens / gen_time if gen_time > 0 else 0

    return {
        "first_latency": first_latency,
        "second_latency": second_latency,
        "max_rss": max_rss,
        "second_tps": second_tps,
        "total_tokens": actual_tokens,
        "total_time": total_time,
    }


def run_benchmark(model_path):
    """运行基准测试。"""
    from .chat import load_model

    if not os.path.isdir(model_path):
        print(f"错误: 找不到模型目录: {model_path}")
        sys.exit(1)

    ctx = load_model(model_path)
    device = ctx["device"]

    print(f"\n{'='*60}")
    print(f"  ov-cli benchmark")
    print(f"  模型: {model_path}")
    print(f"  设备: {device}")
    if ctx.get("model_type"):
        print(f"  架构: {ctx['model_type']}")
    print(f"{'='*60}\n")

    # 预热 (3 轮, 确保 GPU 升频 + KV Cache 就绪)
    import openvino_genai as ov_genai
    import time as _time
    print(f"  {'预热中 (3 轮)...':40s}")
    for _ in range(3):
        if ctx.get("optimum"):
            msgs = [{"role": "user", "content": [{"type": "text", "text": "你好"}]}]
            text = ctx["processor"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = ctx["processor"](text=[text], return_tensors="pt")
            ctx["model"].generate(**inputs, max_new_tokens=10)
        elif ctx.get("is_vlm"):
            ctx["pipe"].generate("你好", images=[], max_new_tokens=10)
        else:
            ctx["pipe"].generate("你好", max_new_tokens=10)
    print(f"  {'休息 3s...':40s}", end=" ", flush=True)
    _time.sleep(3)
    print()

    input_sizes = [32, 1024]
    results = {}

    for size in input_sizes:
        print(f"  输入大小: {size} tokens")
        print(f"  {'-'*40}")

        if ctx.get("optimum"):
            res = _run_optimum_bench(ctx["model"], ctx["processor"], size)
        else:
            pipe = ctx["pipe"]
            res = _run_genai_bench_detailed(pipe, _make_prompt(size) + "请详细解释这段话的含义。")

        results[size] = res
        print(f"    1st latency:    {res['first_latency']:>8.1f} ms")
        print(f"    2nd latency:    {res['second_latency']:>8.1f} ms")
        print(f"    2nd token/s:    {res['second_tps']:>8.1f}")
        print(f"    max RSS:        {res['max_rss']:>8} MB")
        print(f"    total tokens:   {res['total_tokens']:>8}")
        print(f"    total time:     {res['total_time']:>8.3f}s")
        print()

    # 汇总表
    print(f"{'='*60}")
    print(f"  汇总")
    print(f"{'='*60}")
    print(f"  {'Input':>8} | {'1st lat':>8} | {'2nd lat':>8} | {'tok/s':>8} | {'RSS':>8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for size in input_sizes:
        r = results[size]
        print(f"  {size:>8} | {r['first_latency']:>7.0f}ms | {r['second_latency']:>7.0f}ms | {r['second_tps']:>7.1f} | {r['max_rss']:>7}MB")
    print()
