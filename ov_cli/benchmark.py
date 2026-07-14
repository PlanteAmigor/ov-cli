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


def _run_genai_bench_detailed(pipe, prompt):
    """用 streamer 精确测量各指标。"""
    import openvino_genai as ov_genai

    rss_before = _measure_rss()

    # 统计输入 token 数
    tok = pipe.get_tokenizer()
    try:
        input_enc = tok.encode(prompt)
        input_tokens = input_enc.input_ids.shape[-1]
    except Exception:
        input_tokens = len(prompt.replace("\n", "")) // 2  # 近似

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
    prefill_time = (first_token_time - t_start) if first_token_time else total_time
    first_latency = prefill_time * 1000
    second_latency = (second_token_time - first_token_time) * 1000 if second_token_time else 0
    prefill_tps = input_tokens / prefill_time if prefill_time > 0 else 0

    # 实际 token 数：用 tokenizer 编码输出文本
    full_text = "".join(all_text)
    tok_out = tok.encode(full_text)
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
        "prefill_tps": prefill_tps,
        "input_tokens": input_tokens,
    }


def run_benchmark(model_path, device=""):
    """运行基准测试。"""
    from .chat import load_model

    if not os.path.isdir(model_path):
        print(f"错误: 找不到模型目录: {model_path}")
        sys.exit(1)

    ctx = load_model(model_path, device=device)
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
        if ctx.get("is_vlm"):
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

        if ctx.get("is_vlm"):
            res = _run_genai_bench_detailed(ctx["pipe"], _make_prompt(size) + "请详细解释这段话的含义。")

        results[size] = res
        print(f"    prefill:        {res['prefill_tps']:>8.1f} tok/s ({res['input_tokens']} tok in {res['first_latency']/1000:.2f}s)")
        print(f"    decode:         {res['second_tps']:>8.1f} tok/s")
        print(f"    1st lat (TTFT):{res['first_latency']:>8.1f} ms")
        print(f"    2nd lat:        {res['second_latency']:>8.1f} ms")
        print(f"    total tokens:   {res['total_tokens']:>8}")
        print(f"    total time:     {res['total_time']:>8.3f}s")
        print()

    # 汇总表
    print(f"{'='*60}")
    print(f"  汇总")
    print(f"{'='*60}")
    print(f"  {'Input':>8} | {'TTFT':>8} | {'prefill':>8} | {'2nd lat':>8} | {'decode':>8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for size in input_sizes:
        r = results[size]
        print(f"  {r['input_tokens']:>8} | {r['first_latency']:>7.0f}ms | {r['prefill_tps']:>7.1f} | {r['second_latency']:>7.0f}ms | {r['second_tps']:>7.1f}")
    print()
