import openvino_genai as ov_genai
import time

MODEL = "/mnt/orico-remote/ov-cli/model/lfm/LFM2-24B-A2B-int4-ov"

for device in ["GPU.0", "GPU.1"]:
    print(f"\n--- Loading on {device} ---")
    pipe = ov_genai.LLMPipeline(MODEL, device)
    pipe.generate("Hello", ov_genai.GenerationConfig(max_new_tokens=10))

    ts = []
    texts = []
    def streamer(t):
        ts.append(time.perf_counter())
        texts.append(t)

    t0 = time.perf_counter()
    pipe.generate("Hello,how are you", ov_genai.GenerationConfig(max_new_tokens=128), streamer)
    t1 = time.perf_counter()

    tok = pipe.get_tokenizer()
    full = "".join(texts)
    actual = tok.encode(full).input_ids.shape[-1]  # type: ignore

    first = ts[0]
    second = ts[1] if len(ts) > 1 else first
    last = ts[-1]

    ttft = (first - t0) * 1000
    tpot = (second - first) * 1000
    gen_time = last - first
    tok_s = actual / gen_time if gen_time > 0 else 0
    print(f"{device}:  TTFT={ttft:.0f}ms  TPOT={tpot:.0f}ms  decode={tok_s:.1f} tok/s  ({actual} tokens)")
