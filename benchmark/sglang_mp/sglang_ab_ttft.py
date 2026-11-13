#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SGLang A/B 性能对比脚本：纯 SGLang vs SGLang + LMCache-Ascend MP。

同一份脚本跑在两种服务配置下（服务怎么切换见
docs/sglang_mp_mode/ab_benchmark_guide.md），各跑三个阶段：

  阶段1 REFRESH（核心证据）
    N 次「flush_cache + 同一长前缀请求」。
    纯 SGLang：每次都全量 prefill，N 个 TTFT 应基本持平。
    LMCache  ：第 1 次冷（Stored），第 2..N 次热（Retrieved），TTFT 应大幅下降。
    ← 两种模式的差值就是 LMCache 跨请求持久缓存的收益。

  阶段2 RADIX（健全性检查）
    N 次同一请求，不 flush。
    两种模式都应很快（SGLang 本地 RadixCache 命中）——
    若 LMCache 模式在这里反而慢，说明接线有问题，先排查再谈性能。

  阶段3 LOAD（可选，--load-total > 0 时执行）
    唯一前缀并发请求（ThreadPoolExecutor），测吞吐与 TTFT 分布。
    两种模式的原始吞吐预期接近；LMCache 的收益在 TTFT 不在吞吐。

用法：
  # A 组（纯 SGLang 服务上）：
  python sglang_ab_ttft.py --mode pure
  # B 组（带 --enable-lmcache 的服务上）：
  python sglang_ab_ttft.py --mode lmcache
  # 带并发负载与自定义参数：
  python sglang_ab_ttft.py --mode lmcache --repeat 8 --load-total 32 --load-conc 8

只依赖标准库，无需 pip 安装任何东西。
结果同时打印到终端并保存 JSON（--out，默认 /workspace/results/）。
"""

# Standard
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

DEFAULT_PREFIX_UNIT = "以下是关于分布式KV缓存系统的一次内部技术评审记录，涉及内存池分层与传输通道选型。"
DEFAULT_QUESTION = "请用一句话总结上面的内容。"


def http_json(url, payload=None, method=None, timeout=600):
    """发 JSON 请求，返回解析后的 dict；空响应体返回 {}。"""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def wait_ready(base, timeout=900):
    """轮询 /get_model_info 直到服务就绪（A/B 切换重启后使用）。"""
    t0 = time.perf_counter()
    while True:
        try:
            http_json(base + "/get_model_info", timeout=10)
            print(f"服务就绪（等待 {time.perf_counter() - t0:.1f}s）")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError(f"服务 {timeout}s 内未就绪")
            time.sleep(2)


def flush_cache(base):
    """清空 SGLang 本地 RadixCache。阶段1 的每次采样前必调。"""
    urllib.request.urlopen(
        urllib.request.Request(base + "/flush_cache", method="POST"),
        timeout=60,
    ).read()


def one_request(base, model, prompt, max_tokens, stream=True):
    """发一次 chat 请求，返回 (ttft, total) 秒。stream=False 时 ttft 为 None。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft, total = None, None
    with urllib.request.urlopen(req, timeout=600) as r:
        if stream:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0  # 首个非空 token
                    # 流读完拿完整耗时
            total = time.perf_counter() - t0
        else:
            json.loads(r.read())
            total = time.perf_counter() - t0
    return ttft, total


def stats_of(samples):
    """对 TTFT 列表算统计量。"""
    vals = [v for v in samples if v is not None]
    if not vals:
        return {}
    ordered = sorted(vals)
    return {
        "n": len(vals),
        "min": round(ordered[0], 3),
        "median": round(statistics.median(vals), 3),
        "mean": round(statistics.mean(vals), 3),
        "p95": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 3),
        "max": round(ordered[-1], 3),
    }


def fmt_row(tag, ttft, total):
    return f"  {tag:<28} TTFT={ttft if ttft is None else f'{ttft:.3f}s':>8}  total={total:.3f}s"


def phase_refresh(base, model, prompt, repeat):
    """阶段1：每次 flush 后发同一请求 —— A/B 的核心对比数据。"""
    print(f"\n[阶段1 REFRESH] flush_cache + 同前缀请求 × {repeat}")
    ttfts, totals = [], []
    for i in range(repeat):
        flush_cache(base)
        time.sleep(1)  # 给 flush 一点落地时间
        ttft, total = one_request(base, model, prompt)
        ttfts.append(ttft)
        totals.append(total)
        note = "（冷：本地缓存已清）" if i == 0 else "（本地缓存已清，只剩引擎/LMCache 侧缓存）"
        print(fmt_row(f"sample #{i + 1}{note}", ttft, total))
    return {"ttft_samples": [round(t, 3) if t else None for t in ttfts],
            "total_samples": [round(t, 3) for t in totals],
            "stats": stats_of(ttfts)}


def phase_radix(base, model, prompt, repeat):
    """阶段2：不 flush 连发 —— 验证本地 RadixCache 在两种模式下都正常。"""
    print(f"\n[阶段2 RADIX] 不 flush 连发 × {repeat}（预期两种模式都快）")
    ttfts, totals = [], []
    for i in range(repeat):
        ttft, total = one_request(base, model, prompt)
        ttfts.append(ttft)
        totals.append(total)
        print(fmt_row(f"sample #{i + 1}", ttft, total))
    return {"ttft_samples": [round(t, 3) if t else None for t in ttfts],
            "total_samples": [round(t, 3) for t in totals],
            "stats": stats_of(ttfts)}


def phase_load(base, model, prefix_unit, prefix_units, question, total_reqs, conc):
    """阶段3：唯一前缀并发请求（非流式），测吞吐与延迟分布。"""
    print(f"\n[阶段3 LOAD] 唯一前缀 × {total_reqs} 请求，并发 {conc}")
    latencies = []

    def work(i):
        # 每个请求前缀唯一：拼上编号，防止跨请求命中
        prompt = f"{prefix_unit * prefix_units}\n[请求编号 {i}]\n{question}"
        _, total = one_request(base, model, prompt, max_tokens=32, stream=False)
        return total

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        latencies = list(pool.map(work, range(total_reqs)))
    wall = time.perf_counter() - t0
    result = {
        "wall_s": round(wall, 3),
        "req_per_s": round(total_reqs / wall, 3),
        "latency_stats": stats_of(latencies),
        "latency_samples": [round(t, 3) for t in latencies],
    }
    print(f"  吞吐 {result['req_per_s']} req/s，墙钟 {result['wall_s']}s，"
          f"延迟 median={result['latency_stats']['median']}s")
    return result


def main():
    ap = argparse.ArgumentParser(description="SGLang A/B TTFT benchmark")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8998)
    ap.add_argument("--model", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--mode", required=True, choices=["pure", "lmcache"],
                    help="当前服务配置：pure=纯 SGLang，lmcache=带 --enable-lmcache")
    ap.add_argument("--repeat", type=int, default=5, help="阶段1/2 各自的采样次数")
    ap.add_argument("--prefix-units", type=int, default=80,
                    help="长前缀由多少个默认单元拼成（80 单元约 4800 字 ≈ 数千 token，> chunk 256）")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--load-total", type=int, default=0, help=">0 时启用阶段3，请求总数")
    ap.add_argument("--load-conc", type=int, default=8, help="阶段3 并发数")
    ap.add_argument("--skip-wait", action="store_true", help="跳过就绪等待（服务已 ready 时）")
    ap.add_argument("--out-dir", default="/workspace/results")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    if not args.skip_wait:
        wait_ready(base)

    prompt = f"{DEFAULT_PREFIX_UNIT * args.prefix_units}\n\n{args.question}"
    print(f"前缀长度 {len(prompt)} chars，模式 {args.mode}")

    report = {
        "mode": args.mode,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "prompt_chars": len(prompt),
        "repeat": args.repeat,
    }
    report["phase1_refresh"] = phase_refresh(base, args.model, prompt, args.repeat)
    report["phase2_radix"] = phase_radix(base, args.model, prompt, args.repeat)
    if args.load_total > 0:
        report["phase3_load"] = phase_load(
            base, args.model, DEFAULT_PREFIX_UNIT, args.prefix_units,
            DEFAULT_QUESTION, args.load_total, args.load_conc)

    out_path = f"{args.out_dir}/ab_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    s1 = report["phase1_refresh"]["stats"]
    print(f"\n===== 结果摘要（{args.mode}）=====")
    print(f"阶段1 REFRESH TTFT: median={s1.get('median')}s mean={s1.get('mean')}s "
          f"(min={s1.get('min')} max={s1.get('max')})")
    s2 = report["phase2_radix"]["stats"]
    print(f"阶段2 RADIX   TTFT: median={s2.get('median')}s（健全性：两种模式都应很快）")
    print(f"完整数据已保存: {out_path}")


if __name__ == "__main__":
    main()
