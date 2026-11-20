"""Check C500 MLA against an FP32 reference and compare complete calls."""

import argparse
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch


def reference(q, q_pe, kv, k_pe):
    """Dense MLA with FP32 scores, softmax, and value accumulation."""
    if q.shape[0] > 4:
        return torch.cat([reference(*parts) for parts in zip(*(x.split(4) for x in (q, q_pe, kv, k_pe)))])
    k, k_rope = kv[:, :, 0, :].float(), k_pe[:, :, 0, :].float()
    scale = (q.shape[-1] + q_pe.shape[-1]) ** -0.5
    logits = torch.bmm(q.float(), k.transpose(1, 2))
    logits += torch.bmm(q_pe.float(), k_rope.transpose(1, 2))
    return torch.bmm(torch.softmax(logits * scale, dim=-1), k)


def make_inputs(batch_size, kv_length, seed, profile, device, heads=16):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shapes = ((batch_size, heads, 512), (batch_size, heads, 64), (batch_size, kv_length, 1, 512), (batch_size, kv_length, 1, 64))
    q, q_pe, kv, k_pe = [torch.randn(shape, generator=generator, dtype=torch.float32) for shape in shapes]
    if profile in ("early-peak", "late-peak"):
        q.zero_()
        q_pe.zero_()
        kv[..., 0] = 0
        q[..., 0] = 32
        kv[:, 0 if profile == "early-peak" else -1, :, 0] = 64
    elif profile == "large-v":
        # Uniform attention makes the unnormalized partial exceed the FP16 range.
        q.zero_()
        q_pe.zero_()
        kv[..., ::2] = 2048
        kv[..., 1::2] = -2048
    elif profile != "random":
        raise ValueError(f"unknown input profile {profile!r}")
    return tuple(x.to(device=device, dtype=torch.float16).contiguous() for x in (q, q_pe, kv, k_pe))


def load_baseline(path):
    """Use the baseline source from an explicitly selected TileLang-MetaX checkout."""
    spec = importlib.util.spec_from_file_location("tilelang_metax_mla_baseline", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load a Python baseline from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.flashattn


def check_kernel(name, kernel, batch_size, kv_length, cases, device, trace, heads=16):
    max_error = 0.0
    for profile, seed in cases:
        inputs = make_inputs(batch_size, kv_length, seed, profile, device, heads)
        assert all(t.is_contiguous() for t in inputs)
        expected = reference(*inputs)
        if trace:
            torch.cuda.synchronize()
            print(f"{name}: calling {profile}, seed {seed}", file=sys.stderr, flush=True)
        output = kernel(*inputs)
        if trace:
            torch.cuda.synchronize()
            print(f"{name}: checking {profile}, seed {seed}", file=sys.stderr, flush=True)
        if output.shape != expected.shape or output.dtype != torch.float16:
            raise AssertionError(f"{name} returned an unexpected shape or dtype")
        try:
            torch.testing.assert_close(output.float(), expected, rtol=2e-3, atol=2e-3)
        except AssertionError as error:
            raise AssertionError(f"{name}, profile={profile}, seed={seed}: {error}") from error
        max_error = max(max_error, (output.float() - expected).abs().max().item())
    if trace:
        print(f"{name}: passed {len(cases)} checks, max absolute error {max_error:g}", file=sys.stderr, flush=True)
    return max_error


def time_call(call, warmup, iterations, clock):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    if clock == "event":
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / iterations

    start = time.perf_counter()
    for _ in range(iterations):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def prepare_native_flashmla(q, q_pe, kv, k_pe):
    """Prepare the native paged ABI outside timing, with identity page order."""
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    b, h, dim = q.shape
    n = kv.shape[1]
    page = 64
    query = torch.cat((q, q_pe), dim=-1).contiguous().unsqueeze(1)
    cache = torch.cat((kv, k_pe), dim=-1).contiguous().view(-1, page, 1, dim + q_pe.shape[-1])
    table = torch.arange(b * (n // page), dtype=torch.int32, device=q.device).view(b, -1)
    lengths = torch.full((b,), n, dtype=torch.int32, device=q.device)
    metadata, splits = get_mla_metadata(lengths, h, 1)

    def call():
        out, _ = flash_mla_with_kvcache(query, cache, table, lengths, dim, metadata, splits, causal=False)
        return out.squeeze(1)

    return call


def compare(calls, batches, warmup, iterations):
    results = {}
    for name in calls:
        if name == "candidate":
            continue
        for clock in ("event", "wall"):
            rows = []
            for batch in range(batches):
                order = (name, "candidate") if batch % 2 == 0 else ("candidate", name)
                rows.append({key: time_call(calls[key], warmup, iterations, clock) for key in order})
            median_baseline = statistics.median(row[name] for row in rows)
            median_candidate = statistics.median(row["candidate"] for row in rows)
            results[f"{name}_{clock}"] = {
                "samples_us": rows,
                "baseline_median_us": median_baseline,
                "candidate_median_us": median_candidate,
                "paired_median_saving_us": statistics.median(row[name] - row["candidate"] for row in rows),
                "candidate_wins": sum(row["candidate"] < row[name] for row in rows),
            }
    return results


def tune_schedule(build, configs, default, batch, length, heads, device):
    samples = {}
    inputs = make_inputs(batch, length, 17, "random", device, heads)
    expected = reference(*inputs)
    calls = {}
    for cfg in configs:
        call = functools.partial(build(*cfg), *inputs)
        torch.testing.assert_close(call().float(), expected, rtol=2e-3, atol=2e-3)
        calls[cfg] = call
        samples[cfg] = []
    for repeat in range(3):
        offset = repeat * len(configs) // 3
        for cfg in configs[offset:] + configs[:offset]:
            samples[cfg].append(time_call(calls[cfg], 3, 12, "event"))
    shortlist = sorted(configs, key=lambda cfg: statistics.median(samples[cfg]))[:3]
    if default in calls and default not in shortlist:
        shortlist.append(default)
    del calls, call, inputs, expected
    inputs = make_inputs(batch, length, 73, "random", device, heads)
    refined = {cfg: [] for cfg in shortlist}
    for order in (shortlist, list(reversed(shortlist)), shortlist[1:] + shortlist[:1]):
        for cfg in order:
            refined[cfg].append(time_call(functools.partial(build(*cfg), *inputs), 50, 300, "event"))
    winner = min(shortlist, key=lambda cfg: statistics.median(refined[cfg]))
    return winner, {f"{w}/{s}": v for (w, s), v in samples.items()}, {f"{w}/{s}": v for (w, s), v in refined.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path, help="TileLang-MetaX's example_mla_decode.py")
    parser.add_argument("--native-flashmla", action="store_true", help="Time the native prepacked paged FlashMLA interface")
    parser.add_argument("--kv-length", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-tile", type=int, choices=(16, 32, 64, 128))
    parser.add_argument("--tune", action="store_true", help="Screen head tiles and split counts for this shape")
    parser.add_argument("--tune-splits", action="store_true", help="Screen candidate splits before the held-out comparison")
    parser.add_argument("--splits", type=int, default=None, choices=tuple(1 << i for i in range(12)))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--baseline-splits", type=int, default=64, help="Split count for the TileLang-MetaX baseline")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--check-only", action="store_true", help="Compile and run correctness checks without timing")
    args = parser.parse_args()
    if min(args.seeds, args.batches, args.iterations) < 1 or args.warmup < 0:
        parser.error("seeds, batches and iterations must be positive and warmup must be nonnegative")
    if args.heads < 16 or args.heads % 16:
        parser.error("heads must be a positive multiple of 16")
    if args.batch_size < 1 or not 1 <= args.kv_length <= 65536:
        parser.error("batch size must be positive and KV length must be in [1, 65536]")
    if args.baseline_file and (args.baseline_splits < 1 or args.kv_length % (32 * args.baseline_splits)):
        parser.error("the dense baseline requires whole 32-token tiles in each split")
    if args.native_flashmla and args.kv_length % 64:
        parser.error("the native comparison uses complete 64-token pages")
    if args.baseline_file is None and not args.native_flashmla and not args.check_only:
        parser.error("timing requires --baseline-file or --native-flashmla")
    if args.baseline_file is not None and not args.baseline_file.is_file():
        parser.error("the TileLang-MetaX baseline file does not exist")
    if not torch.cuda.is_available() or "c500" not in torch.cuda.get_device_name().lower():
        parser.error("this benchmark requires a visible MetaX C500 and PyTorch-MACA")

    import tilelang

    from example_mla_decode_maca import default_schedule, mla_decode_maca

    torch.set_float32_matmul_precision("highest")
    shape = (args.batch_size, args.heads, 1, args.kv_length, 512, 64)
    softmax_scale = (512 + 64) ** -0.5
    print("Building candidate", file=sys.stderr, flush=True)
    device = torch.device("cuda")

    default_heads, default_splits = default_schedule(args.batch_size, args.heads, args.kv_length)
    head_tile = args.head_tile or default_heads
    if args.heads % head_tile:
        parser.error("the selected head tile must divide the number of query heads")

    def build_candidate(tile, splits):
        compiled = mla_decode_maca(args.batch_size, args.kv_length, args.heads, splits, tile)

        def candidate(q, q_pe, kv, k_pe):
            output = torch.empty_like(q)
            compiled(q, q_pe, kv, k_pe, output)
            return output

        return candidate

    selected = args.splits or default_splits
    screening, refinement = {}, {}
    if args.tune or args.tune_splits:
        tiles = (args.kv_length + 31) // 32
        splits = [1 << i for i in range(tiles.bit_length()) if i == 0 or args.batch_size * args.heads * (1 << i) * 514 * 4 <= 3 * 1024**3]
        widths = [head_tile]
        if args.tune and args.head_tile is None:
            widths = [w for w in (16, 32, 64, 128) if args.heads % w == 0]
        configs = [(w, s) for w in widths for s in splits]
        default = (head_tile, selected)
        (head_tile, selected), screening, refinement = tune_schedule(
            build_candidate, configs, default, args.batch_size, args.kv_length, args.heads, device
        )
    candidate = build_candidate(head_tile, selected)
    cases = [("random", seed) for seed in range(args.seeds)] + [("early-peak", 0), ("late-peak", 0), ("large-v", 0)]
    kernels = {"candidate": candidate}
    errors = {
        "candidate": check_kernel("candidate", candidate, args.batch_size, args.kv_length, cases, device, args.check_only, args.heads)
    }
    result = {
        "shape": shape,
        "candidate_splits": selected,
        "candidate_head_tile": head_tile,
        "candidate_schedule_screen_us": screening,
        "candidate_schedule_refinement_us": refinement,
        "dtype": "float16",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "tilelang": tilelang.__version__,
    }
    if args.baseline_file is not None:
        build_baseline = load_baseline(args.baseline_file)
        configurations = [("baseline_no_split", 1)]
        if args.baseline_splits != 1:
            configurations.append(("baseline_split", args.baseline_splits))
        for name, splits in configurations:
            print(f"Building {name}", file=sys.stderr, flush=True)
            kernels[name] = build_baseline(*shape, 32, 16, splits, softmax_scale)
            errors[name] = check_kernel(name, kernels[name], args.batch_size, args.kv_length, cases, device, args.check_only, args.heads)
        result.update(
            baseline_sha256=hashlib.sha256(args.baseline_file.read_bytes()).hexdigest(),
            baseline_block_n=32,
            baseline_block_h=16,
            baseline_splits=args.baseline_splits,
        )
    if args.native_flashmla:

        def check_native(*xs):
            return prepare_native_flashmla(*xs)()

        errors["paged_native"] = check_kernel(
            "paged_native", check_native, args.batch_size, args.kv_length, cases, device, args.check_only, args.heads
        )
    result["correctness"] = {"passed_per_kernel": len(cases), "max_abs_error": errors}
    if not args.check_only:
        inputs = make_inputs(args.batch_size, args.kv_length, 23, "random", device, args.heads)
        result["timing_config"] = {"seed": 23, "warmup": args.warmup, "iterations": args.iterations, "batches": args.batches}
        calls = {name: functools.partial(fn, *inputs) for name, fn in kernels.items()}
        if args.native_flashmla:
            calls["paged_native"] = prepare_native_flashmla(*inputs)
        result["timings"] = compare(calls, args.batches, args.warmup, args.iterations)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
