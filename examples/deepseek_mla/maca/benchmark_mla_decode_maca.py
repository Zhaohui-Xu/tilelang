"""Check C500 MLA against an FP32 reference and compare complete calls."""

import argparse
import functools
import gc
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tempfile

try:
    import torch
except ImportError:
    torch = None


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


def compare_three_way(calls, batches, warmup, iterations):
    """Balance implementation order, keeping allocations and merge in each call."""
    orders = list(itertools.permutations(calls))
    results = {}
    for clock in ("event", "wall"):
        samples = [{name: time_call(calls[name], warmup, iterations, clock) for name in orders[i % len(orders)]} for i in range(batches)]
        results[clock] = {
            "samples_us": samples,
            "median_us": {name: statistics.median(row[name] for row in samples) for name in calls},
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


def make_report_inputs(batch, heads, length, seed, profile):
    """Match the report's GPU FP16 generator, including its stress inputs."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shapes = ((batch, heads, 512), (batch, heads, 64), (batch, length, 1, 512), (batch, length, 1, 64))
    inputs = tuple(torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator) for shape in shapes)
    if profile != "random":
        q, q_pe, kv, _ = inputs
        q.zero_()
        q_pe.zero_()
        if profile == "large-v":
            kv[..., ::2] = 2048
            kv[..., 1::2] = -2048
        elif profile in ("early-peak", "late-peak"):
            q[..., 0] = 32
            kv[..., 0] = 0
            kv[:, 0 if profile == "early-peak" else -1, :, 0] = 64
        else:
            raise ValueError(f"unknown report profile {profile!r}")
    return inputs


def run_report(args, tilelang, triton, build_candidate, schedule, triton_mla):
    """Replay saved configurations with fresh inputs; do not rerun tuning."""
    from plot_mla_decode_maca import summarize_results, write_csv

    recipe = json.loads(Path(__file__).with_name("benchmark_results.json").read_text())
    baseline_hash = hashlib.sha256(args.baseline_file.read_bytes()).hexdigest()
    if baseline_hash != recipe["sources"]["dense_baseline_sha256"]:
        raise ValueError("report mode requires the pinned dense baseline source linked in README")
    keys = list(recipe["results"]) if args.report_shapes is None else args.report_shapes
    environment = {
        "device": torch.cuda.get_device_name(),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__,
        "tilelang": tilelang.__version__,
        "triton": triton.__version__,
    }
    for label, command in (
        ("mx_smi", ["mx-smi"]),
        ("scheduler", ["mx-smi", "sgpu", "--show-sched-class"]),
        ("timeslice", ["mx-smi", "sgpu", "--show-timeslice"]),
    ):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            environment[label] = {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            environment[label] = {"unavailable": str(error)}
    sources = {"dense_baseline_sha256": baseline_hash}
    for filename in ("example_mla_decode_maca.py", "benchmark_mla_decode_maca.py", "benchmark_mla_decode_maca_triton.py"):
        sources[filename] = hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
    document = {
        "environment": environment,
        "sources": sources,
        "protocol": recipe["protocol"],
        "configuration_source_sha256": hashlib.sha256(Path(__file__).with_name("benchmark_results.json").read_bytes()).hexdigest(),
        "configuration_selection": "Replay published configurations; no tuning performed in this run",
        "requested_shapes": keys,
        "report_complete": False,
        "results": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, prefix=args.output.name, delete=False) as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
            temporary = Path(stream.name)
        temporary.replace(args.output)

    build_baseline = load_baseline(args.baseline_file)
    save()

    def run_shape(key):
        batch, heads, length = map(int, key.split("x"))
        if torch.cuda.mem_get_info()[0] < 4 * 1024**3:
            raise RuntimeError("report mode requires at least 4 GiB of free GPU memory before each shape")
        config = recipe["results"][key]
        if list(schedule(batch, heads, length)) != config["default_config"]:
            raise ValueError(f"{key}: the current default schedule differs from the recorded report")
        row = {name: config[name] for name in ("default_config", "dense_config", "triton_config", "measurement")}
        row.update(validation={}, complete=False)
        document["results"][key] = row
        compiled = build_candidate(batch, length, heads)

        def candidate(*inputs):
            output = torch.empty_like(inputs[0])
            compiled(*inputs, output)
            return output

        tile_h, tile_n, splits = row["dense_config"]
        dense = build_baseline(batch, heads, 1, length, 512, 64, tile_n, tile_h, splits, 1 / 24.0)
        tile_h, tile_n, warps, splits = row["triton_config"]
        functions = {
            "default": candidate,
            "dense": dense,
            "triton": functools.partial(triton_mla, head_tile=tile_h, key_tile=tile_n, warps=warps, splits=splits),
        }
        protocol = recipe["protocol"]
        cases = [("random", seed) for seed in protocol["validation_random_seeds"]]
        cases += [(profile, protocol["seed"]) for profile in protocol["validation_stress_profiles"]]
        for profile, seed in cases:
            inputs = make_report_inputs(batch, heads, length, seed, profile)
            expected = reference(*inputs)
            errors = {}
            for name, function in functions.items():
                output = function(*inputs)
                if output.dtype != torch.float16 or output.shape != expected.shape:
                    raise AssertionError(f"{key}/{name}: unexpected output shape or dtype")
                torch.testing.assert_close(output.float(), expected, atol=protocol["atol"], rtol=protocol["rtol"])
                errors[name] = (output.float() - expected).abs().max().item()
            row["validation"][f"{profile}/{seed}"] = errors
            save()
            del inputs, expected, output
        measure = row["measurement"]
        inputs = make_report_inputs(batch, heads, length, measure["seed"], "random")
        calls = {name: functools.partial(function, *inputs) for name, function in functions.items()}
        orders = list(itertools.permutations(calls))
        if measure["batches"] != len(orders):
            raise ValueError("the recorded three-way report requires all six implementation orders")
        for clock in ("wall", "event"):
            row[clock] = []
            for order in orders:
                row[clock].append({name: time_call(calls[name], measure["warmup"], measure["iterations"], clock) for name in order})
                save()
        row["complete"] = True
        save()
        print(
            f"PASS {key}: "
            + ", ".join(f"{name}={statistics.median(s[name] for s in row['wall']):.2f} us" for name in ("triton", "dense", "default")),
            flush=True,
        )

    for key in keys:
        print(f"Report shape {key}", flush=True)
        try:
            run_shape(key)
        except Exception as error:
            document["failure"] = {"shape": key, "error": str(error)}
            save()
            raise
        gc.collect()
        torch.cuda.empty_cache()
    records = summarize_results(document, keys)
    write_csv(records, args.output.with_suffix(".csv"))
    document["report_complete"] = set(keys) == set(recipe["results"])
    save()
    print(f"Saved {len(keys)} fresh shapes to {args.output} and {args.output.with_suffix('.csv')}", flush=True)


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("smoke", "compare", "report"),
        help="smoke: partial-tile correctness; compare: B1/H16/2K; report: replay the recorded three-way matrix",
    )
    parser.add_argument("--output", type=Path, help="New report JSON path (default: mla-report.json); existing files are not overwritten")
    parser.add_argument("--report-shapes", nargs="+", metavar="BxHxN", help="Validate a subset of report shapes, e.g. 1x16x2048")
    parser.add_argument("--baseline-file", type=Path, help="TileLang-MetaX's example_mla_decode.py")
    parser.add_argument(
        "--triton-config",
        type=int,
        nargs=4,
        metavar=("HEAD_TILE", "KEY_TILE", "WARPS", "SPLITS"),
        help="Add the independent Triton dense baseline with a screened configuration",
    )
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
    args = parser.parse_args(argv)
    if not argv:
        args.preset = "smoke"
    if args.preset:
        fixed = {
            "--kv-length",
            "--heads",
            "--batch-size",
            "--head-tile",
            "--splits",
            "--baseline-splits",
            "--triton-config",
            "--tune",
            "--tune-splits",
        }
        if args.preset == "report":
            fixed.update({"--seeds", "--batches", "--warmup", "--iterations", "--check-only"})
        if any(arg.split("=", 1)[0] in fixed for arg in argv):
            parser.error("presets fix the shape and schedule; omit --preset to customize them")
        args.batch_size, args.heads = 1, 16
        if args.preset == "smoke":
            if args.baseline_file:
                parser.error("the smoke preset checks only the C500 kernel; use --preset compare for baselines")
            args.kv_length, args.check_only = 2049, True
        else:
            args.kv_length = 2048
            saved = Path(__file__).with_name("benchmark_results.json")
            if not saved.is_file():
                parser.error(f"the compare preset requires {saved.name} alongside this script")
            config = json.loads(saved.read_text())["results"]["1x16x2048"]
            args.baseline_splits = config["dense_config"][2]
            args.triton_config = config["triton_config"]
            if args.baseline_file is None:
                root = os.environ.get("TILELANG_METAX_ROOT")
                if root:
                    args.baseline_file = Path(root) / "examples/deepseek_mla/example_mla_decode.py"
                else:
                    parser.error("set TILELANG_METAX_ROOT to your baseline checkout, or pass --baseline-file")
    if args.preset != "report" and (args.output is not None or args.report_shapes is not None):
        parser.error("--output and --report-shapes require --preset report")
    if args.preset == "report":
        args.output = args.output or Path("mla-report.json")
        if args.output.exists() or args.output.with_suffix(".csv").exists():
            parser.error("report output already exists; choose a new --output path")
        if args.output.suffix.lower() != ".json":
            parser.error("--output must have a .json suffix")
        if args.report_shapes is not None:
            keys = json.loads(Path(__file__).with_name("benchmark_results.json").read_text())["results"]
            if (
                not args.report_shapes
                or len(set(args.report_shapes)) != len(args.report_shapes)
                or any(k not in keys for k in args.report_shapes)
            ):
                parser.error("--report-shapes must be distinct keys from benchmark_results.json")
    if min(args.seeds, args.batches, args.iterations) < 1 or args.warmup < 0:
        parser.error("seeds, batches and iterations must be positive and warmup must be nonnegative")
    if args.heads < 16 or args.heads % 16:
        parser.error("heads must be a positive multiple of 16")
    if args.batch_size < 1 or not 1 <= args.kv_length <= 65536:
        parser.error("batch size must be positive and KV length must be in [1, 65536]")
    if args.baseline_file and (args.baseline_splits < 1 or args.kv_length % (32 * args.baseline_splits)):
        parser.error("the dense baseline requires whole 32-token tiles in each split")
    if args.baseline_file is None and not args.check_only:
        parser.error("timing requires --baseline-file")
    if args.baseline_file is not None and not args.baseline_file.is_file():
        parser.error("the TileLang-MetaX baseline file does not exist")
    if args.triton_config:
        h, n, w, s = args.triton_config
        if args.baseline_file is None:
            parser.error("three-way timing requires --baseline-file")
        if h not in (16, 32, 64) or n not in (16, 32, 64) or w not in (1, 2, 4, 8) or s < 1:
            parser.error("invalid Triton head tile, key tile, wave count or split count")
        if args.heads % h or args.kv_length % (n * s):
            parser.error("Triton tiles must divide the head count and per-split KV length")
    return parser, args


def main():
    parser, args = parse_args()
    if torch is None:
        parser.error("activate an environment with the MetaX PyTorch build; --help does not require GPU packages")
    if not torch.cuda.is_available() or "c500" not in torch.cuda.get_device_name().lower():
        parser.error("this benchmark requires a visible MetaX C500 and PyTorch-MACA")

    try:
        import tilelang
        from example_mla_decode_maca import default_schedule, mla_decode_maca
    except ImportError as error:
        parser.error(f"this example requires the TileLang-MetaX MACA backend; check the active environment and PYTHONPATH: {error}")

    if args.triton_config:
        try:
            import triton
            from benchmark_mla_decode_maca_triton import run as triton_mla
        except ImportError as error:
            parser.error(f"the three-way comparison requires MetaX Triton in this environment: {error}")

    torch.set_float32_matmul_precision("highest")
    if args.preset == "report":
        run_report(args, tilelang, triton, mla_decode_maca, default_schedule, triton_mla)
        return
    shape = (args.batch_size, args.heads, 1, args.kv_length, 512, 64)
    softmax_scale = (512 + 64) ** -0.5
    print(f"C500 MLA: batch={args.batch_size}, heads={args.heads}, KV length={args.kv_length}", file=sys.stderr, flush=True)
    print("Building candidate (the first call includes compilation)", file=sys.stderr, flush=True)
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
        configurations = [("baseline_dense", args.baseline_splits)] if args.triton_config else [("baseline_no_split", 1)]
        if args.baseline_splits != 1 and not args.triton_config:
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
    if args.triton_config:
        h, n, w, s = args.triton_config
        kernels["triton_dense"] = functools.partial(triton_mla, head_tile=h, key_tile=n, warps=w, splits=s)
        errors["triton_dense"] = check_kernel(
            "triton_dense", kernels["triton_dense"], args.batch_size, args.kv_length, cases, device, args.check_only, args.heads
        )
        result.update(triton=triton.__version__, triton_config=args.triton_config)
    result["correctness"] = {"passed_per_kernel": len(cases), "max_abs_error": errors}
    if not args.check_only:
        inputs = make_inputs(args.batch_size, args.kv_length, 23, "random", device, args.heads)
        result["timing_config"] = {"seed": 23, "warmup": args.warmup, "iterations": args.iterations, "batches": args.batches}
        calls = {name: functools.partial(fn, *inputs) for name, fn in kernels.items()}
        measure = compare_three_way if args.triton_config else compare
        result["timings"] = measure(calls, args.batches, args.warmup, args.iterations)
    passed = result["correctness"]["passed_per_kernel"] * len(errors)
    print(f"PASS: {passed} full-output checks; max absolute error {max(errors.values()):.6g}", file=sys.stderr)
    if args.triton_config and not args.check_only:
        for name in ("triton_dense", "baseline_dense", "candidate"):
            latency = result["timings"]["wall"]["median_us"][name]
            print(f"{name}: {latency:.2f} us (median full-call wall)", file=sys.stderr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
