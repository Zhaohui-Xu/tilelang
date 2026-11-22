"""Validate raw MLA measurements and regenerate both README charts (no GPU needed)."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, stdev


REFERENCE = Path(__file__).with_name("benchmark_results.json")
IMPLEMENTATIONS = ("triton", "dense", "default")


def summarize_results(document, expected_shapes):
    """Refuse missing shapes or incomplete samples instead of filling a chart."""
    rows = document["results"]
    if set(rows) != set(expected_shapes):
        raise ValueError("shape set differs from the requested matrix; partial reports cannot produce the full charts")
    output = []
    for key, row in rows.items():
        if not row.get("complete") or len(row.get("validation", {})) != 5:
            raise ValueError(f"{key}: five completed output checks per implementation are required")
        for errors in row["validation"].values():
            if set(errors) != set(IMPLEMENTATIONS) or any(not math.isfinite(v) or v < 0 for v in errors.values()):
                raise ValueError(f"{key}: missing or invalid validation error record")
        batch, heads, length = map(int, key.split("x"))
        record = {"batch": batch, "heads": heads, "kv_length": length}
        for clock in ("wall", "event"):
            samples = row.get(clock, [])
            if len(samples) != 6 or any(set(s) != set(IMPLEMENTATIONS) for s in samples):
                raise ValueError(f"{key}: six complete three-way {clock} samples are required")
            for name in IMPLEMENTATIONS:
                values = [s[name] for s in samples]
                if any(not math.isfinite(v) or v <= 0 for v in values):
                    raise ValueError(f"{key}/{name}: invalid latency")
                record[f"{name}_{clock}_us"] = median(values)
                record[f"{name}_{clock}_cv"] = stdev(values) / mean(values)
        for name in ("triton", "dense"):
            record[f"speedup_vs_{name}"] = record[f"{name}_wall_us"] / record["default_wall_us"]
        output.append(record)
    return output


def write_csv(records, path):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def render(records, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "axes.labelcolor": "#253041",
            "axes.edgecolor": "#8a97a6",
        }
    )
    series = [
        ("triton", "Triton (saved config)", "#d18c37", ""),
        ("dense", "TileLang-MetaX dense (saved config)", "#627da9", "//"),
        ("default", "C500 TileLang (default)", "#00746e", ""),
    ]
    lookup = {(r["batch"], r["heads"], r["kv_length"]): r for r in records}
    for small in (False, True):
        batches = (1, 2, 4) if small else (64, 128)
        fig, axes = plt.subplots(
            1, len(batches), figsize=(12.4, 4.5), sharey=True, gridspec_kw={"width_ratios": [6, 3, 3] if small else [1, 1]}
        )
        fig.subplots_adjust(left=0.065, right=0.99, bottom=0.17, top=0.76, wspace=0.09)
        values_all = []
        for ax, b in zip(axes, batches):
            h = 16 if small else 128
            ns = sorted(n for bb, hh, n in lookup if bb == b and hh == h)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e0e6ec", linewidth=0.8)
            for i, (impl, label, color, hatch) in enumerate(series):
                values = [
                    lookup[b, h, n]["dense_wall_us"] / lookup[b, h, n][impl + "_wall_us"]
                    if small
                    else 2 * b * h * n * (576 + 512) / (lookup[b, h, n][impl + "_wall_us"] * 1e6)
                    for n in ns
                ]
                values_all.extend(values)
                bars = ax.bar(
                    [j + (i - 1) * 0.25 for j in range(len(ns))],
                    values,
                    width=0.24,
                    label=label,
                    color=color,
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.4,
                )
                if not small or impl != "dense":
                    ax.bar_label(bars, labels=[f"{v:.2f}" if small else f"{v:.1f}" for v in values], padding=3, fontsize=8, rotation=0)
            if small:
                ax.axhline(1, color="#657181", linestyle="--", linewidth=0.8)
            ax.tick_params(length=0, pad=5)
            ax.set(title=f"Batch {b}", xticks=range(len(ns)), xticklabels=[f"{n // 1024}K" for n in ns])
        axes[0].set_ylabel("Speedup over TileLang dense (×)" if small else "Effective throughput (TFLOP/s)")
        axes[0].set_ylim(0, max(values_all) * (1.25 if small else 1.14))
        fig.supxlabel("KV length (tokens)")
        fig.suptitle("Dense MLA decode on C500 · " + ("16 heads" if small else "128 heads"), fontsize=15, y=0.98)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False, fontsize=10)
        stem = "flashmla-maca-small" if small else "flashmla-maca"
        for ext in ("png", "pdf"):
            fig.savefig(output_dir / f"{stem}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REFERENCE, help="Published or freshly measured report JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("mla-report-plots"))
    args = parser.parse_args()
    document = json.loads(args.input.read_text())
    expected = json.loads(REFERENCE.read_text())["results"]
    try:
        records = summarize_results(document, expected)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(records, args.output_dir / "summary.csv")
    summary = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(), "shapes": len(records), "groups": {}}
    for name, heads in (("small", 16), ("large", 128)):
        group = [r for r in records if r["heads"] == heads]
        summary["groups"][name] = {
            baseline: {
                "range": [min(r[f"speedup_vs_{baseline}"] for r in group), max(r[f"speedup_vs_{baseline}"] for r in group)],
                "wins": sum(r[f"speedup_vs_{baseline}"] > 1 for r in group),
                "count": len(group),
            }
            for baseline in ("triton", "dense")
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render(records, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
