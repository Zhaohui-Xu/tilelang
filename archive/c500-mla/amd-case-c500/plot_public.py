"""Render the README figures from the six raw wall samples per implementation."""
import json
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
figures = Path("contributions/mla-c500-upstream/tilelang/examples/deepseek_mla/figures")
data = json.loads((root / "defaults.json").read_text())
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#253041", "axes.edgecolor": "#8a97a6", "pdf.fonttype": 42})
def wall(batch, heads, length, name):
    samples = data[f"{batch}x{heads}x{length}"]["wall"]
    assert len(samples) == 6
    return median(s[name] for s in samples)
def base(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e0e6ec", linewidth=.8)
    ax.tick_params(length=0, pad=6)
def save(fig, name):
    fig.savefig(figures / (name + ".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(root / (name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

fig, ax = plt.subplots(figsize=(8.2, 3.6), layout="constrained")
base(ax)
lengths = [2, 4, 8, 16, 32, 64]
for batch, offset, color, marker in [(1,-.16,"#00746e","o"),(2,0,"#b55c14","s"),(4,.16,"#5269a6","D")]:
    for i,k in enumerate(lengths):
        if f"{batch}x16x{k*1024}" not in data: continue
        v = wall(batch,16,k*1024,"dense") / wall(batch,16,k*1024,"default")
        ax.scatter(i+offset,v,marker=marker,s=56,color=color,label=f"Batch {batch}" if i==0 else None)
ax.axhline(1,color="#657181",linestyle="--",linewidth=1)
ax.set(xticks=range(6),xticklabels=[f"{k}K" for k in lengths],xlabel="KV length (tokens)",
       ylabel="Speedup over tuned dense baseline",ylim=(.9,3.05),yticks=[1,1.5,2,2.5,3])
ax.legend(frameon=False,loc="upper left",ncol=3)
save(fig,"flashmla-maca-small")

fig, panels = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
fig.subplots_adjust(top=.76, bottom=.14, left=.07, right=.99, wspace=.09)
fig.suptitle("Dense MLA decode on MetaX C500", fontsize=16, y=.98)
lengths = [1, 2, 4, 8, 16]
series = [
    ("dense", "TileLang-MetaX dense", "#627da9", "//"),
    ("default", "C500 dense (default)", "#00746e", ""),
    ("paged_native", "FlashMLA (prepacked)", "#ba681e", ".."),
]
width = .27
for ax, batch in zip(panels, (64, 128)):
    base(ax)
    for index, (name, label, color, hatch) in enumerate(series):
        values = [2 * batch * 128 * k * 1024 * (576 + 512) /
                  (wall(batch, 128, k * 1024, name) * 1e6) for k in lengths]
        bars = ax.bar([i + (index - 1) * width for i in range(5)], values,
                      width=width, label=label, color=color, hatch=hatch,
                      edgecolor="white", linewidth=.55)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set(title=f"Batch size = {batch}", xticks=range(5),
           xticklabels=[f"{k}K" for k in lengths], xlabel="KV context length (tokens)",
           ylim=(0, 76))
panels[0].set_ylabel("Effective throughput (TFLOP/s)")
handles, labels = panels[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.93), ncol=3, frameon=False)
save(fig, "flashmla-maca")
