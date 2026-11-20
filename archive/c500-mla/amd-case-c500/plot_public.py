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

fig, panels = plt.subplots(1,2,figsize=(10,4.2),sharey=True,layout="constrained")
lengths = [1,2,4,8,16]
for ax,batch in zip(panels,(64,128)):
    base(ax)
    for name,label,color,mark in [('dense','TileLang-MetaX dense','#536b9a','s'),
                                  ('default','C500 dense (default)','#00746e','o'),
                                  ('paged_native','FlashMLA (prepacked)','#b55c14','^')]:
        vals=[2*batch*128*k*1024*(576+512)/(wall(batch,128,k*1024,name)*1e6) for k in lengths]
        ax.plot(range(5),vals,label=label,marker=mark,color=color,lw=2,ms=6)
    ax.set(title=f"Batch {batch}",xticks=range(5),xticklabels=[f"{k}K" for k in lengths],xlabel="KV length (tokens)",ylim=(0,None))
panels[0].set_ylabel("Effective throughput (TFLOP/s)")
handles, labels = panels[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="outside upper center",ncol=3,frameon=False)
save(fig,"flashmla-maca")
