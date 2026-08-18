#!/usr/bin/env python3
"""Figure: t-SNE map of all 20,813 SOTU sentence embeddings."""
import os, csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

BASE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(BASE, "work")
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; AXIS = "#c3c2b7"
S1 = "#2a78d6"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

ts = np.load(os.path.join(W, "tsne.npy"))
idx = list(csv.DictReader(open(os.path.join(W, "embedding_index.csv"))))
years = np.array([int(r["year"]) for r in idx])
scores = {(int(r["year"]), int(r["id"])): float(r["score"])
          for r in csv.DictReader(open(os.path.join(W, "embedding_scores.csv")))}
proj = np.array([scores[(int(r["year"]), int(r["id"]))] for r in idx])
TAU = 0.2022
en = proj > TAU

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
                     "text.color": INK, "axes.edgecolor": AXIS, "axes.titlecolor": INK})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 7.2))
fig.patch.set_facecolor(SURFACE)

for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_aspect("equal")

# left: energy vs rest
ax1.scatter(ts[~en, 0], ts[~en, 1], s=2.2, c=MUTED, alpha=0.14, lw=0, rasterized=True)
ax1.scatter(ts[en, 0], ts[en, 1], s=7, c=S1, alpha=0.8, lw=0)
ax1.set_title("Energy sentences (blue) in the embedding space", fontsize=11.5, loc="left", color=INK2)

# right: energy sentences colored by year
cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
norm = Normalize(vmin=1946, vmax=2020)
ax2.scatter(ts[~en, 0], ts[~en, 1], s=2.2, c="#e1e0d9", alpha=0.25, lw=0, rasterized=True)
sc = ax2.scatter(ts[en, 0], ts[en, 1], s=8, c=years[en], cmap=cmap, norm=norm, alpha=0.9, lw=0)
ax2.set_title("Energy sentences colored by year — vocabulary drift", fontsize=11.5, loc="left", color=INK2)
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax2, fraction=0.035, pad=0.02)
cb.outline.set_visible(False)
cb.ax.tick_params(color=MUTED, labelcolor=MUTED, labelsize=8.5)

fig.suptitle("t-SNE map of 20,813 State of the Union sentences, 1946–2020",
             fontsize=14, fontweight="bold", x=0.01, y=0.985, ha="left")
fig.text(0.01, 0.925, "MiniLM sentence embeddings (PCA 50, then t-SNE, perplexity 30); "
                      "energy = projection on learned energy axis above τ = 0.20",
         fontsize=9.5, color=INK2)
fig.text(0.01, 0.01, "Group amber · SODAS retreat exercise", fontsize=8, color=MUTED)
fig.tight_layout(rect=[0, 0.02, 1, 0.88])
fig.savefig(os.path.join(W, "energy_tsne.png"), dpi=200, bbox_inches="tight", facecolor=SURFACE)
print("wrote work/energy_tsne.png")
