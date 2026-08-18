#!/usr/bin/env python3
"""Final figure: presidential attention to energy in SOTU addresses, 1946-2020."""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

BASE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(BASE, "work")

# palette (reference instance, light mode)
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; AXIS = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # embedding, dictionary, LLM labels

series = list(csv.DictReader(open(os.path.join(W, "embedding_series.csv"))))
years = [int(r["year"]) for r in series]
est = [float(r["estimate"]) for r in series]
dic = {int(r["year"]): float(r["estimate"]) for r in csv.DictReader(open(os.path.join(W, "dictionary_results.csv")))}
labels = json.load(open(os.path.join(W, "llm_labels.json")))
ntot = {}
for r in csv.DictReader(open(os.path.join(W, "embedding_index.csv"))):
    ntot[int(r["year"])] = ntot.get(int(r["year"]), 0) + 1
llm = {int(y): len(ids) / ntot[int(y)] for y, ids in labels.items()}

scores = list(csv.DictReader(open(os.path.join(W, "embedding_scores.csv"))))
sy = np.array([int(r["year"]) for r in scores])
sp = np.array([float(r["score"]) for r in scores])
TAU = 0.2022

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True,
                               gridspec_kw={"height_ratios": [3, 2], "hspace": 0.14})
fig.patch.set_facecolor(SURFACE)

# ---- top: annual series ----
ax1.set_facecolor(SURFACE)
ax1.plot(sorted(dic), [dic[y] for y in sorted(dic)], lw=1.4, color=S2, alpha=0.9, zorder=2)
ly = sorted(llm)
ax1.plot(ly, [llm[y] for y in ly], lw=1.4, color=S3, alpha=0.9, zorder=2)
ax1.plot(years, est, lw=2.2, color=S1, zorder=3,
         marker="o", ms=3.5, markerfacecolor=S1, markeredgecolor=SURFACE, markeredgewidth=0.6)

# direct labels for the three series (right edge, staggered) + legend
ax1.legend(["Dictionary (era-aware keywords)", "LLM sentence labels", "Embedding axis (primary)"],
           loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)

events = [(1974.5, 0.245, "1973–74 OPEC embargo\n(1975 peak: 23% of sentences)", "center"),
          (1980, 0.138, "1979–80 second oil shock,\nCarter Doctrine", "left"),
          (2006, 0.062, "“addicted to oil”", "right"),
          (2012.6, 0.098, "all-of-the-above /\nclean-energy push", "left"),
          (1949, 0.058, "postwar fuels &\natomic power", "left")]
for x, y, txt, ha in events:
    ax1.annotate(txt, (x, y), fontsize=8.2, color=INK2, ha=ha, va="bottom", linespacing=1.25,
                 bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1.5))

ax1.set_ylabel("Share of sentences about energy", fontsize=10)
ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax1.set_ylim(0, 0.27)
ax1.grid(axis="y", color=GRID, lw=0.7)
ax1.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax1.spines[s].set_visible(False)
ax1.tick_params(length=0)
ax1.set_title("Presidential attention to energy, State of the Union addresses 1946–2020",
              fontsize=13.5, loc="left", pad=26, fontweight="bold")
ax1.text(0, 1.045, "Three independent measures of the same construct; the embedding measure projects each of 20,813 sentences "
                   "on a learned energy axis (TCAV-style)", transform=ax1.transAxes, fontsize=9.5, color=INK2, va="top")

# ---- bottom: sentence projections (mechanism) ----
ax2.set_facecolor(SURFACE)
rng = np.random.default_rng(7)
jit = rng.uniform(-0.33, 0.33, size=len(sp))
above = sp > TAU
ax2.scatter(sy[~above] + jit[~above], sp[~above], s=1.6, c=MUTED, alpha=0.10, lw=0, rasterized=True)
ax2.scatter(sy[above] + jit[above], sp[above], s=6, c=S1, alpha=0.75, lw=0)
ax2.axhline(TAU, color=INK2, ls=(0, (4, 3)), lw=1)
ax2.text(1945.2, TAU + 0.012, "τ = 0.20 — sentences above count as energy", fontsize=8.5, color=INK2,
         bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9, pad=1.5))
ax2.set_ylabel("Projection on energy axis", fontsize=10)
ax2.set_xlabel("Year", fontsize=10)
ax2.grid(axis="y", color=GRID, lw=0.7)
ax2.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax2.spines[s].set_visible(False)
ax2.tick_params(length=0)
ax2.set_xlim(1944.5, 2021.5)
ax2.set_xticks(range(1950, 2021, 10))

fig.text(0.01, 0.008, "Corpus: 75 addresses (corpus.csv, SODAS retreat exercise) · embeddings: all-MiniLM-L6-v2 · "
                      "axis: logistic-regression CAV trained on agent labels (held-out acc 0.92) · group amber",
         fontsize=8, color=MUTED)
fig.savefig(os.path.join(W, "energy_attention_figure.png"), dpi=200,
            bbox_inches="tight", facecolor=SURFACE)
print("wrote work/energy_attention_figure.png")
