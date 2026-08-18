#!/usr/bin/env python3
"""TCAV-style energy axis: train CAV on LLM labels, project, calibrate tau, build series + figure.

Input : work/embeddings.npy, work/embedding_index.csv, work/llm_labels.json,
        work/dictionary_results.csv, work/sentences/<year>.txt
Output: work/embedding_scores.csv (year,id,score,label)
        work/embedding_series.csv (year,estimate,mean_projection)
        work/energy_axis.png
"""
import os, csv, json, random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve

random.seed(42); np.random.seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(BASE, "work")

emb = np.load(os.path.join(W, "embeddings.npy"))
index = list(csv.DictReader(open(os.path.join(W, "embedding_index.csv"))))
labels = json.load(open(os.path.join(W, "llm_labels.json")))
pos_keys = {(int(y), i) for y, ids in labels.items() for i in ids}
labeled_years = {int(y) for y in labels}

keys = [(int(r["year"]), int(r["id"])) for r in index]
y_all = np.array([1 if k in pos_keys else 0 for k in keys])
in_labeled = np.array([k[0] in labeled_years for k in keys])

# CAV training set: all positives + 4x random negatives, from labeled years only
pos_idx = np.where(y_all == 1)[0]
neg_pool = np.where((y_all == 0) & in_labeled)[0]
neg_idx = np.random.choice(neg_pool, size=min(len(neg_pool), 4 * len(pos_idx)), replace=False)
train_rows = np.concatenate([pos_idx, neg_idx])
Xl, yl = emb[train_rows], y_all[train_rows]
Xtr, Xte, ytr, yte = train_test_split(Xl, yl, test_size=0.25, random_state=42, stratify=yl)

clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
clf.fit(Xtr, ytr)
acc = accuracy_score(yte, clf.predict(Xte))
cav = (clf.coef_[0] / np.linalg.norm(clf.coef_[0])).astype(np.float32)
proj = emb @ cav  # scalar projection of every sentence on the energy axis

# tau: best F1 on held-out
te_rows = train_rows[np.isin(np.arange(len(train_rows)), [])]  # placeholder
# recompute held-out projections directly
proj_te = Xte @ cav
prec, rec, thr = precision_recall_curve(yte, proj_te)
f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
tau = float(thr[np.argmax(f1[:-1])])
print(f"CAV held-out accuracy={acc:.3f}  best F1={np.max(f1[:-1]):.3f}  tau={tau:.4f}")

# annual series
years = sorted({k[0] for k in keys})
series = []
for yr in years:
    m = np.array([k[0] == yr for k in keys])
    share = float((proj[m] > tau).mean())
    series.append((yr, share, float(proj[m].mean())))

with open(os.path.join(W, "embedding_scores.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["year", "id", "score", "llm_label"])
    for (yr, i), p, lab in zip(keys, proj, y_all):
        w.writerow([yr, i, f"{p:.5f}", lab])
with open(os.path.join(W, "embedding_series.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["year", "estimate", "mean_projection"])
    for yr, s, mp in series:
        w.writerow([yr, f"{s:.4f}", f"{mp:.5f}"])

# validation: correlation with dictionary series
dic = {int(r["year"]): float(r["estimate"]) for r in csv.DictReader(open(os.path.join(W, "dictionary_results.csv")))}
a = np.array([s for _, s, _ in series]); b = np.array([dic[yr] for yr, _, _ in series])
def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]
print(f"corr(embedding, dictionary): pearson={np.corrcoef(a,b)[0,1]:.3f} spearman={spearman(a,b):.3f}")
llm_share = {int(y): (len(ids) / sum(1 for k in keys if k[0] == int(y))) for y, ids in labels.items()}
c = np.array([llm_share[yr] for yr, _, _ in series if yr in llm_share])
a2 = np.array([s for yr, s, _ in series if yr in llm_share])
print(f"corr(embedding, LLM labels): pearson={np.corrcoef(a2,c)[0,1]:.3f} spearman={spearman(a2,c):.3f}")

# figure: per-year projection distributions + estimates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
rng = np.random.default_rng(42)
for yr in years:
    m = np.array([k[0] == yr for k in keys])
    p = proj[m]
    jitter = rng.uniform(-0.32, 0.32, size=len(p))
    en = p > tau
    ax1.scatter(np.full(en.sum(), yr) + jitter[en], p[en], s=5, alpha=0.7, c="#d95f02", lw=0)
    ax1.scatter(np.full((~en).sum(), yr) + jitter[~en], p[~en], s=2, alpha=0.12, c="#4477aa", lw=0)
ax1.axhline(tau, color="#333", ls="--", lw=1)
ax1.annotate(f"  τ = {tau:.3f}", (years[0], tau), va="bottom", fontsize=9, color="#333")
ax1.set_ylabel("projection on energy axis (CAV)")
ax1.set_title("SOTU sentences on the learned energy axis, 1946–2020  (orange = above τ)")
ax2.plot([s[0] for s in series], [s[1] for s in series], "-o", ms=3, c="#d95f02", label="embedding-axis share")
ax2.plot(sorted(dic), [dic[y] for y in sorted(dic)], "-", lw=1, c="#4477aa", alpha=0.8, label="dictionary share")
ly = sorted(llm_share); ax2.plot(ly, [llm_share[y] for y in ly], ":", lw=1.2, c="#117733", label="LLM-label share")
ax2.set_ylabel("share of sentences"); ax2.set_xlabel("year"); ax2.legend(frameon=False, fontsize=9)
for ax in (ax1, ax2): ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(W, "energy_axis.png"), dpi=150)
print("wrote work/energy_axis.png")
