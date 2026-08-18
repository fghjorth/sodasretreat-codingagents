#!/usr/bin/env python3
"""KMeans clusters on sentence embeddings + c-TF-IDF top terms and samples per cluster."""
import os, csv, json, re, glob
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(BASE, "work")
emb = np.load(os.path.join(W, "embeddings.npy"))

K = 14
km = KMeans(n_clusters=K, n_init=4, random_state=42).fit(emb)
np.save(os.path.join(W, "clusters.npy"), km.labels_.astype(np.int16))

# load sentence texts in index order
idx = list(csv.DictReader(open(os.path.join(W, "embedding_index.csv"))))
texts_by_year = {}
for p in glob.glob(os.path.join(W, "sentences", "*.txt")):
    yr = int(os.path.basename(p)[:4])
    texts_by_year[yr] = {}
    for line in open(p, encoding="utf-8"):
        i, _, s = line.rstrip("\n").partition("\t")
        texts_by_year[yr][int(i)] = s
texts = [texts_by_year[int(r["year"])][int(r["id"])] for r in idx]

STOP = set("""the of and to in a for our that we is are will it this with as be have has on by
i my you your they their them us all more not no or an at from was were can must new great
american america americans people nation nations national congress year years than but if so
should would may do does did been who what which there its it's let now us""".split())

def tokens(s):
    return [w for w in re.findall(r"[a-z][a-z'-]+", s.lower()) if w not in STOP and len(w) > 2]

# c-TF-IDF: term freq within cluster * log(K / cluster-presence)
cluster_tf, presence = [], Counter()
for k in range(K):
    c = Counter()
    for t, lab in zip(texts, km.labels_):
        if lab == k:
            c.update(set(tokens(t)))
    cluster_tf.append(c)
    for w in c: presence[w] += 1

out = {}
rng = np.random.default_rng(42)
for k in range(K):
    n = int((km.labels_ == k).sum())
    scores = {w: f * np.log(K / presence[w]) for w, f in cluster_tf[k].items() if f >= 5}
    top = [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:12]]
    rows = np.where(km.labels_ == k)[0]
    samp = [texts[i][:160] for i in rng.choice(rows, size=min(6, len(rows)), replace=False)]
    out[str(k)] = {"n": n, "top_terms": top, "samples": samp}

json.dump(out, open(os.path.join(W, "cluster_info.json"), "w"), indent=1)
for k, v in out.items():
    print(k, v["n"], ", ".join(v["top_terms"][:8]))
