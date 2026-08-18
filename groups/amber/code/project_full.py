#!/usr/bin/env python3
"""Full-dimension (384-d, no PCA) t-SNE and UMAP of all sentence embeddings."""
import os, sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(BASE, "work")
emb = np.load(os.path.join(W, "embeddings.npy")).astype(np.float32)
which = sys.argv[1]

if which == "tsne":
    from openTSNE import TSNE
    ts = TSNE(perplexity=30, metric="cosine", n_jobs=-1, random_state=42,
              verbose=True).fit(emb)
    np.save(os.path.join(W, "tsne_full.npy"), np.asarray(ts, dtype=np.float32))
    print("saved work/tsne_full.npy")
elif which == "umap":
    import umap
    um = umap.UMAP(n_neighbors=30, min_dist=0.08, metric="cosine",
                   random_state=42, verbose=True).fit_transform(emb)
    np.save(os.path.join(W, "umap_full.npy"), um.astype(np.float32))
    print("saved work/umap_full.npy")
