#!/usr/bin/env python3
"""t-SNE map of all sentence embeddings, saved for plotting."""
import os, csv, numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

BASE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(BASE, "work")
emb = np.load(os.path.join(W, "embeddings.npy"))
print("PCA 384 -> 50", flush=True)
X = PCA(n_components=50, random_state=42).fit_transform(emb)
print("t-SNE on", X.shape, flush=True)
ts = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
          random_state=42, verbose=1).fit_transform(X)
np.save(os.path.join(W, "tsne.npy"), ts.astype(np.float32))
print("saved work/tsne.npy", flush=True)
