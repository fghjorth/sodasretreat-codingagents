#!/usr/bin/env python3
"""Embed all SOTU sentences. Runs identically on GPU host or local.

Input : work/sentences/<year>.txt  (lines: "<id>\t<sentence>")
Output: work/embeddings.npy  (float32, N x D)
        work/embedding_index.csv (row,year,id)
"""
import os, sys, glob, csv
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))
SENT_DIR = os.path.join(BASE, "work", "sentences")

rows, texts = [], []
for path in sorted(glob.glob(os.path.join(SENT_DIR, "*.txt"))):
    year = int(os.path.basename(path)[:4])
    with open(path, encoding="utf-8") as f:
        for line in f:
            i, _, s = line.rstrip("\n").partition("\t")
            rows.append((year, int(i)))
            texts.append(s)

print(f"{len(texts)} sentences from {len(set(y for y,_ in rows))} years", flush=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(texts, batch_size=256, show_progress_bar=True,
                   convert_to_numpy=True, normalize_embeddings=True)
np.save(os.path.join(BASE, "work", "embeddings.npy"), emb.astype(np.float32))
with open(os.path.join(BASE, "work", "embedding_index.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["row", "year", "id"])
    for r, (year, i) in enumerate(rows):
        w.writerow([r, year, i])
print("saved", emb.shape, flush=True)
