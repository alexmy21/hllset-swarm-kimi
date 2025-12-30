# src/hllset_swarm/ingest.py
import torch
from .swarm_hrt import SwarmHRT
from .constants import HASH_FUNC

def ingest_stream(corpus_iter, hrt: SwarmHRT, swarm_iters_per_chunk: int = 3):
    """generator yielding (belief_vector, commit_dict) for every chunk"""
    p = torch.zeros(len(hrt.h2i), device="cuda")
    parent = "000000"
    for chunk_id, text in enumerate(corpus_iter):
        # ---- 3-gram window ----
        for i in range(len(text) - 2):
            w = text[i:i+3]
            h1, h2, h3 = HASH_FUNC(w[0]), HASH_FUNC(w[:2]), HASH_FUNC(w)
            hrt.add_edge(h1, h2)
            hrt.add_edge(h2, h3)
            hrt.add_edge(h3, h1)
            # burn tokens into HLLSets
            hrt.row_hll[hrt.h2i[h1]].add(w[0])
            hrt.row_hll[hrt.h2i[h2]].add(w[:2])
            hrt.row_hll[hrt.h2i[h3]].add(w)
        # ---- swarm burn ----
        for step in range(swarm_iters_per_chunk):
            p = hrt.swarm_step(p)
            sha = HASH_FUNC(f"chunk{chunk_id}step{step}") & 0x7FFFFFFF
            meta = {"chunk": chunk_id, "step": step, "mass": float(p.sum())}
            yield p, {"sha": hex(sha)[2:9], "parent": parent, "meta": meta}
            parent = hex(sha)[2:9]