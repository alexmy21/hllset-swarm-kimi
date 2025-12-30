# main.py
from pathlib import Path
from hllset_swarm import SwarmHRT, ingest_stream, HASH_FUNC
import torch, json, sys

def demo():
    hrt = SwarmHRT(max_edges=20_000)
    corpus = ["人工智能引领未来", "机器学习改变世界", "深度学习驱动创新"]
    commits = []
    belief = torch.zeros(len(hrt.h2i), device="cuda")
    for belief_vec, commit in ingest_stream(corpus, hrt, swarm_iters_per_chunk=3):
        commits.append(commit)
    # ---- guided finish ----
    dest_hashes = [HASH_FUNC(c) for c in ["未", "来", "世", "界"]]
    p_star = torch.zeros(len(hrt.h2i), device="cuda")
    for h in dest_hashes:
        if h in hrt.h2i:
            p_star[hrt.h2i[h]] = 1.0
    p_star = p_star / p_star.sum()
    for step in range(5):
        belief = hrt.guided_step(belief, p_star)
        if torch.norm(belief - p_star, 1) < 1e-3:
            break
    top = torch.topk(belief, 10).indices
    tokens = [hrt.i2h[i.item()] for i in top]
    print("Destination tokens:", "".join(tokens))
    Path("kimi_git_log.json").write_text(json.dumps(commits, indent=2))

if __name__ == "__main__":
    demo()