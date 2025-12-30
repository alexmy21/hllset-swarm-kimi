# src/hllset_swarm/swarm_hrt.py
from typing import Dict, Tuple
import torch
from .constants import HASH_FUNC, P_BITS
from .hll import HLL

class SwarmHRT:
    def __init__(self, max_edges: int = 100_000):
        self.max_edges = max_edges
        # sparse tensor
        self.rows, self.cols, self.vals = [], [], []
        self.h2i: Dict[int, int] = {}
        self.i2h: Dict[int, int] = {}
        # HLLSets attached to every row/col
        self.row_hll: Dict[int, HLL] = {}
        self.col_hll: Dict[int, HLL] = {}

    def add_edge(self, h1: int, h2: int, v: float = 1.0):
        if len(self.rows) >= self.max_edges: return
        for h in (h1, h2):
            if h not in self.h2i:
                idx = len(self.h2i)
                self.h2i[h] = idx
                self.i2h[idx] = h
                self.row_hll[idx] = HLL()
                self.col_hll[idx] = HLL()
        self.rows.append(self.h2i[h1])
        self.cols.append(self.h2i[h2])
        self.vals.append(v)

    def csr(self, device="cuda"):
        idx = torch.tensor([self.rows, self.cols], dtype=torch.long, device=device)
        val = torch.tensor(self.vals, dtype=torch.float16, device=device)
        adj = torch.sparse_coo_tensor(idx, val, size=(len(self.h2i), len(self.h2i)), device=device).coalesce()
        return adj

    # ---- swarm belief update ----
    def swarm_step(self, p: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
        p_new = torch.sparse.mm(self.csr(), p)
        p_new = torch.clamp(p_new, 0, 1)
        # mask by row-HLL non-zero
        mask = torch.tensor([float(self.row_hll[i].count() > 0) for i in range(len(self.h2i))], device=p.device)
        p_new *= mask
        return p_new / (p_new.sum() + 1e-8)

    def guided_step(self, p: torch.Tensor, p_star: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        p_inert = torch.sparse.mm(self.csr(), p)
        grad = 2 * alpha * (p - p_star)
        p_guided = torch.clamp(p_inert - 0.1 * grad, 0, 1)
        # sparse top-k projection
        k = int(1.5 * p_star.count_nonzero())
        top_val, top_idx = torch.topk(p_guided, k)
        p_out = torch.zeros_like(p_guided)
        p_out[top_idx] = top_val
        return p_out / (p_out.sum() + 1e-8)
    
    def particles_from_cell(self, u: int, v: int) -> Tuple[HLL, HLL, HLL]:
        """return (D, R, N) for edge u→v"""
        row_hll = self.row_hll[u]
        col_hll = self.col_hll[v]
        inter = row_hll.intersect(col_hll)   # retained
        deleted = row_hll.diff(col_hll)      # deleted
        new = col_hll.diff(row_hll)          # new
        return deleted, inter, new           # the 3 particles