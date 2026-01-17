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
        self.h2i = {}
        self.i2h = {}
        self.h2token = {}  # NEW: hash -> original token string
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
        val = torch.tensor(self.vals, dtype=torch.float32, device=device)
        adj = torch.sparse_coo_tensor(idx, val, size=(len(self.h2i), len(self.h2i)), device=device).coalesce()
        return adj

    # ---- swarm belief update ----
    def swarm_step(self, p: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
        # Ensure p is float32 for sparse operations
        original_dtype = p.dtype
        if p.dtype != torch.float32:
            p = p.float()
    
        # Ensure p is 2D for sparse matrix multiplication
        if p.dim() == 1:
            p = p.unsqueeze(1)  # Convert to column vector
        
        p_new = torch.sparse.mm(self.csr(), p)

        # Convert back to 1D if needed
        if p_new.dim() == 2 and p_new.size(1) == 1:
            p_new = p_new.squeeze(1)
            
        p_new = torch.clamp(p_new, 0, 1)

        # Convert back to original dtype if needed
        if original_dtype != torch.float32:
            p_new = p_new.to(original_dtype)

        # mask by row-HLL non-zero
        mask = torch.tensor([float(self.row_hll[i].count() > 0) for i in range(len(self.h2i))], device=p.device)
        p_new *= mask
        return p_new / (p_new.sum() + 1e-8)

    def guided_step(self, p: torch.Tensor, p_star: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        # Ensure p is float32 for sparse operations
        original_dtype = p.dtype
        if p.dtype != torch.float32:
            p = p.float()
            p_star = p_star.float()
        
        # Ensure p is 2D for sparse matrix multiplication
        squeeze_output = False
        if p.dim() == 1:
            p = p.unsqueeze(1)  # Convert to column vector
            p_star = p_star.unsqueeze(1)
            squeeze_output = True
        
        p_inert = torch.sparse.mm(self.csr(), p)
        
        # Convert back to 1D for gradient computation if needed
        if squeeze_output:
            p_inert_1d = p_inert.squeeze(1)
            p_1d = p.squeeze(1)
            p_star_1d = p_star.squeeze(1)
            grad = 2 * alpha * (p_1d - p_star_1d)
            p_guided = torch.clamp(p_inert_1d - 0.1 * grad, 0, 1)
        else:
            grad = 2 * alpha * (p - p_star)
            p_guided = torch.clamp(p_inert - 0.1 * grad, 0, 1)
        
        # sparse top-k projection
        k = int(1.5 * p_star.count_nonzero())

        k = min(k, p_guided.numel())  # Clamp k to valid range
        if k == 0:
            k = 1  # Ensure at least 1 element is selected

        top_val, top_idx = torch.topk(p_guided.flatten(), k)
        p_out = torch.zeros_like(p_guided.flatten())
        p_out[top_idx] = top_val
        
        # Convert back to original dtype if needed
        if original_dtype != torch.float32:
            p_out = p_out.to(original_dtype)
        
        return p_out / (p_out.sum() + 1e-8)
    
    def particles_from_cell(self, u: int, v: int) -> Tuple[HLL, HLL, HLL]:
        """return (D, R, N) for edge u→v"""
        row_hll = self.row_hll[u]
        col_hll = self.col_hll[v]
        inter = row_hll.intersect(col_hll)   # retained
        deleted = row_hll.diff(col_hll)      # deleted
        new = col_hll.diff(row_hll)          # new
        return deleted, inter, new           # the 3 particles