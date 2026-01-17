# ------------------------------------------------------------
# HLLSet-Swarm Demo with HLLSets.jl Wrapper (RTX 3060-ready)
# ------------------------------------------------------------
import torch
import torch.sparse as tsp
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Import your HLL wrapper
# ------------------------------------------------------------
# Adjust this import path to where your hll.py lives
from .hll import HLL, AddResult  # <--- Use your wrapper
from .constants import P_BITS, SHARED_SEED

# ------------------------------------------------------------
# 2. Dense Cortex Vector (fast union/forecast)
# ------------------------------------------------------------
class CortexVector:
    """Dense bit-vector for macro-state; separate from probabilistic HLL."""
    def __init__(self, m: int, device: str = 'cpu'):
        self.m = m
        self.device = device
        self.bits = torch.zeros(m, dtype=torch.uint8, device=device)
    
    def union_with_tokens(self, tokens: List[str], seed: int = SHARED_SEED):
        """Add tokens by hashing to bit positions (deterministic for demo)."""
        for token in tokens:
            h = hash(f"{seed}:{token}") & (self.m - 1)
            self.bits[h] = 1
    
    def union_with_hll(self, hll: HLL):
        """Convert HLL counts to bit mask (threshold > 0)."""
        counts = torch.tensor(hll.dump(), device=self.device, dtype=torch.uint8)
        self.bits = torch.bitwise_or(self.bits, counts.clamp(max=1))
    
    def count(self) -> int:
        return self.bits.sum().item()
    
    def to_float(self) -> torch.Tensor:
        return self.bits.float()

# ------------------------------------------------------------
# 3. AM Builder using proper HLL probabilistic counts
# ------------------------------------------------------------
# def build_am(corpus: List[str], m: int, ngram: int = 2, device: str = 'cpu') -> tsp.FloatTensor:
#     """
#     Build sparse adjacency matrix using HLLSets.jl for true probabilistic counting.
#     Each transition source -> dest is stored as an HLL to enable BSS computations.
#     """
#     # HLL for each source token (or use Radix/HashRelationTensor for production)
#     src_hlls: Dict[int, HLL] = {}
#     trans_counts = defaultdict(int)
    
#     for sent in corpus:
#         chars = ["<START>"] + list(sent) + ["<END>"]
#         for i in range(len(chars) - 1):
#             src_hash = hash(chars[i]) & (m - 1)
#             dst_token = chars[i + 1]
            
#             if src_hash not in src_hlls:
#                 src_hlls[src_hash] = HLL(P_BITS)  # Proper Julia HLL
            
#             # Add destination token to source HLL (probabilistic counting)
#             src_hlls[src_hash].add(dst_token)
#             trans_counts[(src_hash, dst_token)] += 1
    
#     # Build sparse matrix from transition frequencies (stochastic)
#     if not trans_counts:
#         idx = torch.empty((2, 0), dtype=torch.long, device=device)
#         val = torch.empty((0,), dtype=torch.float32, device=device)
#     else:
#         idx = torch.tensor(list(trans_counts.keys()), dtype=torch.long, device=device).t()
#         val = torch.tensor(list(trans_counts.values()), dtype=torch.float32, device=device)
#         # Normalize rows
#         row_sum = torch.zeros(m, dtype=torch.float32, device=device)
#         row_sum.scatter_add_(0, idx[0], val).clamp_(min=1.0)
#         val /= row_sum[idx[0]]
    
#     return tsp.FloatTensor(idx, val, torch.Size([m, m]))
def build_am(corpus: List[str], m: int, ngram: int = 2, device: str = 'cpu') -> tsp.FloatTensor:
    """
    Build sparse adjacency matrix using HLLSets.jl for true probabilistic counting.
    Each transition source -> dest is stored as an HLL to enable BSS computations.
    
    Args:
        corpus: List of text strings (sentences)
        m: Size of hash space (must be power of 2)
        ngram: N-gram size (currently only 2 supported)
        device: 'cpu' or 'cuda'
    
    Returns:
        Sparse adjacency matrix [m x m] with normalized transition probabilities
    """
    # HLL for each source token
    src_hlls: Dict[int, HLL] = {}
    trans_counts = defaultdict(int)
    
    for sent in corpus:
        chars = ["<START>"] + list(sent) + ["<END>"]
        for i in range(len(chars) - 1):
            src_token = chars[i]
            dst_token = chars[i + 1]
            
            # Hash both tokens to integer indices
            src_hash = hash(src_token) & (m - 1)
            dst_hash = hash(dst_token) & (m - 1)
            
            # Initialize HLL for source if needed
            if src_hash not in src_hlls:
                src_hlls[src_hash] = HLL(P_BITS)
            
            # Add destination token to source HLL (for BSS computations)
            src_hlls[src_hash].add(dst_token)
            
            # Count transition frequency
            trans_counts[(src_hash, dst_hash)] += 1
    
    # Build sparse COO tensor
    if not trans_counts:
        # Empty matrix
        idx = torch.empty((2, 0), dtype=torch.long, device=device)
        val = torch.empty((0,), dtype=torch.float32, device=device)
    else:
        # Extract edges and values
        edges = list(trans_counts.keys())
        values = list(trans_counts.values())
        
        # Create indices tensor [2, num_edges]: row 0 = sources, row 1 = destinations
        idx = torch.tensor(edges, dtype=torch.long, device=device).t()
        val = torch.tensor(values, dtype=torch.float32, device=device)
        
        # Normalize each row to make stochastic (rows sum to 1)
        row_sum = torch.zeros(m, dtype=torch.float32, device=device)
        row_sum.scatter_add_(0, idx[0], val)
        row_sum.clamp_(min=1.0)  # Prevent division by zero
        val = val / row_sum[idx[0]]
    
    # Create sparse tensor: AM[i,j] = P(token_j | token_i)
    return tsp.FloatTensor(idx, val, torch.Size([m, m]))

# ------------------------------------------------------------
# 4. PSM State (with metadata)
# ------------------------------------------------------------
@dataclass(frozen=True)
class PSMState:
    cort: CortexVector
    am: tsp.FloatTensor
    knobs: Dict[str, float]
    swarm: Optional[List[CortexVector]] = None
    metadata: Optional[Dict[str, Any]] = None

# ------------------------------------------------------------
# 5. Pipeline Steps (refactored for HLL consistency)
# ------------------------------------------------------------
class IngestStep:
    """Update AM with new text using HLL for counting."""
    def __call__(self, state: PSMState, corpus: List[str], delta: str) -> PSMState:
        # Rebuild AM incrementally (simplified: rebuild from full corpus + delta)
        new_am = build_am(corpus + [delta], state.cort.m, device=state.am.device())
        return PSMState(cort=state.cort, am=new_am, knobs=state.knobs, 
                        swarm=state.swarm, metadata=state.metadata)

class DecayStep:
    """Apply exponential forgetting to Cortex bits."""
    def __call__(self, state: PSMState) -> PSMState:
        lam = state.knobs['lambda_forget']
        mask = torch.rand(state.cort.m, device=state.cort.device) > lam
        new_cort = CortexVector(state.cort.m, state.cort.device)
        new_cort.bits = torch.bitwise_and(state.cort.bits, mask.byte())
        return PSMState(cort=new_cort, am=state.am, knobs=state.knobs,
                        swarm=state.swarm, metadata=state.metadata)

class PredictStep:
    """Probabilistic forecast: r · AM (with float32 matmul)."""
    def __call__(self, state: PSMState) -> PSMState:
        r = state.cort.to_float().to(torch.float32)
        am_fp32 = state.am.to(torch.float32)
        
        p = tsp.mm(am_fp32, r.unsqueeze(1)).squeeze()
        p = torch.clamp(p, min=0)
        norm = p.sum()
        if norm > 0:
            p /= norm
        p = p.to(torch.float16)
        
        # Convert predictions to HLL for consistency, then back to bits
        k = int(state.knobs['theta'] * state.cort.m)
        top_idx = torch.topk(p, k).indices
        
        # Build HLL from predictions (demonstrates wrapper usage)
        pred_hll = HLL(P_BITS)
        for idx in top_idx:
            pred_hll.add(f"pred_{idx}")  # Tokenize prediction indices
        
        # Merge back to Cortex bits
        new_cort = CortexVector(state.cort.m, state.cort.device)
        new_cort.bits = state.cort.bits.clone()
        new_cort.union_with_hll(pred_hll)
        
        meta = {'N_bits': new_cort.count() - state.cort.count()}
        return PSMState(cort=new_cort, am=state.am, knobs=state.knobs,
                        swarm=state.swarm, metadata=meta)

class NoetherCheckStep:
    """Enforce |N| - |D| = 0 (adjust τ/ρ if violated)."""
    def __call__(self, state: PSMState) -> PSMState:
        D_bits = int(state.cort.count() * state.knobs['lambda_forget'])
        N_bits = state.metadata.get('N_bits', 0) if state.metadata else 0
        phi = N_bits - D_bits
        
        knobs = dict(state.knobs)
        if abs(phi) > knobs['phi_band']:
            knobs['tau'] = max(0.1, knobs['tau'] - 0.01 * phi / state.cort.m)
            knobs['rho'] = min(0.9, knobs['rho'] + 0.01 * phi / state.cort.m)
        
        return PSMState(cort=state.cort, am=state.am, knobs=knobs,
                        swarm=state.swarm, metadata=dict(state.metadata or {}, phi=phi))

class UpdateStep:
    """Finalize Cortex(t+1) = R ⊕ N (already done in PredictStep here)."""
    def __call__(self, state: PSMState) -> PSMState:
        # In this design PredictStep already updated cort; just pass through
        return state

class ReclusterStep:
    """Extract canonical cover: convert Cortex bits to swarm particles."""
    def __call__(self, state: PSMState) -> PSMState:
        particles = []
        idx = state.cort.bits.nonzero().squeeze()
        if idx.numel() == 0:
            return state
        
        # Each non-zero bit becomes a particle with its own HLL
        for i in idx:
            p = CortexVector(state.cort.m, state.cort.device)
            p.bits[i] = 1
            # Attach a miniature HLL for that token (for BSS computations)
            token_hll = HLL(P_BITS)
            token_hll.add(f"token_{i}")
            particles.append((p, token_hll))
        
        # Unpack for state
        part_vectors, part_hlls = zip(*particles)
        return PSMState(cort=state.cort, am=state.am, knobs=state.knobs,
                        swarm=list(part_vectors), metadata=state.metadata)

