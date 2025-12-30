# src/hllset_swarm/hll.py
from julia import Main, Julia
# Julia.install()

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

# Auto-detect HllSets.jl path if not set
hllsets_path = os.getenv("HLLSETS_PATH")

if not hllsets_path:
    # Try to find HllSets.jl relative to this file
    current_dir = Path(__file__).parent
    hllsets_jl = current_dir / "HllSets.jl"
    
    if hllsets_jl.exists():
        hllsets_path = str(hllsets_jl)
    else:
        raise EnvironmentError(
            f"HLLSETS_PATH environment variable is not set and HllSets.jl not found at {hllsets_jl}"
        )

# Load the HllSets.jl file
Main.include(hllsets_path)
Main.using(".HllSets")

from .constants import P_BITS
from .constants import SHARED_SEED, HASH_FUNC

class HLL:
    def __init__(self):
        self.P = P_BITS
        self.hll = Main.HllSet(P_BITS)
    def add(self, token: str):
        add_func = getattr(Main, "add!")
        return add_func(self.hll, token, seed=SHARED_SEED)
        # Main.add!(self.jl, token, seed=SHARED_SEED)
    def count(self) -> float: return float(Main.count(self.hll))
    def intersect(self, other: "HLL") -> "HLL":
        return HLL.from_julia(Main.intersect(self.hll, other.hll))
    def union(self, other: "HLL") -> "HLL":
        return HLL.from_julia(Main.union(self.hll, other.hll))
    def diff(self, other: "HLL") -> "HLL":
        return HLL.from_julia(Main.diff(self.hll, other.hll)[0])  # deleted part
    @staticmethod
    def from_julia(hll): h = HLL(); h.hll = hll; return h