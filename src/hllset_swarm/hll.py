# src/hllset_swarm/hll.py
from julia import Main, Julia
# Julia.install()

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
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

@dataclass
class AddResult:
    """Result from adding a token to HLL"""
    token: str
    hash_value: int
    register: int
    leading_zeros: int
    
class HLL:
    def __init__(self, P_BITS: int = P_BITS):
        self.P = P_BITS
        self.hll = Main.HllSet(P_BITS)
        
    # def add(self, token: str, seed: int = SHARED_SEED):
    #     add_func = getattr(Main, "add!")
    #     return add_func(self.hll, token, seed=seed)
    #     # Main.add!(self.jl, token, seed=SHARED_SEED)

    def add(self, token: Union[str, List[str]], seed: int = SHARED_SEED) -> Union[AddResult, List[AddResult], None]:
        """
        Add token(s) to the HLL set
        
        Args:
            token: Single token (str) or list of tokens
            seed: Hash seed value
            
        Returns:
            - AddResult: if single token
            - List[AddResult]: if list of tokens
            - None: if token is empty
        """
        add_func = getattr(Main, "add!")
        
        # Handle single token
        if isinstance(token, str):
            result = add_func(self.hll, token, seed=seed)
            if result is None:
                return None
            return self._parse_add_result(result)
        
        # Handle list of tokens
        elif isinstance(token, (list, tuple)):
            results = add_func(self.hll, token, seed=seed)
            if results is None:
                return None
            return [self._parse_add_result(r) for r in results]
        
        else:
            raise TypeError(f"Token must be str or list, got {type(token)}")
    
    def _parse_add_result(self, result: tuple) -> AddResult:
        """
        Parse the tuple returned by Julia add! function
        
        Args:
            result: Tuple (token, hash, register, leading_zeros)
            
        Returns:
            AddResult dataclass
        """
        if result is None or len(result) != 4:
            raise ValueError(f"Invalid add! result: {result}")
        
        token, hash_value, register, leading_zeros = result
        
        return AddResult(
            token=str(token),
            hash_value=int(hash_value),
            register=int(register),
            leading_zeros=int(leading_zeros)
        )

    def count(self) -> float: return float(Main.count(self.hll))

    # def dump(self) -> bytes: return bytes(Main.dump(self.hll))

    def dump(self) -> list:
        """
        Get the counts vector from the HLL set
        
        Returns:
            List of UInt32 values representing the HLL counts
        """
        # Access the counts field directly from Julia object
        counts_vector = self.hll.counts
        # Convert Julia vector to Python list
        return list(counts_vector)
    
    def dump_numpy(self):
        """
        Get the counts vector as a numpy array
        
        Returns:
            numpy array of the HLL counts
        """
        import numpy as np
        counts_vector = self.hll.counts
        # Convert to numpy array
        return np.array(list(counts_vector), dtype=np.uint32)

    def isempty(self) -> float: return float(Main.isempty(self.hll))

    def isequal(self, other: "HLL") -> float: return float(Main.isequal(self.hll, other.hll))

    def intersect(self, other: "HLL") -> "HLL":
        return HLL.from_julia(Main.intersect(self.hll, other.hll))
    
    def union(self, other: "HLL") -> "HLL":
        return HLL.from_julia(Main.union(self.hll, other.hll))
    
    def diff(self, other: "HLL") -> "HLL":
        return HLL.from_julia(Main.diff(self.hll, other.hll)[0])  
    
    def cosine(self, other: "HLL") -> float:
        return float(Main.cosine(self.hll, other.hll))  
    
    def match(self, other: "HLL") -> float:
        return float(Main.match(self.hll, other.hll))  
    
    @staticmethod
    def from_julia(hll): h = HLL(); h.hll = hll; return h