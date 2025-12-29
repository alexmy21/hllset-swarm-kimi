"""
HyperLogLog implementation for cardinality estimation.

This is a light-weight implementation without external dependencies.
"""

import hashlib
import math
from typing import Any, List, Union


class HyperLogLog:
    """
    Light-weight HyperLogLog cardinality estimator.
    
    HyperLogLog is a probabilistic data structure for estimating the
    cardinality (number of unique elements) in a dataset.
    
    Args:
        precision: The precision parameter (4-16). Higher values give
                  better accuracy but use more memory. Default is 14.
    """
    
    def __init__(self, precision: int = 14):
        if not 4 <= precision <= 16:
            raise ValueError("Precision must be between 4 and 16")
        
        self.precision = precision
        self.m = 1 << precision  # 2^precision
        self.registers = [0] * self.m
        self.alpha = self._get_alpha()
    
    def _get_alpha(self) -> float:
        """Get the bias correction constant based on m."""
        if self.m >= 128:
            return 0.7213 / (1 + 1.079 / self.m)
        elif self.m >= 64:
            return 0.709
        elif self.m >= 32:
            return 0.697
        else:
            return 0.673
    
    def _hash(self, item: Any) -> int:
        """Hash an item to a 64-bit integer."""
        if not isinstance(item, bytes):
            item = str(item).encode('utf-8')
        return int(hashlib.sha256(item).hexdigest()[:16], 16)
    
    def add(self, item: Any) -> None:
        """
        Add an item to the HyperLogLog.
        
        Args:
            item: The item to add (will be converted to string if needed)
        """
        x = self._hash(item)
        # Use first precision bits for register index
        j = x & ((1 << self.precision) - 1)
        # Count leading zeros in remaining bits + 1
        w = x >> self.precision
        self.registers[j] = max(self.registers[j], self._leading_zeros(w) + 1)
    
    def _leading_zeros(self, w: int) -> int:
        """Count leading zero bits."""
        if w == 0:
            return 64 - self.precision
        return (64 - self.precision) - w.bit_length()
    
    def cardinality(self) -> int:
        """
        Estimate the cardinality (number of unique elements).
        
        Returns:
            Estimated number of unique elements
        """
        raw_estimate = self.alpha * (self.m ** 2) / sum(2 ** (-x) for x in self.registers)
        
        # Small range correction
        if raw_estimate <= 2.5 * self.m:
            zeros = self.registers.count(0)
            if zeros != 0:
                return int(self.m * math.log(self.m / zeros))
        
        # Large range correction
        if raw_estimate <= (1 << 32) / 30:
            return int(raw_estimate)
        else:
            return int(-1 * (1 << 32) * math.log(1 - raw_estimate / (1 << 32)))
    
    def merge(self, other: 'HyperLogLog') -> None:
        """
        Merge another HyperLogLog into this one.
        
        Args:
            other: Another HyperLogLog instance with the same precision
            
        Raises:
            ValueError: If precision doesn't match
        """
        if self.precision != other.precision:
            raise ValueError("Cannot merge HLLs with different precision")
        
        for i in range(self.m):
            self.registers[i] = max(self.registers[i], other.registers[i])
    
    def clear(self) -> None:
        """Reset all registers to zero."""
        self.registers = [0] * self.m
    
    def to_bytes(self) -> bytes:
        """
        Serialize the HyperLogLog to bytes.
        
        Returns:
            Byte representation of the HLL
        """
        # Store precision and registers
        data = bytes([self.precision])
        # Pack registers efficiently (each register is at most 64-p+1, so fits in a byte)
        # Validate registers are in valid range
        for reg in self.registers:
            if not 0 <= reg <= 255:
                raise ValueError(f"Register value {reg} out of byte range")
            data += bytes([reg])
        return data
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'HyperLogLog':
        """
        Deserialize a HyperLogLog from bytes.
        
        Args:
            data: Byte representation of an HLL
            
        Returns:
            HyperLogLog instance
        """
        precision = data[0]
        hll = cls(precision)
        hll.registers = list(data[1:1 + hll.m])
        return hll
    
    def __len__(self) -> int:
        """Return estimated cardinality."""
        return self.cardinality()
    
    def __repr__(self) -> str:
        return f"HyperLogLog(precision={self.precision}, cardinality≈{self.cardinality()})"
