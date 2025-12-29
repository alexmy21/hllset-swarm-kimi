"""
Unit tests for HyperLogLog implementation.
"""

import pytest
from hllset import HyperLogLog


class TestHyperLogLog:
    """Test cases for HyperLogLog class."""
    
    def test_initialization(self):
        """Test HLL initialization."""
        hll = HyperLogLog(precision=10)
        assert hll.precision == 10
        assert hll.m == 1024
        assert hll.cardinality() == 0
    
    def test_precision_validation(self):
        """Test precision parameter validation."""
        with pytest.raises(ValueError):
            HyperLogLog(precision=3)
        with pytest.raises(ValueError):
            HyperLogLog(precision=17)
        
        # Valid precisions should work
        for p in range(4, 17):
            hll = HyperLogLog(precision=p)
            assert hll.precision == p
    
    def test_add_single_item(self):
        """Test adding a single item."""
        hll = HyperLogLog()
        hll.add("test")
        cardinality = hll.cardinality()
        assert cardinality >= 1
        assert cardinality <= 2  # Should be close to 1
    
    def test_add_multiple_unique_items(self):
        """Test adding multiple unique items."""
        hll = HyperLogLog()
        n = 1000
        for i in range(n):
            hll.add(f"item_{i}")
        
        cardinality = hll.cardinality()
        # Should be within reasonable error margin (typically < 2% for p=14)
        error = abs(cardinality - n) / n
        assert error < 0.05  # 5% error margin
    
    def test_add_duplicate_items(self):
        """Test that duplicate items don't increase cardinality."""
        hll = HyperLogLog()
        for _ in range(100):
            hll.add("same_item")
        
        cardinality = hll.cardinality()
        assert cardinality >= 1
        assert cardinality <= 2  # Should still be close to 1
    
    def test_cardinality_estimation_accuracy(self):
        """Test cardinality estimation accuracy for various sizes."""
        test_cases = [10, 100, 1000, 10000]
        
        for n in test_cases:
            hll = HyperLogLog(precision=14)
            for i in range(n):
                hll.add(str(i))
            
            cardinality = hll.cardinality()
            error = abs(cardinality - n) / n
            # HLL typically has ~1.04/sqrt(m) standard error for p=14
            assert error < 0.1, f"Error too high for n={n}: {error}"
    
    def test_merge_hlls(self):
        """Test merging two HLLs."""
        hll1 = HyperLogLog(precision=10)
        hll2 = HyperLogLog(precision=10)
        
        # Add different items to each
        for i in range(500):
            hll1.add(f"a_{i}")
        for i in range(500):
            hll2.add(f"b_{i}")
        
        # Merge hll2 into hll1
        hll1.merge(hll2)
        
        # Should estimate ~1000 unique items
        cardinality = hll1.cardinality()
        error = abs(cardinality - 1000) / 1000
        assert error < 0.1
    
    def test_merge_with_overlap(self):
        """Test merging HLLs with overlapping items."""
        hll1 = HyperLogLog(precision=10)
        hll2 = HyperLogLog(precision=10)
        
        # Add some common items
        for i in range(800):
            hll1.add(str(i))
        for i in range(200, 1000):
            hll2.add(str(i))
        
        # Merge should result in ~1000 unique items
        hll1.merge(hll2)
        cardinality = hll1.cardinality()
        error = abs(cardinality - 1000) / 1000
        assert error < 0.15
    
    def test_merge_different_precision_fails(self):
        """Test that merging HLLs with different precision fails."""
        hll1 = HyperLogLog(precision=10)
        hll2 = HyperLogLog(precision=12)
        
        with pytest.raises(ValueError):
            hll1.merge(hll2)
    
    def test_clear(self):
        """Test clearing an HLL."""
        hll = HyperLogLog()
        for i in range(100):
            hll.add(str(i))
        
        assert hll.cardinality() > 0
        hll.clear()
        assert hll.cardinality() == 0
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        hll1 = HyperLogLog(precision=10)
        for i in range(1000):
            hll1.add(str(i))
        
        cardinality1 = hll1.cardinality()
        
        # Serialize and deserialize
        data = hll1.to_bytes()
        hll2 = HyperLogLog.from_bytes(data)
        
        # Should have same cardinality
        cardinality2 = hll2.cardinality()
        assert cardinality1 == cardinality2
        assert hll2.precision == hll1.precision
    
    def test_len_method(self):
        """Test __len__ method."""
        hll = HyperLogLog()
        for i in range(100):
            hll.add(str(i))
        
        assert len(hll) == hll.cardinality()
    
    def test_repr(self):
        """Test __repr__ method."""
        hll = HyperLogLog(precision=10)
        hll.add("test")
        repr_str = repr(hll)
        assert "HyperLogLog" in repr_str
        assert "precision=10" in repr_str
        assert "cardinality" in repr_str
    
    def test_different_data_types(self):
        """Test adding different data types."""
        hll = HyperLogLog()
        
        # String
        hll.add("string")
        # Integer
        hll.add(42)
        # Float
        hll.add(3.14)
        # Bytes
        hll.add(b"bytes")
        
        # Should count all as unique
        cardinality = hll.cardinality()
        assert cardinality >= 3
        assert cardinality <= 5
    
    def test_empty_hll_cardinality(self):
        """Test cardinality of empty HLL."""
        hll = HyperLogLog()
        assert hll.cardinality() == 0
    
    def test_large_dataset(self):
        """Test with a larger dataset."""
        hll = HyperLogLog(precision=14)
        n = 50000
        
        for i in range(n):
            hll.add(f"item_{i}")
        
        cardinality = hll.cardinality()
        error = abs(cardinality - n) / n
        # Should be within 3% for large datasets with p=14
        assert error < 0.03
