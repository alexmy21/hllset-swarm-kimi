"""
Basic example of using HyperLogLog for cardinality estimation.
"""

from hllset import HyperLogLog


def main():
    print("HyperLogLog Basic Example")
    print("=" * 50)
    
    # Create a HyperLogLog with default precision (14)
    hll = HyperLogLog()
    
    # Add some items
    print("\nAdding 10,000 unique items...")
    for i in range(10000):
        hll.add(f"user_{i}")
    
    estimated = hll.cardinality()
    error = abs(estimated - 10000) / 10000 * 100
    
    print(f"True cardinality: 10,000")
    print(f"Estimated cardinality: {estimated:,}")
    print(f"Error: {error:.2f}%")
    
    # Test with duplicates
    print("\n" + "=" * 50)
    print("Testing with duplicates...")
    hll2 = HyperLogLog()
    
    # Add same items multiple times
    for _ in range(5):
        for i in range(1000):
            hll2.add(f"item_{i}")
    
    estimated2 = hll2.cardinality()
    error2 = abs(estimated2 - 1000) / 1000 * 100
    
    print(f"Added 5,000 items (1,000 unique)")
    print(f"True cardinality: 1,000")
    print(f"Estimated cardinality: {estimated2:,}")
    print(f"Error: {error2:.2f}%")
    
    # Merging HLLs
    print("\n" + "=" * 50)
    print("Merging two HyperLogLogs...")
    
    hll_a = HyperLogLog()
    hll_b = HyperLogLog()
    
    for i in range(5000):
        hll_a.add(f"a_{i}")
    for i in range(3000, 8000):
        hll_b.add(f"a_{i}")
    
    print(f"HLL A cardinality: {hll_a.cardinality():,}")
    print(f"HLL B cardinality: {hll_b.cardinality():,}")
    
    hll_a.merge(hll_b)
    merged_cardinality = hll_a.cardinality()
    
    print(f"Merged cardinality: {merged_cardinality:,}")
    print(f"Expected: 8,000 (overlap from 3000-5000)")
    
    # Serialization
    print("\n" + "=" * 50)
    print("Serialization example...")
    
    hll3 = HyperLogLog(precision=10)
    for i in range(1000):
        hll3.add(str(i))
    
    # Serialize
    data = hll3.to_bytes()
    print(f"Serialized size: {len(data)} bytes")
    
    # Deserialize
    hll4 = HyperLogLog.from_bytes(data)
    print(f"Original cardinality: {hll3.cardinality()}")
    print(f"Deserialized cardinality: {hll4.cardinality()}")
    print(f"Match: {hll3.cardinality() == hll4.cardinality()}")


if __name__ == "__main__":
    main()
