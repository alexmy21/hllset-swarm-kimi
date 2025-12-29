"""
Example of using SwarmNode for distributed cardinality estimation.
"""

import time
from hllset import SwarmNode


def main():
    print("SwarmNode Distributed Example")
    print("=" * 50)
    
    # Create three nodes in a swarm
    with SwarmNode("node1", precision=12) as node1, \
         SwarmNode("node2", precision=12) as node2, \
         SwarmNode("node3", precision=12) as node3:
        
        print("\nCreated 3 nodes:")
        print(f"  - Node1 on port {node1.port}")
        print(f"  - Node2 on port {node2.port}")
        print(f"  - Node3 on port {node3.port}")
        
        # Connect nodes in a network
        print("\nConnecting nodes...")
        node1.add_peer("node2", "localhost", node2.port)
        node1.add_peer("node3", "localhost", node3.port)
        
        node2.add_peer("node1", "localhost", node1.port)
        node2.add_peer("node3", "localhost", node3.port)
        
        node3.add_peer("node1", "localhost", node1.port)
        node3.add_peer("node2", "localhost", node2.port)
        
        # Add data to different nodes
        print("\nAdding data to nodes...")
        print("  - Node1: users 0-999")
        for i in range(1000):
            node1.add(f"user_{i}")
        
        print("  - Node2: users 500-1499")
        for i in range(500, 1500):
            node2.add(f"user_{i}")
        
        print("  - Node3: users 1000-1999")
        for i in range(1000, 2000):
            node3.add(f"user_{i}")
        
        # Check local cardinalities before sync
        print("\nLocal cardinalities (before sync):")
        print(f"  - Node1: {node1.cardinality()}")
        print(f"  - Node2: {node2.cardinality()}")
        print(f"  - Node3: {node3.cardinality()}")
        
        # Announce nodes to each other
        print("\nAnnouncing nodes...")
        node1.announce()
        node2.announce()
        node3.announce()
        time.sleep(0.2)
        
        # Sync data across nodes
        print("\nSyncing data across nodes...")
        node1.sync()
        node2.sync()
        node3.sync()
        time.sleep(0.5)  # Give time for all syncs to complete
        
        # Check cardinalities after sync
        print("\nLocal cardinalities (after sync):")
        card1 = node1.cardinality()
        card2 = node2.cardinality()
        card3 = node3.cardinality()
        
        print(f"  - Node1: {card1}")
        print(f"  - Node2: {card2}")
        print(f"  - Node3: {card3}")
        
        avg_cardinality = (card1 + card2 + card3) / 3
        print(f"\nAverage cardinality: {avg_cardinality:.0f}")
        print(f"Expected: ~2000 unique users")
        print(f"Error: {abs(avg_cardinality - 2000) / 2000 * 100:.2f}%")
        
        # Demonstrate adding more data after sync
        print("\n" + "=" * 50)
        print("Adding more data and re-syncing...")
        
        for i in range(2000, 2500):
            node1.add(f"user_{i}")
        
        node1.sync()
        time.sleep(0.3)
        
        print(f"\nNode2 cardinality after Node1 added more data: {node2.cardinality()}")
        print(f"Expected: ~2500")


def simple_two_node_example():
    """Simple example with just two nodes."""
    print("\n" + "=" * 50)
    print("Simple Two-Node Example")
    print("=" * 50)
    
    with SwarmNode("alice", precision=10) as alice, \
         SwarmNode("bob", precision=10) as bob:
        
        # Alice tracks her users
        print("\nAlice adds 500 users...")
        for i in range(500):
            alice.add(f"user_{i}")
        
        # Bob tracks his users
        print("Bob adds 500 users (with some overlap)...")
        for i in range(300, 800):
            bob.add(f"user_{i}")
        
        print(f"\nAlice's local count: {alice.cardinality()}")
        print(f"Bob's local count: {bob.cardinality()}")
        
        # Connect and sync
        print("\nConnecting nodes...")
        alice.add_peer("bob", "localhost", bob.port)
        bob.add_peer("alice", "localhost", alice.port)
        
        alice.sync()
        bob.sync()
        time.sleep(0.3)
        
        print(f"\nAfter sync:")
        print(f"Alice's count: {alice.cardinality()}")
        print(f"Bob's count: {bob.cardinality()}")
        print(f"Expected: ~800 unique users (500 + 500 - 200 overlap)")


if __name__ == "__main__":
    main()
    simple_two_node_example()
