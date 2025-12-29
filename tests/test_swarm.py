"""
Unit tests for SwarmNode implementation.
"""

import pytest
import time
from hllset import SwarmNode


class TestSwarmNode:
    """Test cases for SwarmNode class."""
    
    def test_initialization(self):
        """Test SwarmNode initialization."""
        node = SwarmNode("node1", precision=10)
        assert node.node_id == "node1"
        assert node.precision == 10
        assert node.port > 0  # Should have auto-assigned port
        assert len(node.peers) == 0
    
    def test_start_stop(self):
        """Test starting and stopping a node."""
        node = SwarmNode("node1")
        assert not node.running
        
        node.start()
        assert node.running
        
        node.stop()
        assert not node.running
    
    def test_context_manager(self):
        """Test using SwarmNode as context manager."""
        with SwarmNode("node1") as node:
            assert node.running
        
        assert not node.running
    
    def test_add_item(self):
        """Test adding items to a swarm node."""
        node = SwarmNode("node1")
        node.add("item1")
        node.add("item2")
        node.add("item3")
        
        cardinality = node.cardinality()
        assert cardinality >= 2
        assert cardinality <= 4
    
    def test_add_peer(self):
        """Test adding a peer."""
        node = SwarmNode("node1")
        node.add_peer("node2", "localhost", 5000)
        
        assert "node2" in node.peers
        assert node.peers["node2"] == ("localhost", 5000)
    
    def test_single_node_cardinality(self):
        """Test cardinality estimation on a single node."""
        node = SwarmNode("node1", precision=12)
        
        for i in range(1000):
            node.add(f"item_{i}")
        
        cardinality = node.cardinality()
        error = abs(cardinality - 1000) / 1000
        assert error < 0.1
    
    def test_two_nodes_sync(self):
        """Test syncing data between two nodes."""
        # Create two nodes
        with SwarmNode("node1", precision=10) as node1, \
             SwarmNode("node2", precision=10) as node2:
            
            # Add items to node1
            for i in range(500):
                node1.add(f"a_{i}")
            
            # Add items to node2
            for i in range(500):
                node2.add(f"b_{i}")
            
            # Connect nodes
            node1.add_peer("node2", "localhost", node2.port)
            node2.add_peer("node1", "localhost", node1.port)
            
            # Sync from node1 to node2
            node1.sync()
            time.sleep(0.2)  # Give time for sync
            
            # Node2 should now have data from both
            cardinality = node2.cardinality()
            # Should be close to 1000 (500 + 500)
            error = abs(cardinality - 1000) / 1000
            assert error < 0.2
    
    def test_announce(self):
        """Test node announcement to peers."""
        with SwarmNode("node1") as node1, \
             SwarmNode("node2") as node2:
            
            # Node1 announces to node2
            node1.add_peer("node2", "localhost", node2.port)
            node1.announce()
            time.sleep(0.1)
            
            # Node2 should know about node1
            assert "node1" in node2.peers
    
    def test_repr(self):
        """Test __repr__ method."""
        node = SwarmNode("test_node")
        node.add("item1")
        repr_str = repr(node)
        
        assert "SwarmNode" in repr_str
        assert "test_node" in repr_str
        assert "cardinality" in repr_str
        assert "peers" in repr_str
    
    def test_multiple_adds_same_item(self):
        """Test adding the same item multiple times."""
        node = SwarmNode("node1")
        
        for _ in range(100):
            node.add("same_item")
        
        cardinality = node.cardinality()
        assert cardinality >= 1
        assert cardinality <= 2
    
    def test_get_global_cardinality(self):
        """Test getting global cardinality (basic test without peers)."""
        node = SwarmNode("node1")
        
        for i in range(100):
            node.add(f"item_{i}")
        
        # Without peers, global cardinality should be same as local
        global_card = node.get_global_cardinality()
        local_card = node.cardinality()
        
        assert abs(global_card - local_card) <= 1
    
    def test_node_isolation(self):
        """Test that nodes are isolated without sync."""
        with SwarmNode("node1", precision=10) as node1, \
             SwarmNode("node2", precision=10) as node2:
            
            # Add items only to node1
            for i in range(500):
                node1.add(f"item_{i}")
            
            # Node2 should still be empty
            assert node2.cardinality() == 0
    
    def test_bidirectional_sync(self):
        """Test bidirectional sync between nodes."""
        with SwarmNode("node1", precision=10) as node1, \
             SwarmNode("node2", precision=10) as node2:
            
            # Add different items to each node
            for i in range(300):
                node1.add(f"a_{i}")
            for i in range(400):
                node2.add(f"b_{i}")
            
            # Connect nodes bidirectionally
            node1.add_peer("node2", "localhost", node2.port)
            node2.add_peer("node1", "localhost", node1.port)
            
            # Both sync
            node1.sync()
            node2.sync()
            time.sleep(0.3)
            
            # Both should have similar cardinality (~700)
            card1 = node1.cardinality()
            card2 = node2.cardinality()
            
            # Both should be reasonably close to 700
            assert abs(card1 - 700) < 150
            assert abs(card2 - 700) < 150
            # And close to each other
            assert abs(card1 - card2) < 100
