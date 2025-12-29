"""
Light-weight swarm coordination for distributed HyperLogLog.

This module provides basic node coordination without external dependencies.
"""

import json
import socket
import threading
import time
from typing import Dict, List, Optional, Callable
from .hll import HyperLogLog


class SwarmNode:
    """
    A simple swarm node for coordinating HyperLogLog instances.
    
    This is a lightweight implementation that allows nodes to share
    and merge HLL data for distributed cardinality estimation.
    
    Args:
        node_id: Unique identifier for this node
        precision: HLL precision parameter (default 14)
        port: Port to listen on for incoming connections (default 0 = auto)
    """
    
    def __init__(self, node_id: str, precision: int = 14, port: int = 0):
        self.node_id = node_id
        self.precision = precision
        self.hll = HyperLogLog(precision)
        self.peers: Dict[str, tuple] = {}  # peer_id -> (host, port)
        self.running = False
        self.lock = threading.Lock()
        
        # Set up listener
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('localhost', port))
        self.port = self.socket.getsockname()[1]
        
        self._listener_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start the swarm node listener."""
        if self.running:
            return
        
        self.running = True
        self.socket.listen(5)
        self._listener_thread = threading.Thread(target=self._listen, daemon=True)
        self._listener_thread.start()
    
    def stop(self) -> None:
        """Stop the swarm node listener."""
        self.running = False
        if self._listener_thread:
            self.socket.close()
            self._listener_thread.join(timeout=1)
    
    def _listen(self) -> None:
        """Listen for incoming connections."""
        while self.running:
            try:
                self.socket.settimeout(1.0)
                conn, addr = self.socket.accept()
                threading.Thread(
                    target=self._handle_connection,
                    args=(conn,),
                    daemon=True
                ).start()
            except socket.timeout:
                continue
            except Exception:
                if self.running:
                    break
    
    def _handle_connection(self, conn: socket.socket) -> None:
        """Handle an incoming connection."""
        try:
            data = b''
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            
            if data:
                message = json.loads(data.decode('utf-8'))
                self._process_message(message)
        except Exception:
            pass
        finally:
            conn.close()
    
    def _process_message(self, message: dict) -> None:
        """Process a message from another node."""
        msg_type = message.get('type')
        
        if msg_type == 'sync':
            # Merge HLL data from peer
            hll_data = bytes.fromhex(message['hll_data'])
            peer_hll = HyperLogLog.from_bytes(hll_data)
            with self.lock:
                self.hll.merge(peer_hll)
        
        elif msg_type == 'peer_announce':
            # Register a new peer
            peer_id = message['peer_id']
            host = message['host']
            port = message['port']
            self.peers[peer_id] = (host, port)
    
    def add(self, item) -> None:
        """
        Add an item to this node's HyperLogLog.
        
        Args:
            item: Item to add
        """
        with self.lock:
            self.hll.add(item)
    
    def add_peer(self, peer_id: str, host: str, port: int) -> None:
        """
        Add a peer node to the swarm.
        
        Args:
            peer_id: Unique identifier for the peer
            host: Hostname or IP address of the peer
            port: Port number of the peer
        """
        self.peers[peer_id] = (host, port)
    
    def sync(self) -> None:
        """Synchronize HLL data with all peers."""
        with self.lock:
            hll_data = self.hll.to_bytes().hex()
        
        message = {
            'type': 'sync',
            'node_id': self.node_id,
            'hll_data': hll_data
        }
        
        self._broadcast(message)
    
    def announce(self) -> None:
        """Announce this node to all peers."""
        message = {
            'type': 'peer_announce',
            'peer_id': self.node_id,
            'host': 'localhost',
            'port': self.port
        }
        
        self._broadcast(message)
    
    def _broadcast(self, message: dict) -> None:
        """Broadcast a message to all peers."""
        data = json.dumps(message).encode('utf-8')
        
        for peer_id, (host, port) in list(self.peers.items()):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect((host, port))
                sock.sendall(data)
                sock.close()
            except Exception:
                # Peer might be down, continue with others
                pass
    
    def cardinality(self) -> int:
        """
        Get the estimated cardinality from this node's HLL.
        
        Returns:
            Estimated number of unique elements
        """
        with self.lock:
            return self.hll.cardinality()
    
    def get_global_cardinality(self) -> int:
        """
        Get the global cardinality across all peers.
        
        Note: This performs a sync first to ensure up-to-date data.
        
        Returns:
            Estimated number of unique elements globally
        """
        self.sync()
        time.sleep(0.1)  # Give peers time to sync
        return self.cardinality()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def __repr__(self) -> str:
        return f"SwarmNode(id={self.node_id}, cardinality≈{self.cardinality()}, peers={len(self.peers)})"
