# hllset-swarm-kimi

Light version of hllset-swarm - A minimal distributed HyperLogLog cardinality estimation system.

## Overview

hllset-swarm-kimi is a lightweight implementation of distributed HyperLogLog (HLL) for approximate cardinality estimation across multiple nodes. This "light" version provides core functionality without heavy external dependencies, making it easy to integrate and deploy.

## Features

- **Pure Python HyperLogLog**: No external dependencies for core HLL functionality
- **Distributed Architecture**: Swarm-based coordination for sharing cardinality data across nodes
- **Lightweight**: Minimal footprint, easy to embed in existing systems
- **Accurate**: Typical error rate < 2% for cardinality estimation
- **Serializable**: Easy persistence and data transfer

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic HyperLogLog Usage

```python
from hllset import HyperLogLog

# Create an HLL counter
hll = HyperLogLog(precision=14)

# Add items
for i in range(10000):
    hll.add(f"user_{i}")

# Estimate cardinality
print(f"Estimated unique users: {hll.cardinality()}")

# Merge with another HLL
hll2 = HyperLogLog(precision=14)
for i in range(5000, 15000):
    hll2.add(f"user_{i}")

hll.merge(hll2)
print(f"Combined cardinality: {hll.cardinality()}")
```

### Distributed Swarm Usage

```python
from hllset import SwarmNode
import time

# Create nodes
with SwarmNode("node1") as node1, SwarmNode("node2") as node2:
    # Connect nodes
    node1.add_peer("node2", "localhost", node2.port)
    node2.add_peer("node1", "localhost", node1.port)
    
    # Add data to different nodes
    for i in range(1000):
        node1.add(f"user_{i}")
    for i in range(500, 1500):
        node2.add(f"user_{i}")
    
    # Sync data
    node1.sync()
    node2.sync()
    time.sleep(0.2)
    
    # Both nodes now have merged view
    print(f"Node1 cardinality: {node1.cardinality()}")
    print(f"Node2 cardinality: {node2.cardinality()}")
```

## Examples

See the `examples/` directory for more detailed usage:
- `basic_hll.py` - Basic HyperLogLog operations
- `swarm_example.py` - Distributed swarm coordination

Run examples:
```bash
python examples/basic_hll.py
python examples/swarm_example.py
```

## Testing

Run tests with pytest:
```bash
pytest tests/
```

With coverage:
```bash
pytest --cov=hllset tests/
```

## Architecture

### HyperLogLog

The `HyperLogLog` class implements the HLL algorithm for cardinality estimation:
- **Precision**: Configurable from 4 to 16 (default 14)
- **Accuracy**: Standard error ≈ 1.04/√m where m = 2^precision
- **Memory**: Uses m registers, each 1 byte (e.g., 16KB for precision=14)
- **Operations**: add, merge, cardinality, serialization

### SwarmNode

The `SwarmNode` class provides distributed coordination:
- **Peer Discovery**: Manual peer registration
- **Synchronization**: Push-based HLL data sharing
- **Network**: Simple TCP socket-based communication
- **Thread-safe**: Safe for concurrent operations

## Use Cases

- **Web Analytics**: Count unique visitors across multiple servers
- **Distributed Systems**: Estimate unique items in sharded databases
- **Stream Processing**: Approximate distinct counts in data streams
- **IoT**: Aggregate unique event counts from edge devices
- **Microservices**: Share cardinality metrics across service instances

## Limitations (Light Version)

This light version intentionally excludes some features for simplicity:
- No persistent storage (in-memory only)
- No automatic peer discovery
- No TLS/encryption for node communication
- Basic error handling
- Single-threaded sync operations

For production use cases requiring these features, consider the full hllset-swarm implementation.

## Performance

Typical performance characteristics:
- **Add operation**: O(1), ~1-2 microseconds
- **Cardinality estimation**: O(m), ~100-500 microseconds for p=14
- **Merge operation**: O(m), ~100-500 microseconds for p=14
- **Memory usage**: 2^precision bytes per HLL

## Contributing

This is a lightweight demonstration version. For contributions, please ensure:
1. Tests pass: `pytest tests/`
2. Code is documented
3. Changes maintain the "light" philosophy (minimal dependencies)

## License

MIT License - see LICENSE file for details.

## References

- [HyperLogLog Paper](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)
- [HyperLogLog in Practice](https://research.google/pubs/pub40671/)

## Version

Current version: 0.1.0 (Light Release)
