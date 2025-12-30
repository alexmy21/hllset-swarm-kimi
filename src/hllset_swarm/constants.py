# src/hllset_swarm/constants.py
import hashlib


SHARED_SEED = 42
P_BITS = 10          # HLL precision
HASH_FUNC = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF