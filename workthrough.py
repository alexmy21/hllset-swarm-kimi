import numpy as np
from collections import defaultdict, Counter
from typing import List, Set, Dict, Tuple, Optional
import random
from dataclasses import dataclass
from functools import lru_cache
import heapq
from scipy.sparse import csr_matrix, lil_matrix
import matplotlib.pyplot as plt

# ============================================================================
# 1. TOKEN LAYER & HRT AM IMPLEMENTATION
# ============================================================================

class TokenLayer:
    """Handles token vocabulary, n-gram extraction, and HRT AM construction"""
    
    def __init__(self, base_vocab_size: int = 1000, seed: int = 42):
        """
        Initialize token layer with base vocabulary
        
        Args:
            base_vocab_size: Size of 1-gram vocabulary (simplified from 80K for POC)
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Base vocabulary (1-grams)
        self.vocab = [f"chr_{i}" for i in range(base_vocab_size)]
        self.vocab_dict = {token: idx for idx, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        # Special tokens
        self.START = "START"
        self.END = "END"
        
        # Extended vocabulary with n-grams
        self.n_gram_vocab = {}
        self.n_gram_to_idx = {}
        
        # HRT Adjacency Matrix (sparse)
        self.hrt_am = lil_matrix((self.vocab_size, self.vocab_size), dtype=np.float32)
        self.transpose_am = None
        
        # Frequency statistics
        self.freq_matrix = lil_matrix((self.vocab_size, self.vocab_size), dtype=np.int32)
        
    def extract_n_grams(self, text: str, max_n: int = 3) -> List[str]:
        """
        Extract n-grams from text
        
        Args:
            text: Input text string
            max_n: Maximum n-gram size (1, 2, or 3)
            
        Returns:
            List of tokens (1-grams + n-grams)
        """
        # Split into characters (1-grams)
        chars = list(text)
        
        # Build n-grams
        tokens = []
        for n in range(1, max_n + 1):
            for i in range(len(chars) - n + 1):
                n_gram = ''.join(chars[i:i+n])
                tokens.append(n_gram)
                
                # Add to extended vocabulary if not exists
                if n_gram not in self.n_gram_to_idx and n_gram not in self.vocab_dict:
                    idx = len(self.vocab_dict) + len(self.n_gram_to_idx)
                    self.n_gram_to_idx[n_gram] = idx
                    self.n_gram_vocab[idx] = n_gram
        
        return tokens
    
    def update_hrt_am(self, token_sequences: List[List[str]]):
        """
        Update HRT AM with token sequences
        
        Args:
            token_sequences: List of token sequences with START/END markers
        """
        for seq in token_sequences:
            # Add START and END to sequence
            full_seq = [self.START] + seq + [self.END]
            
            # Update adjacency frequencies
            for i in range(len(full_seq) - 1):
                token_from = full_seq[i]
                token_to = full_seq[i + 1]
                
                # Get indices (handle base vocab and n-grams)
                idx_from = self._get_token_idx(token_from)
                idx_to = self._get_token_idx(token_to)
                
                # Update frequency matrix
                if idx_from is not None and idx_to is not None:
                    self.freq_matrix[idx_from, idx_to] += 1
        
        # Normalize to create probability matrix
        row_sums = self.freq_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        
        # Convert to probabilities
        for i in range(self.vocab_size):
            row_sum = row_sums[i, 0]
            if row_sum > 0:
                for j in range(self.vocab_size):
                    if self.freq_matrix[i, j] > 0:
                        self.hrt_am[i, j] = self.freq_matrix[i, j] / row_sum
        
        # Cache transpose
        self.transpose_am = self.hrt_am.T.tolil()
    
    def _get_token_idx(self, token: str) -> Optional[int]:
        """Get index of token in vocabulary"""
        if token == self.START:
            return -1  # Special handling
        elif token == self.END:
            return -2  # Special handling
        elif token in self.vocab_dict:
            return self.vocab_dict[token]
        elif token in self.n_gram_to_idx:
            return self.n_gram_to_idx[token]
        else:
            return None
    
    def get_followers(self, token_idx: int, threshold: float = 0.0) -> Set[int]:
        """
        Get set of tokens that follow given token with probability > threshold
        
        Args:
            token_idx: Index of token
            threshold: Probability threshold
            
        Returns:
            Set of follower token indices
        """
        followers = set()
        if 0 <= token_idx < self.vocab_size:
            row = self.hrt_am.getrow(token_idx)
            for col_idx in row.nonzero()[1]:
                if row[0, col_idx] > threshold:
                    followers.add(col_idx)
        return followers
    
    def get_predecessors(self, token_idx: int, threshold: float = 0.0) -> Set[int]:
        """
        Get set of tokens that precede given token with probability > threshold
        
        Args:
            token_idx: Index of token
            threshold: Probability threshold
            
        Returns:
            Set of predecessor token indices
        """
        predecessors = set()
        if 0 <= token_idx < self.vocab_size and self.transpose_am is not None:
            row = self.transpose_am.getrow(token_idx)
            for col_idx in row.nonzero()[1]:
                if row[0, col_idx] > threshold:
                    predecessors.add(col_idx)
        return predecessors
    
# ============================================================================
# 2. HLLSET IMPLEMENTATION (Simplified for POC)
# ============================================================================

class HLLSet:
    """Simplified HLLSet implementation (using Python sets for clarity)"""
    
    def __init__(self, token_indices: Set[int], 
                 hll_id: str,
                 tau: float = 0.7,
                 rho: float = 0.3):
        """
        Initialize HLLSet
        
        Args:
            token_indices: Set of token indices
            hll_id: Unique identifier
            tau: Inclusion tolerance
            rho: Exclusion intolerance
        """
        self.tokens = set(token_indices)
        self.id = hll_id
        self.tau = tau
        self.rho = rho
        self.cardinality = len(token_indices)
        self.creation_time = 0
        
    def union(self, other: 'HLLSet') -> 'HLLSet':
        """Union of two HLLSets"""
        new_tokens = self.tokens.union(other.tokens)
        return HLLSet(new_tokens, f"{self.id}_U_{other.id}")
    
    def intersection(self, other: 'HLLSet') -> 'HLLSet':
        """Intersection of two HLLSets"""
        new_tokens = self.tokens.intersection(other.tokens)
        return HLLSet(new_tokens, f"{self.id}_I_{other.id}")
    
    def difference(self, other: 'HLLSet') -> 'HLLSet':
        """Difference of two HLLSets"""
        new_tokens = self.tokens.difference(other.tokens)
        return HLLSet(new_tokens, f"{self.id}_D_{other.id}")
    
    def jaccard_similarity(self, other: 'HLLSet') -> float:
        """Jaccard similarity (simplified BSS)"""
        if not self.tokens and not other.tokens:
            return 1.0
        intersection = len(self.tokens.intersection(other.tokens))
        union = len(self.tokens.union(other.tokens))
        return intersection / union if union > 0 else 0.0
    
    def bss_tau(self, other: 'HLLSet') -> float:
        """Bell State Similarity for inclusion tolerance"""
        intersection = len(self.tokens.intersection(other.tokens))
        return intersection / len(other.tokens) if len(other.tokens) > 0 else 0.0
    
    def bss_rho(self, other: 'HLLSet') -> float:
        """Bell State Similarity for exclusion intolerance"""
        difference = len(self.tokens.difference(other.tokens))
        return difference / len(other.tokens) if len(other.tokens) > 0 else 0.0
    
    def is_related(self, other: 'HLLSet') -> bool:
        """Check if two HLLSets are related based on tau and rho"""
        tau_sim = self.bss_tau(other)
        rho_sim = self.bss_rho(other)
        return tau_sim >= self.tau and rho_sim <= self.rho
    
    def __repr__(self):
        return f"HLLSet(id={self.id}, |tokens|={self.cardinality})"

# ============================================================================
# 3. CORTEX & BASIC HLLSETS
# ============================================================================

class HLLSetCortex:
    """Manages HLLSet cortex and basic HLLSet generation"""
    
    def __init__(self, token_layer: TokenLayer):
        """
        Initialize cortex
        
        Args:
            token_layer: TokenLayer instance
        """
        self.token_layer = token_layer
        self.basic_hllsets = {}  # Map from (row/col, idx) to HLLSet
        self.active_hllsets = {}  # Currently active HLLSets (particles in swarm)
        self.cortex_tokens = set()  # Union of all active HLLSet tokens
        self.iteration = 0
        
    def build_basic_hllsets(self, freq_threshold: float = 0.0):
        """
        Build basic HLLSets from HRT AM
        
        Args:
            freq_threshold: Minimum frequency threshold for inclusion
        """
        # Row HLLSets (tokens that follow)
        for i in range(self.token_layer.vocab_size):
            followers = self.token_layer.get_followers(i, freq_threshold)
            if followers:
                hll_id = f"row_{i}"
                self.basic_hllsets[('row', i)] = HLLSet(followers, hll_id)
        
        # Column HLLSets (tokens that precede)
        for j in range(self.token_layer.vocab_size):
            predecessors = self.token_layer.get_predecessors(j, freq_threshold)
            if predecessors:
                hll_id = f"col_{j}"
                self.basic_hllsets[('col', j)] = HLLSet(predecessors, hll_id)
        
        print(f"Built {len(self.basic_hllsets)} basic HLLSets")
    
    def initialize_active_hllsets(self, n_active: int = 50):
        """
        Initialize active HLLSets (particles in swarm)
        
        Args:
            n_active: Number of active HLLSets to initialize
        """
        basic_keys = list(self.basic_hllsets.keys())
        if len(basic_keys) < n_active:
            n_active = len(basic_keys)
        
        selected_keys = random.sample(basic_keys, n_active)
        for key in selected_keys:
            basic_hll = self.basic_hllsets[key]
            # Create active version (could add noise/adaptation)
            active_hll = HLLSet(
                basic_hll.tokens.copy(),
                f"active_{key[0]}_{key[1]}",
                tau=basic_hll.tau,
                rho=basic_hll.rho
            )
            active_hll.creation_time = self.iteration
            self.active_hllsets[active_hll.id] = active_hll
        
        self._update_cortex_tokens()
    
    def _update_cortex_tokens(self):
        """Update cortex tokens from active HLLSets"""
        self.cortex_tokens = set()
        for hll in self.active_hllsets.values():
            self.cortex_tokens.update(hll.tokens)
    
    def find_canonical_cover(self, target_tokens: Set[int] = None) -> Dict[str, HLLSet]:
        """
        Find minimal set of basic HLLSets that covers target tokens
        
        Args:
            target_tokens: Tokens to cover (defaults to cortex tokens)
            
        Returns:
            Dictionary of covering HLLSets
        """
        if target_tokens is None:
            target_tokens = self.cortex_tokens.copy()
        
        if not target_tokens:
            return {}
        
        # Greedy set cover algorithm (can be improved with constraint programming)
        uncovered = target_tokens.copy()
        cover = {}
        
        # Sort basic HLLSets by coverage per size
        basic_items = list(self.basic_hllsets.items())
        basic_items.sort(key=lambda x: len(x[1].tokens.intersection(uncovered)), reverse=True)
        
        while uncovered and basic_items:
            best_key, best_hll = basic_items.pop(0)
            covered = best_hll.tokens.intersection(uncovered)
            
            if covered:
                cover[best_hll.id] = best_hll
                uncovered -= covered
                
                # Re-sort based on remaining uncovered
                basic_items.sort(key=lambda x: len(x[1].tokens.intersection(uncovered)), reverse=True)
        
        return cover
    
    def get_optimal_cover_size(self, target_tokens: Set[int] = None) -> Tuple[int, float]:
        """
        Get optimal cover size and coverage ratio
        
        Args:
            target_tokens: Tokens to cover
            
        Returns:
            Tuple of (cover_size, coverage_ratio)
        """
        if target_tokens is None:
            target_tokens = self.cortex_tokens.copy()
        
        if not target_tokens:
            return 0, 1.0
        
        cover = self.find_canonical_cover(target_tokens)
        cover_tokens = set()
        for hll in cover.values():
            cover_tokens.update(hll.tokens)
        
        coverage = len(cover_tokens.intersection(target_tokens)) / len(target_tokens)
        return len(cover), coverage


# ============================================================================
# 4. PARTICLE SWARM MANAGEMENT (PSM)
# ============================================================================

@dataclass
class ParticleState:
    """State of a particle (HLLSet) in the swarm"""
    hllset_id: str
    tokens: Set[int]
    velocity: Set[int]  # Tokens to add/remove
    best_local: Set[int]  # Best local configuration (canonical cover)
    age: int = 0

class ParticleSwarmManager:
    """Manages HLLSet particles as a swarm"""
    
    def __init__(self, cortex: HLLSetCortex,
                 w: float = 0.5,  # Inertia weight
                 c1: float = 1.0,  # Local best weight
                 c2: float = 1.0,  # Global best weight
                 c3: float = 0.5): # Environmental force weight
        """
        Initialize PSM
        
        Args:
            cortex: HLLSetCortex instance
            w, c1, c2, c3: PSM parameters
        """
        self.cortex = cortex
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        
        # Swarm state
        self.particles = {}  # Map from HLLSet ID to ParticleState
        self.global_best = set()  # Global best token set
        self.conservation_target = 0  # Noether: |N| - |D| = 0
        self.iteration = 0
        
        # Statistics
        self.noether_violations = []
        self.swarm_sizes = []
    
    def initialize_particles(self):
        """Initialize particles from active HLLSets"""
        self.particles = {}
        for hll_id, hllset in self.cortex.active_hllsets.items():
            # Get canonical cover for this HLLSet as local best
            local_cover = self.cortex.find_canonical_cover(hllset.tokens)
            local_best = set()
            for hll in local_cover.values():
                local_best.update(hll.tokens)
            
            self.particles[hll_id] = ParticleState(
                hllset_id=hll_id,
                tokens=hllset.tokens.copy(),
                velocity=set(),
                best_local=local_best
            )
        
        # Initialize global best as union of all cortex tokens
        self.global_best = self.cortex.cortex_tokens.copy()
    
    def update_velocity(self, D: Set[int], R: Set[int], N: Set[int]):
        """
        Update particle velocities based on forces
        
        Args:
            D: Tokens to delete
            R: Tokens to retain
            N: Tokens to add
        """
        environmental_force = N  # Simplified: N as environmental push
        
        for particle in self.particles.values():
            # Current tokens
            current = particle.tokens
            
            # 1. Inertia: Continue with current velocity
            inertia = set(random.sample(list(particle.velocity), 
                                       min(len(particle.velocity), 
                                           int(self.w * len(particle.velocity)))))
            
            # 2. Local best attraction
            local_attraction = set()
            if particle.best_local:
                # Move toward local best
                to_add = particle.best_local - current
                to_remove = current - particle.best_local
                # Sample based on c1
                if to_add:
                    local_attraction.update(random.sample(list(to_add), 
                                                         min(len(to_add), 
                                                             int(self.c1 * len(to_add)))))
            
            # 3. Global best attraction
            global_attraction = set()
            if self.global_best:
                to_add_global = self.global_best - current
                if to_add_global:
                    global_attraction.update(random.sample(list(to_add_global),
                                                          min(len(to_add_global),
                                                              int(self.c2 * len(to_add_global)))))
            
            # 4. Environmental force (N tokens)
            env_force = set()
            if environmental_force:
                env_force.update(random.sample(list(environmental_force),
                                              min(len(environmental_force),
                                                  int(self.c3 * len(environmental_force)))))
            
            # Combine forces
            new_velocity = inertia.union(local_attraction).union(global_attraction).union(env_force)
            
            # Limit velocity size
            max_velocity = max(10, len(current) // 2)
            if len(new_velocity) > max_velocity:
                new_velocity = set(random.sample(list(new_velocity), max_velocity))
            
            particle.velocity = new_velocity
    
    def update_positions(self, D: Set[int], R: Set[int], N: Set[int]):
        """
        Update particle positions (token sets)
        
        Args:
            D: Tokens to delete
            R: Tokens to retain
            N: Tokens to add
            
        Returns:
            Tuple of (actual_D, actual_N) for conservation tracking
        """
        total_D = set()
        total_N = set()
        
        for particle in self.particles.values():
            current = particle.tokens
            
            # Apply velocity: add tokens from velocity, remove some existing
            tokens_to_add = particle.velocity - current
            
            # Remove tokens: some from D, some random
            tokens_to_remove = set()
            if D:
                tokens_to_remove.update(current.intersection(D))
            
            # Random removal for exploration
            if len(current) > 10:
                n_random_remove = min(5, len(current) // 10)
                tokens_to_remove.update(random.sample(list(current), n_random_remove))
            
            # Apply changes
            new_tokens = (current - tokens_to_remove).union(tokens_to_add)
            
            # Update particle
            particle.tokens = new_tokens
            particle.age += 1
            
            # Update statistics
            total_D.update(tokens_to_remove)
            total_N.update(tokens_to_add)
        
        # Update global best (union of all particles after update)
        all_tokens = set()
        for particle in self.particles.values():
            all_tokens.update(particle.tokens)
        self.global_best = all_tokens
        
        return total_D, total_N
    
    def check_noether_conservation(self, D: Set[int], N: Set[int]) -> float:
        """
        Check Noether conservation: |N| - |D| should be ~0
        
        Args:
            D: Tokens deleted
            N: Tokens added
            
        Returns:
            Conservation error (absolute value of |N| - |D|)
        """
        conservation_error = abs(len(N) - len(D))
        self.noether_violations.append(conservation_error)
        return conservation_error
    
    def manage_swarm_density(self, target_density: float = 0.3):
        """
        Manage swarm density by adding/removing particles
        
        Args:
            target_density: Target tokens/particle ratio
        """
        total_tokens = len(self.global_best)
        current_particles = len(self.particles)
        
        if current_particles == 0:
            return
        
        current_density = total_tokens / current_particles
        
        if current_density > target_density * 1.5:
            # Too dense, add more particles
            n_to_add = int(current_particles * 0.1) + 1
            self._add_particles(n_to_add)
        elif current_density < target_density / 1.5:
            # Too sparse, remove weakest particles
            n_to_remove = int(current_particles * 0.1) + 1
            self._remove_weakest_particles(n_to_remove)
    
    def _add_particles(self, n: int):
        """Add new particles to swarm"""
        for i in range(n):
            # Create new particle from random basic HLLSet
            if self.cortex.basic_hllsets:
                basic_key = random.choice(list(self.cortex.basic_hllsets.keys()))
                basic_hll = self.cortex.basic_hllsets[basic_key]
                
                new_id = f"new_particle_{self.iteration}_{i}"
                new_particle = ParticleState(
                    hllset_id=new_id,
                    tokens=basic_hll.tokens.copy(),
                    velocity=set(),
                    best_local=basic_hll.tokens.copy(),
                    age=0
                )
                self.particles[new_id] = new_particle
    
    def _remove_weakest_particles(self, n: int):
        """Remove weakest particles (oldest or smallest)"""
        if len(self.particles) <= n:
            return
        
        # Score particles by age and size
        particle_scores = []
        for particle in self.particles.values():
            # Higher score = weaker (older and smaller)
            age_score = particle.age / (self.iteration + 1)
            size_score = 1 - (len(particle.tokens) / max(1, len(self.global_best)))
            total_score = age_score * 0.7 + size_score * 0.3
            particle_scores.append((total_score, particle.hllset_id))
        
        # Remove weakest
        particle_scores.sort(reverse=True)  # Higher score first
        to_remove = [pid for _, pid in particle_scores[:n]]
        
        for pid in to_remove:
            if pid in self.particles:
                del self.particles[pid]

 # ============================================================================
# 5. SELF-GENERATION LOOP
# ============================================================================

class SGSaiSelfGenerationLoop:
    """Main self-generation loop orchestrator"""
    
    def __init__(self, 
                 token_layer: TokenLayer,
                 cortex: HLLSetCortex,
                 psm: ParticleSwarmManager,
                 horizon: int = 2):
        """
        Initialize self-generation loop
        
        Args:
            token_layer: TokenLayer instance
            cortex: HLLSetCortex instance
            psm: ParticleSwarmManager instance
            horizon: Forecasting horizon
        """
        self.token_layer = token_layer
        self.cortex = cortex
        self.psm = psm
        self.horizon = horizon
        
        # Loop state
        self.iteration = 0
        self.history = []
        
        # Control knobs
        self.swarm_density_target = 0.3
        self.forecasting_mode = True  # True = forecast, False = historical analysis
        
        # Statistics
        self.stats = {
            'noether_errors': [],
            'swarm_sizes': [],
            'cortex_sizes': [],
            'forecast_accuracies': []
        }
    
    def run_iteration(self, new_data: List[List[str]] = None):
        """
        Run one iteration of the self-generation loop
        
        Args:
            new_data: New token sequences to ingest (optional)
            
        Returns:
            Dictionary of iteration statistics
        """
        self.iteration += 1
        print(f"\n{'='*60}")
        print(f"Self-Generation Loop Iteration {self.iteration}")
        print(f"{'='*60}")
        
        # 1. INGEST: Update HRT AM with new data
        if new_data:
            print("1. Ingesting new data...")
            self.token_layer.update_hrt_am(new_data)
            self.cortex.build_basic_hllsets()  # Rebuild basic HLLSets
        
        # 2. FORECAST: Project future cortex state
        print("2. Forecasting...")
        forecast_tokens = self._forecast_cortex()
        
        # 3. OPTIMIZE: Find canonical cover
        print("3. Finding canonical cover...")
        cover_size, coverage = self.cortex.get_optimal_cover_size(forecast_tokens)
        print(f"   Cover size: {cover_size}, Coverage: {coverage:.3f}")
        
        # 4. STEER: Update particle swarm
        print("4. Steering swarm with PSM...")
        
        # Calculate D, R, N forces
        D, R, N = self._calculate_forces(forecast_tokens)
        
        # Update PSM
        self.psm.update_velocity(D, R, N)
        actual_D, actual_N = self.psm.update_positions(D, R, N)
        
        # Check Noether conservation
        noether_error = self.psm.check_noether_conservation(actual_D, actual_N)
        print(f"   Noether conservation error: {noether_error}")
        
        # Manage swarm density
        self.psm.manage_swarm_density(self.swarm_density_target)
        
        # 5. UPDATE: Update active HLLSets from particles
        print("5. Updating cortex...")
        self._update_active_hllsets_from_particles()
        
        # 6. VALIDATE: Check consistency
        print("6. Validating...")
        validation_result = self._validate_state()
        
        # Update statistics
        self._update_stats(validation_result)
        
        # Save history
        self.history.append({
            'iteration': self.iteration,
            'cortex_size': len(self.cortex.cortex_tokens),
            'swarm_size': len(self.psm.particles),
            'noether_error': noether_error,
            'cover_size': cover_size,
            'coverage': coverage
        })
        
        return self.history[-1]
    
    def _forecast_cortex(self) -> Set[int]:
        """
        Forecast cortex tokens using HRT AM
        
        Returns:
            Set of forecasted token indices
        """
        if not self.forecasting_mode or not self.cortex.cortex_tokens:
            return self.cortex.cortex_tokens.copy()
        
        # Get current cortex tokens
        current_tokens = self.cortex.cortex_tokens
        
        # One step back using transpose
        one_step_back = set()
        for token in current_tokens:
            predecessors = self.token_layer.get_predecessors(token)
            one_step_back.update(predecessors)
        
        # Two steps forward using HRT AM^2 (simplified)
        forecasted = set()
        for token in one_step_back:
            # Get tokens that follow after h steps
            followers = self._get_followers_n_steps(token, self.horizon)
            forecasted.update(followers)
        
        return forecasted if forecasted else current_tokens.copy()
    
    def _get_followers_n_steps(self, token_idx: int, steps: int) -> Set[int]:
        """
        Get tokens that follow after n steps (simplified)
        
        Args:
            token_idx: Starting token index
            steps: Number of steps
            
        Returns:
            Set of follower token indices
        """
        current = {token_idx}
        for _ in range(steps):
            next_set = set()
            for token in current:
                followers = self.token_layer.get_followers(token)
                next_set.update(followers)
            current = next_set
            if not current:
                break
        return current
    
    def _calculate_forces(self, forecast_tokens: Set[int]) -> Tuple[Set[int], Set[int], Set[int]]:
        """
        Calculate D (delete), R (retain), N (new) forces
        
        Args:
            forecast_tokens: Forecasted token set
            
        Returns:
            Tuple of (D, R, N) token sets
        """
        current_tokens = self.cortex.cortex_tokens
        
        # Tokens to delete: in current but not in forecast
        D = current_tokens - forecast_tokens
        
        # Tokens to retain: intersection
        R = current_tokens.intersection(forecast_tokens)
        
        # Tokens to add: in forecast but not current
        N = forecast_tokens - current_tokens
        
        # Apply limits
        max_change = max(50, len(current_tokens) // 10)
        if len(D) > max_change:
            D = set(random.sample(list(D), max_change))
        if len(N) > max_change:
            N = set(random.sample(list(N), max_change))
        
        return D, R, N
    
    def _update_active_hllsets_from_particles(self):
        """Update active HLLSets from particle states"""
        # Clear current active HLLSets
        self.cortex.active_hllsets.clear()
        
        # Create HLLSets from particles
        for particle_id, particle in self.psm.particles.items():
            hllset = HLLSet(
                particle.tokens.copy(),
                particle_id,
                tau=0.7,
                rho=0.3
            )
            self.cortex.active_hllsets[particle_id] = hllset
        
        # Update cortex tokens
        self.cortex._update_cortex_tokens()
    
    def _validate_state(self) -> Dict:
        """
        Validate current state
        
        Returns:
            Dictionary of validation metrics
        """
        # Check that all active HLLSets have tokens
        empty_hllsets = sum(1 for hll in self.cortex.active_hllsets.values() if not hll.tokens)
        
        # Check token overlap between HLLSets
        all_tokens = []
        for hll in self.cortex.active_hllsets.values():
            all_tokens.extend(list(hll.tokens))
        
        token_counts = Counter(all_tokens)
        avg_token_frequency = np.mean(list(token_counts.values())) if token_counts else 0
        
        return {
            'empty_hllsets': empty_hllsets,
            'avg_token_frequency': avg_token_frequency,
            'unique_tokens': len(token_counts),
            'total_tokens': len(all_tokens)
        }
    
    def _update_stats(self, validation_result: Dict):
        """Update statistics"""
        self.stats['noether_errors'].append(self.psm.noether_violations[-1] if self.psm.noether_violations else 0)
        self.stats['swarm_sizes'].append(len(self.psm.particles))
        self.stats['cortex_sizes'].append(len(self.cortex.cortex_tokens))
    
    def set_control_knob(self, knob: str, value):
        """
        Set control knob value
        
        Args:
            knob: Knob name ('density', 'mode', 'horizon')
            value: New value
        """
        if knob == 'density':
            self.swarm_density_target = max(0.1, min(1.0, value))
        elif knob == 'mode':
            self.forecasting_mode = bool(value)
        elif knob == 'horizon':
            self.horizon = max(1, int(value))
    
    def plot_statistics(self):
        """Plot loop statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        iterations = list(range(1, len(self.stats['noether_errors']) + 1))
        
        # Noether conservation error
        axes[0, 0].plot(iterations, self.stats['noether_errors'], 'r-', linewidth=2)
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 0].set_title('Noether Conservation Error')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('|N| - |D|')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Swarm size
        axes[0, 1].plot(iterations, self.stats['swarm_sizes'], 'b-', linewidth=2)
        axes[0, 1].set_title('Swarm Size (Number of HLLSets)')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cortex size
        axes[1, 0].plot(iterations, self.stats['cortex_sizes'], 'g-', linewidth=2)
        axes[1, 0].set_title('Cortex Size (Total Tokens)')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Token Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Token frequency distribution
        if self.history:
            cover_sizes = [h['cover_size'] for h in self.history]
            coverages = [h['coverage'] for h in self.history]
            
            axes[1, 1].plot(iterations, cover_sizes, 'm-', linewidth=2, label='Cover Size')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Cover Size', color='m')
            axes[1, 1].tick_params(axis='y', labelcolor='m')
            axes[1, 1].grid(True, alpha=0.3)
            
            ax2 = axes[1, 1].twinx()
            ax2.plot(iterations, coverages, 'c--', linewidth=2, label='Coverage')
            ax2.set_ylabel('Coverage', color='c')
            ax2.tick_params(axis='y', labelcolor='c')
            
            axes[1, 1].set_title('Canonical Cover Performance')
        
        plt.tight_layout()
        plt.show()               
    
