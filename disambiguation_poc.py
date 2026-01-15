"""
HyperLogLog Multi-Seed Disambiguation - SQLite Backend

Key Benefits:
- Persistent storage of TokenSpace
- Efficient queries with indexes
- Can handle large token universes
- Multiple TokenSpaces in one database
"""

import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from src.hllset_swarm.hll import HLL, AddResult


@dataclass
class DisambiguationResult:
    """Results from multi-seed disambiguation"""
    original_tokens: Set[str]
    candidate_tokens_per_seed: Dict[int, Set[str]]
    intersection_tokens: Set[str]
    precision: float
    recall: float
    false_positives: Set[str]
    false_negatives: Set[str]


class TokenSpaceDB:
    """
    SQLite-backed TokenSpace storage
    
    Schema:
    - token_spaces: metadata about each seed
    - token_entries: (space_id, token, hash, register, leading_zeros)
    
    Indexed by (space_id, register) for fast filtering
    """
    
    def __init__(self, db_path: str = ":memory:", debug: bool = False):
        """
        Initialize TokenSpace database
        
        Args:
            db_path: Path to SQLite database file (or ":memory:" for in-memory)
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._create_schema()
        self.debug = debug
    
    def _create_schema(self):
        """Create database schema with indexes"""
        cursor = self.conn.cursor()
        
        # TokenSpace metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_spaces (
                space_id INTEGER PRIMARY KEY AUTOINCREMENT,
                seed INTEGER NOT NULL,
                num_entries INTEGER NOT NULL,
                num_registers INTEGER NOT NULL
            )
        """)
        
        # Token entries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_entries (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                space_id INTEGER NOT NULL,
                token TEXT NOT NULL,
                hash_value INTEGER NOT NULL,
                register INTEGER NOT NULL,
                leading_zeros INTEGER NOT NULL,
                FOREIGN KEY (space_id) REFERENCES token_spaces(space_id)
            )
        """)
        
        # Critical index for fast filtering by (space_id, register)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_space_register 
            ON token_entries(space_id, register)
        """)
        
        # Index for token lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_space_token 
            ON token_entries(space_id, token)
        """)
        
        self.conn.commit()
    
    def create_token_space(self, 
                          tokens: List[str], 
                          seed: int,
                          precision: int = 10) -> int:
        """
        Create a TokenSpace by hashing tokens with given seed
        
        Args:
            tokens: List of tokens
            seed: Hash seed
            precision: HLL precision
            
        Returns:
            space_id: Database ID for this TokenSpace
        """
        cursor = self.conn.cursor()
        
        # Check if space already exists
        cursor.execute("SELECT space_id FROM token_spaces WHERE seed = ?", (seed,))
        existing = cursor.fetchone()
        if existing:
            print(f"TokenSpace for seed {seed} already exists (space_id={existing['space_id']})")
            return existing['space_id']
        
        # Hash all tokens
        hll = HLL(P_BITS=precision)
        results = hll.add(tokens, seed=seed)
        print(f"Hashed {results}")
        
        if not results:
            raise ValueError("No results from HLL.add()")
        
        # Count unique registers
        registers_used = len(set(r.register for r in results))
        
        # Insert TokenSpace metadata
        cursor.execute("""
            INSERT INTO token_spaces (seed, num_entries, num_registers)
            VALUES (?, ?, ?)
        """, (seed, len(results), registers_used))
        
        space_id = cursor.lastrowid
        
        # Batch insert token entries
        entries = [
            (space_id, r.token, r.hash_value, r.register, r.leading_zeros)
            for r in results
        ]
        
        cursor.executemany("""
            INSERT INTO token_entries (space_id, token, hash_value, register, leading_zeros)
            VALUES (?, ?, ?, ?, ?)
        """, entries)
        
        self.conn.commit()
        
        print(f"Created TokenSpace: space_id={space_id}, seed={seed}, entries={len(results)}")
        
        return space_id
    
    def get_space_id_by_seed(self, seed: int) -> Optional[int]:
        """Get space_id for a given seed"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT space_id FROM token_spaces WHERE seed = ?", (seed,))
        row = cursor.fetchone()
        return row['space_id'] if row else None
    
    def get_space_info(self, space_id: int) -> Dict:
        """Get metadata about a TokenSpace"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT seed, num_entries, num_registers
            FROM token_spaces
            WHERE space_id = ?
        """, (space_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return dict(row)
    
    def filter_by_hll_counts(self, 
                            space_id: int, 
                            hll_counts: List[int]) -> Set[str]:
        """
        Get candidate tokens matching HLL counts bitmap
        
        Optimized query:
        1. For each register with non-zero bitmap
        2. Query token_entries WHERE space_id=X AND register=R
        3. Filter by bitwise match in Python (fast)
        
        Args:
            space_id: TokenSpace ID
            hll_counts: HLL counts vector
            
        Returns:
            Set of candidate tokens
        """
        cursor = self.conn.cursor()
        candidates = set()
        
        for _register, bitmap in enumerate(hll_counts):
            if bitmap == 0:
                continue
            
            register = _register + 1  # Registers are 1-indexed

            if self.debug:
                print(f"Filtering register {register} with bitmap {bitmap}, {bin(bitmap)}")
                
            cursor.execute("""
                SELECT token, leading_zeros
                FROM token_entries
                WHERE space_id = ? AND register = ?
            """, (space_id, register))  
            
            for row in cursor:
                leading_zeros = row['leading_zeros']
                bit_position = leading_zeros - 1
                bit_mask = 1 << bit_position
                
                match = bitmap & bit_mask
                
                if self.debug and register < 5:  # Debug first 5 registers
                    print(f"  Register {register}: bitmap={bin(bitmap)}, "
                        f"token={row['token']}, zeros={leading_zeros}, "
                        f"mask={bin(bit_mask)}, match={match!=0}")
                
                if match:
                    candidates.add(row['token'])
        
        return candidates
    
    def get_all_tokens(self, space_id: int) -> Set[str]:
        """Get all unique tokens in a TokenSpace"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT token
            FROM token_entries
            WHERE space_id = ?
        """, (space_id,))
        
        return {row['token'] for row in cursor}
    
    def get_register_distribution(self, space_id: int) -> Dict[int, int]:
        """Get distribution of tokens per register"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT register, COUNT(*) as count
            FROM token_entries
            WHERE space_id = ?
            GROUP BY register
            ORDER BY register
        """, (space_id,))
        
        return {row['register']: row['count'] for row in cursor}
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MultiSeedDisambiguator:
    """Disambiguate original tokens using multiple HLL seeds with SQLite backend"""
    
    def __init__(self, precision: int = 10, db_path: str = ":memory:", debug: bool = False):
        
        self.precision = precision
        self.db = TokenSpaceDB(db_path, debug=debug)
        self.debug = debug
    
    def disambiguate(self,
                    original_tokens: List[str],
                    token_universe: List[str],
                    seeds: List[int]) -> DisambiguationResult:
        """
        Perform multi-seed disambiguation using SQLite backend
        
        Args:
            original_tokens: The actual tokens (unknown in real scenario)
            token_universe: The complete space of all possible tokens
            seeds: List of seeds to use
            
        Returns:
            DisambiguationResult with analysis
        """
        print(f"\n{'='*80}")
        print(f"Multi-Seed Disambiguation (SQLite Backend)")
        print(f"{'='*80}")
        print(f"Original tokens (to recover): {len(original_tokens)}")
        print(f"Token universe (search space): {len(token_universe)}")
        print(f"Number of seeds: {len(seeds)}")
        print(f"Database: {self.db.db_path}")
        print()
        
        # Step 1: Create HLLs from original tokens
        print("Step 1: Creating HLLs from original tokens...")
        hlls = {}
        
        for seed in seeds:
            hll = HLL(P_BITS=self.precision)
            hll.add(original_tokens, seed=seed)
            hlls[seed] = hll
            if self.debug:
                print(f"  Seed {seed}: cardinality {hll.count():.0f}")
        
        # Step 2: Create TokenSpaces in database
        print("\nStep 2: Creating TokenSpaces in database...")
        space_ids = {}
        
        for seed in seeds:
            space_id = self.db.create_token_space(token_universe, seed, self.precision)
            space_ids[seed] = space_id
            
            info = self.db.get_space_info(space_id)
            print(f"  Seed {seed}: space_id={space_id}, "
                  f"entries={info['num_entries']}, registers={info['num_registers']}")
        
        # Step 3: Match TokenSpace against HLL counts
        print("\nStep 3: Querying candidates from database...")
        candidate_sets = {}
        
        for seed in seeds:
            hll = hlls[seed]
            space_id = space_ids[seed]
            
            # Get HLL counts
            hll_counts = hll.dump()
            print(f"  Seed {seed}: HLL counts extracted {len(hll_counts)} registers")
            
            # Query matching candidates from database
            candidates = self.db.filter_by_hll_counts(space_id, hll_counts)
            candidate_sets[seed] = candidates
            
            print(f"  Seed {seed}: {len(candidates)} candidates matched")
        
        # Step 4: Intersect candidates
        print("\nStep 4: Computing intersection across all seeds...")
        
        if len(candidate_sets) == 0:
            intersection = set()
        else:
            intersection = set.intersection(*candidate_sets.values())
        
        original_set = set(original_tokens)
        
        # Analysis
        true_positives = intersection & original_set
        false_positives = intersection - original_set
        false_negatives = original_set - intersection
        
        precision = len(true_positives) / len(intersection) if len(intersection) > 0 else 0.0
        recall = len(true_positives) / len(original_set) if len(original_set) > 0 else 0.0
        
        print(f"\nResults:")
        print(f"  Intersection size: {len(intersection)}")
        print(f"  True positives: {len(true_positives)}")
        print(f"  False positives: {len(false_positives)}")
        print(f"  False negatives: {len(false_negatives)}")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall: {recall*100:.2f}%")
        
        return DisambiguationResult(
            original_tokens=original_set,
            candidate_tokens_per_seed=candidate_sets,
            intersection_tokens=intersection,
            precision=precision,
            recall=recall,
            false_positives=false_positives,
            false_negatives=false_negatives
        )
    
    def close(self):
        """Close database connection"""
        self.db.close()

# Visualization function

def visualize_venn_diagram(result: DisambiguationResult, save_path: str = None):
    """
    Visualize overlap between original tokens and reconstructed tokens
    
    Args:
        result: DisambiguationResult
        save_path: Path to save figure
    """
    try:
        from matplotlib_venn import venn2
    except ImportError:
        print("matplotlib-venn not installed. Run: pip install matplotlib-venn")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Calculate set sizes
    original = result.original_tokens
    reconstructed = result.intersection_tokens
    
    # Create Venn diagram
    venn = venn2(
        [original, reconstructed],
        set_labels=('Original Tokens', 'Reconstructed Tokens'),
        ax=ax
    )
    
    # Customize colors
    venn.get_patch_by_id('10').set_color('lightcoral')  # Only in original
    venn.get_patch_by_id('01').set_color('lightblue')   # Only in reconstructed
    venn.get_patch_by_id('11').set_color('lightgreen')  # Intersection
    
    # Add labels with counts
    venn.get_label_by_id('10').set_text(f'False Negatives\n{len(result.false_negatives)}')
    venn.get_label_by_id('01').set_text(f'False Positives\n{len(result.false_positives)}')
    venn.get_label_by_id('11').set_text(f'True Positives\n{len(result.original_tokens & reconstructed)}')
    
    # Title with metrics
    ax.set_title(
        f'Token Reconstruction Venn Diagram\n'
        f'Precision: {result.precision*100:.2f}% | Recall: {result.recall*100:.2f}%',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_seed_contribution(results_by_num_seeds: List[Tuple[int, DisambiguationResult]], 
                                save_path: str = None):
    """
    Visualize how each additional seed reduces candidate space
    
    Args:
        results_by_num_seeds: List of (num_seeds, result) tuples
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    num_seeds_list = [x[0] for x in results_by_num_seeds]
    intersection_sizes = [len(x[1].intersection_tokens) for x in results_by_num_seeds]
    
    # Calculate reduction per seed
    reductions = [0]  # First seed has no reduction
    for i in range(1, len(intersection_sizes)):
        reduction = intersection_sizes[i-1] - intersection_sizes[i]
        reductions.append(reduction)
    
    # Plot 1: Absolute candidate size
    ax1 = axes[0]
    bars = ax1.bar(num_seeds_list, intersection_sizes, color='skyblue', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Number of Seeds', fontsize=12)
    ax1.set_ylabel('Candidate Space Size', fontsize=12)
    ax1.set_title('Candidate Space vs Number of Seeds', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Incremental reduction
    ax2 = axes[1]
    bars = ax2.bar(num_seeds_list, reductions, color='coral', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Number of Seeds', fontsize=12)
    ax2.set_ylabel('Tokens Eliminated', fontsize=12)
    ax2.set_title('Incremental Reduction per Additional Seed', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_register_distribution(db: TokenSpaceDB, space_id: int, save_path: str = None):
    """
    Visualize how tokens are distributed across registers
    
    Args:
        db: TokenSpaceDB instance
        space_id: TokenSpace ID
        save_path: Path to save figure
    """
    # Get register distribution
    reg_dist = db.get_register_distribution(space_id)
    
    if not reg_dist:
        print("No data for this space_id")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    registers = sorted(reg_dist.keys())
    counts = [reg_dist[r] for r in registers]
    
    # Plot 1: Bar chart
    ax1 = axes[0]
    ax1.bar(registers, counts, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Register Index', fontsize=12)
    ax1.set_ylabel('Number of Tokens', fontsize=12)
    ax1.set_title('Token Distribution Across Registers', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Histogram of counts
    ax2 = axes[1]
    ax2.hist(counts, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Tokens per Register', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Token Counts per Register', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    info = db.get_space_info(space_id)
    stats_text = (f"Total Registers: {len(reg_dist)}\n"
                 f"Total Entries: {info['num_entries']}\n"
                 f"Avg per Register: {np.mean(counts):.2f}\n"
                 f"Max per Register: {max(counts)}")
    
    ax2.text(0.95, 0.95, stats_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_pattern_complexity(db: TokenSpaceDB, space_ids: List[int], seeds: List[int], 
                                 save_path: str = None):
    """
    Compare pattern complexity across different seeds
    
    Args:
        db: TokenSpaceDB instance
        space_ids: List of space IDs
        seeds: Corresponding seed values
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    data = []
    for space_id, seed in zip(space_ids, seeds):
        reg_dist = db.get_register_distribution(space_id)
        data.append({
            'seed': seed,
            'registers_used': len(reg_dist),
            'total_entries': sum(reg_dist.values()),
            'avg_per_register': np.mean(list(reg_dist.values()))
        })
    
    # Create grouped bar chart
    x = np.arange(len(seeds))
    width = 0.25
    
    ax.bar(x - width, [d['registers_used'] for d in data], width, 
           label='Registers Used', color='skyblue', edgecolor='black')
    ax.bar(x, [d['total_entries']/10 for d in data], width,
           label='Total Entries / 10', color='lightcoral', edgecolor='black')
    ax.bar(x + width, [d['avg_per_register'] for d in data], width,
           label='Avg per Register', color='lightgreen', edgecolor='black')
    
    ax.set_xlabel('Seed', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('TokenSpace Pattern Complexity Across Seeds', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Seed {s}' for s in seeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_disambiguation_convergence(results_by_num_seeds: List[Tuple[int, DisambiguationResult]],
                                        save_path: str = None):
    """Visualize how disambiguation improves with more seeds"""
    
    num_seeds_list = [x[0] for x in results_by_num_seeds]
    precisions = [x[1].precision * 100 for x in results_by_num_seeds]
    recalls = [x[1].recall * 100 for x in results_by_num_seeds]
    intersection_sizes = [len(x[1].intersection_tokens) for x in results_by_num_seeds]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Precision and Recall
    ax1 = axes[0]
    ax1.plot(num_seeds_list, precisions, 'o-', linewidth=2, markersize=8, label='Precision', color='blue')
    ax1.plot(num_seeds_list, recalls, 's-', linewidth=2, markersize=8, label='Recall', color='green')
    ax1.set_xlabel('Number of Seeds', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Precision & Recall vs Number of Seeds', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Intersection Size
    ax2 = axes[1]
    ax2.plot(num_seeds_list, intersection_sizes, 'o-', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Number of Seeds', fontsize=12)
    ax2.set_ylabel('Intersection Size', fontsize=12)
    ax2.set_title('Candidate Intersection Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # F1 Score
    ax3 = axes[2]
    f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 
                for p, r in zip([x/100 for x in precisions], [x/100 for x in recalls])]
    ax3.plot(num_seeds_list, [f*100 for f in f1_scores], 'o-', linewidth=2, markersize=8, color='orange')
    ax3.set_xlabel('Number of Seeds', fontsize=12)
    ax3.set_ylabel('F1 Score (%)', fontsize=12)
    ax3.set_title('F1 Score vs Number of Seeds', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def run_disambiguation_poc():
    """Run complete disambiguation proof of concept with SQLite"""
    
    print("="*80)
    print("HyperLogLog Multi-Seed Disambiguation POC (SQLite Backend)")
    print("="*80)
    print()
    
    # Create datasets
    np.random.seed(42)
    token_universe = [f"token_{i:04d}" for i in range(500)]
    original_tokens = token_universe[:50]
    
    print(f"Token universe: {len(token_universe)} tokens")
    print(f"Original tokens (to recover): {len(original_tokens)} tokens")
    print()
    
    # Use persistent database
    db_path = "token_spaces.db"
    
    # Test with increasing number of seeds
    seed_configs = [
        [0],
        [0, 42],
        [0, 42, 123],
        [0, 42, 123, 456],
    ]
    
    disambiguator = MultiSeedDisambiguator(precision=10, db_path=db_path)
    results = []
    
    try:
        for seeds in seed_configs:
            result = disambiguator.disambiguate(
                original_tokens=original_tokens,
                token_universe=token_universe,
                seeds=seeds
            )
            results.append((len(seeds), result))
            print()
        
        # Visualizations
        print("\n" + "="*80)
        print("Generating visualizations...")
        print("="*80)
        
        visualize_disambiguation_convergence(results, save_path='disambiguation_convergence.png')
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print()
        
        for num_seeds, result in results:
            print(f"  {num_seeds} seed(s):")
            print(f"    - Candidates: {len(result.intersection_tokens)}")
            print(f"    - Precision: {result.precision*100:.2f}%")
            print(f"    - Recall: {result.recall*100:.2f}%")
            print()
        
        print(f"✓ TokenSpaces persisted in: {db_path}")
        print("✓ Efficient SQL queries with indexes")
        print("✓ Can reuse TokenSpaces across runs")
        print("✓ Scales to large token universes!")
        print()
        
    finally:
        disambiguator.close()


if __name__ == "__main__":
    run_disambiguation_poc()