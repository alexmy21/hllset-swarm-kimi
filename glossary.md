# **Glossary of HLLSet-Swarm Framework**

**HLLSet**  
*Probabilistic set structure.*  
An `m`-bit vector where each bucket stores a max-zero count (HyperLogLog) or a full bit-vector (your implementation). Represents a token set with bounded memory and inherent ambiguity. *In code*: `torch.uint8[m]` or `torch.float16[m]` depending on density.

---

**Chinese Assembly Language (80 K Opcodes)**  
*Universal semantic base.*  
The 80,000-character instruction set that all natural languages compile through. Chosen for hierarchical radicals, compact representation, and 5,000-year evolutionary stability. *In code*: vocabulary size `V = 80_000`; each token maps to a Kangxi radical tree.

---

**τ-ρ Duality**  
*Inclusion–exclusion gates.*  
Two thresholds that validate a directed relationship:

- `τ` (inclusion tolerance): minimum overlap `|A ∩ B| / |B|` for a link to exist.  
- `ρ` (exclusion intolerance): maximum extra mass `|A \ B| / |B|` allowed.  
*In code*: `if BSS_tau >= tau and BSS_rho <= rho: edge = True`.

---

**Bell State Similarity (BSS)**  
*Directed similarity metric.*  
For HLLSets `A → B`: `BSS_τ = |A ∩ B| / |B|`, `BSS_ρ = |A \ B| / |B|`. Quantifies how much of `B` is covered by `A` versus how much `A` adds noise. *In code*: bit-vector AND → population count.

---

**Basic HLLSets**  
*Contextual primitives.*  
Two per token:

- **Row HLLSet** `R_k`: tokens that *follow* token `k` (forward context).  
- **Column HLLSet** `C_k`: tokens that *precede* token `k` (backward context).  
Form the `2V + 2` basis vectors of the semantic space. *In code*: slices of AM rows/columns.

---

**Ambiguity Resolution**  
*Consensus-driven disambiguation.*  
Two mechanisms:

- **Multi-Seed Triangulation**: intersect candidate sets from `k` independent hash seeds; accuracy ≈ 99.2 % at `k = 8`.  
- **Cohomological Disambiguation**: sheaf-theoretic filter; `H⁰` dimension predicts success (AUC = 0.96).  
*In code*: `intersection(*[hllset[s] for s in seeds])`.

---

**HLLSet Entanglement Theory**  
*Structural invariance across seeds.*  
Two swarms using different hash functions have **pairwise disjoint** HLLSets (empty intersection), yet their **concept lattices** are ε-isomorphic (relationship patterns preserved). Enables federated learning without raw-data sharing.

---

**ε-isomorphic Lattices**  
*Approximate structural identity.*  
Two lattices `L_s`, `L_s'` are ε-isomorphic if a bijection `φ` exists such that for all `A, B`:  
`|BSS(A,B) – BSS(φ(A),φ(B))| ≤ ε`.  
*In code*: `abs(bss_old - bss_new).max() < eps`.

---

**Kinematic Dynamics**  
*Time evolution of probabilistic knowledge.*  
The state transition `H(t+1) = [H(t) ⊖ D] ⊕ N`: retain core knowledge `R`, forget unused patterns `D`, add novel predictions `N`. *In code*: `r = r * (1 - lambda_forget); r[novel_idx] += novelty_boost`.

---

**Retro-Forward Duality**  
*Time-reversible propagation.*

- **Forecast**: `p→ = normalize(r · AM)`.  
- **Retrocast**: `p← = normalize(r · AMᵀ)`.  
Noether’s theorem guarantees `Φ = |N| – |D| = 0` when symmetry is preserved. *In code*: `AM_t = AM if fwd else AM.t()`.

---

**Noether Current (Φ)**  
*Conservation of token flux.*  
`Φ = |new tokens| – |forgotten tokens|`. Must remain near zero for a stable trajectory. Drift indicates hash collisions, immaturity, or numerical errors. *In code*: `phi = new_union.sum() - old_union.sum()`.

---

**Perpetual Self-Generation Loop**  
*Core four-operation cycle.*  
Cortex evolves continuously: **Learn** (ingest), **Adapt** (τ/ρ knobs), **Forget** (decay), **Forecast** (AM projection). No final optimum; only a stable trajectory.

---

**Particle Swarm Management (PSM)**  
*Swarm-as-a-single-particle abstraction.*  
Unlike classical PSO, PSM treats the whole swarm as one macro-particle whose state is `Cort(t)`. Knobs `κ, τ, ρ, λ, θ, h` steer the trajectory, not individual particles. *In code*: `tick()` updates one global `r` vector.

---

**Cortex**  
*Union of all HLLSets.*  
The semantic state of the entire system at time `t`: `Cort(t) = ⋃ X_i(t)`. Macro-state used for forecasting. *In code*: `cortex = torch.bitwise_or(*particle_hllsets)`.

---

**Swarm**  
*Partition of Cortex for parallel ingest.*  
Collection of perceptron-local HLLSets `X_i` that together form `Cort`. Enables concurrent text ingestion without lock-contention. *In code*: list of `HLLSet` objects, periodically synchronized.

---

**World of Things (WOT)**  
*Relational ontology.*  
The universal HLLSet `⊤` (top element) serves as the terminal object in the HLL category. All concepts are morphisms from `⊤`; the lattice of HLLSets maps the “world” of semantic relationships.

---

**Transfer Learning (HLLSet Context)**  
*Structural invariance across domains.*  
A model trained on Chinese text can forecast in English because the **relationship grammar** (BSS patterns) is ε-isomorphic across languages. No retraining; only τ/ρ retuning. *In code*: load `AM_zh`, freeze, adjust thresholds for `en` corpus.

---

**Sheaf Theory (Cohomology)**  
*Ambiguity quantification.*  
The cochain groups `H⁰` (consistency) and `H¹` (obstruction) measure how well local token contexts glue into a global semantic. Used for early termination of disambiguation.

---

**Lattice of HLLSets**  
*Relational map of meaning.*  
Directed graph where vertices = Basic HLLSets and edges = BSS scores. Encodes the entire semantic topology; AM is a sparse encoding of this lattice.

---

**Canonical Cover**  
*Minimal-overlap basis representation.*  
For any HLLSet `X`, the cover `Cover(X) = {B_i ∈ Basic HLLSets}` such that `X ⊆ ⋃ B_i` and overlap `ω = 1 – |⋃ B_i| / Σ|B_i|` is minimized under τ/ρ constraints. *In code*: greedy BSS-gated walk, stability-bounded.

---
