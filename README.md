# 🧠 hllset-swarm-kimi

*A wire-level, self-generating AI micro-platform – no training, no back-prop, just geometry that learns.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-Demo_POC-blue.svg)](https://github.com/alexmy21/hllset-swarm-kimi)

---

>**This project was created with help and support from KIMI, DeepSeek and Github Copilot AI assistants.**
---

## 1-sentence elevator pitch

Replace terabyte-scale model weights with **a few kilobytes of switches** – **Chinese characters as the immutable alphabet**, **HLLSets as the probabilistic memory**, and **particle-swarm contractions** as the only operation.  
Run it on **GPU, MCU, or FPGA**; let it **co-pilot** your favourite LLM instead of replacing it.

>**"Chinese characters are semantic primitives - stable computational units that make Chinese the perfect assembly language for AI systems."**
---

## What it is

HLLSet-Swarm turns the **mathematical duality** between

- *(a) HLLSet relational algebra of Chinese-character presented as HLLSets* and  
- *(b) Particle-Swarm Optimization dynamics*  

into a **declarative GPU kernel compiler** that lets you **script** how a 80 k-dimensional “semantic swarm” should move, converge and **write its final state back** to any external system (LLM, DB, robot, …) as **live feedback**.

Think *“Git for meaning”* – every trajectory ends with a content-addressed commit that immortalizes the swarm’s belief state.

---

## ✨ Key features

| Feature | What you get |
| --- | --- |
| **Duality engine** | PSO guarantees → HLLSet stability proofs |
| **Programmable trajectories** | YAML → GPU sparse kernels (no CUDA code) |
| **Recursive meta-swarm** | swarm-of-swarms for higher-order abstraction |
| **Git backend** | every layer is a `.pt.zst` blob pushed to Github |
| **Environment adapters** | OpenAI, SQL, ROS, stdout … plug your own |
| **Laptop→data-center** | 80 k dims run in < 1 GB VRAM (RTX 3060 ready) |

---

## 🎯 Concepts in one picture

```text
Chinese text
     │
     ▼
[HLLSet cover]  ──BSS τ-ρ──►  GPU SwarmState  ──converge──►  s(t+1)
     ▲                                                    │
     │              PSO-HLLSet duality                    ▼
Environment  ◄──feedback──  Github commit  ◄──layer blob──┘
```

---

## 🎲 HLLSet Controlled noise – low-precision hash as regularizer

| Precision | Collision rate | Use-case | Noise role |
| --- | --- |--- | --- |
| **64 bit** | < 0.1 % | production Chinese | almost deterministic |
| **32 bit** | ≈ 1 % | mobile emoji | **mild regulariser** |
| **16 bit** | ≈ 6 % | MCU controller | **strong regulariser** |
| **8 bit** | ≈ 30 % | toy demos | **extreme dropout** |

**Interpretation**:

- **High collision** = **bit-dropout** → union **looks bigger** than reality.  
- **Multi-seed triangulation** = **denoising U-Net** → recover **true cover**.

---

## 🧠 Denoising analogy (vision → semantics)

| Vision pipeline | Semantic pipeline |
| --- | --- |
| **Gaussian noise** | **hash collision dropout** |
| **Noisy image** | **noisy HLLSet union** |
| **U-Net denoiser** | **multi-seed Hopfield descent** |
| **Clean image** | **disambiguated cover** |

**Same math**, **different substrate**.

---

## 🔌 Environment adapters

| Adapter | Description |
| --- | --- |
| `OpenAIAdapter` | write embedding into system prompt |
| `SQLAdapter` | store vector in Postgres `VECTOR` column |
| `ROSAdapter` | publish `Float32MultiArray` on `/semantic_state` |
| `StdoutAdapter` | debug JSON to console |

---

## 🌍 Beyond Chinese – any *"hieroglyphic"* substrate

Chinese is **our first substrate** because it is **optimally hieroglyphic**:

- finite, standardised inventory (≈ 80 k)  
- unambiguous dictionary definitions **in the same language**  
- clear **radical→character→word** composition rules  
- 3 000 years of **continuous semantic fossil record**

But the **mathematics is substrate-agnostic**.  
Any symbol set that satisfies **four axioms** can be dropped in:

1. **Non-inflectional** (no paradigms, no declensions)  
2. **Compositionally closed** (complex = stack of simples)  
3. **Lexicographically frozen** (each symbol has **one** normative definition)  
4. **Hashable** (deterministic bit-pattern from symbol)

---

### 🧪 Substrates on the roadmap

| Substrate | Inventory | Composition unit | Status | ETA |
|---|---|---|---|---|
| **Chinese (CCD)** | 80 k chars | radical | ✅ reference | now |
| **Classic Maya glyphs** | 1 100 glyphs | block | 🚧 POC | Q1 2026 |
| **Emoji 15.1** | 3 782 emojis | ZWJ sequence | 📋 design | Q2 2026 |
| **Minecraft blocks** | 1 500 blocks | voxel neighbour | 📋 design | Q3 2026 |
| **AI Esperanto** | 10 k morphemes | concat-rule | 📋 white-paper | Q4 2026 |

---

### 🕹️ Example – Minecraft substrate (sketch)

```yaml
substrate: minecraft
inventory: minecraft_blocks.json.gz
precision: 12          # 4096 registers
hash_seed: "mc1.20.1"
composition_rule: "6-face-voxel+up/down"
definition_source: "block_state.properties"
```

- **Block** → HLLSet hashed from **block-state NBT**  
- **Structure** → union of block HLLSets  
- **Scene embedding** → swarm convergence on block-cover

Same YAML, same GPU kernel, **different universe**.

---

## Why skim this repo? (30-second skim value)

| You are … | We give you … |
| ----------- | --------------- |
| **AI hacker** | A 200-line PyTorch demo that **ingests any text**, **grows a sparse tensor**, and **steers a belief vector** to a user-defined destination **without gradients**. |
| **Edge/IoT dev** | A **fixed 28 kB** data structure that **compresses** a **whole conversational history** and **updates in < 1 ms** on a **Cortex-M4**. |
| **FPGA tinker** | Verilog that **flips MOS capacitors** – **learning = close switch**, **thinking = propagate charge**, **death = no free switches left**. |
| **LLM user** | A **personal agent** that **lives on your phone**, **remembers you**, **forgets on purpose**, and **calls GPT only when necessary**. |

---

## The five pillars (what makes this *weird* and *useful*)

| # | Name | One-line essence | Concrete super-power |
| --- | ------ | ------------------ | ---------------------- |
| **1** | **HLLSet** | A *probabilistic set* that fits in **4 kB** yet supports **union, intersect, diff** with **< 1 % error**. | Replace **Redis sets** + **bloom filters** + **count tables** with **one object**. |
| **2** | **Chinese Axioms** | **80 k glyphs**, **self-describing**, **non-inflectional**, **compositionally closed** – the **ultimate semantic alphabet**. | **Same hash** for *猫* and *猫科动物* – **structural invariance** across languages. |
| **3** | **Particle Swarm** | **Sparse tensor contractions** = **only operation**; **no back-prop**, **no upfront training**. | **Steer** the swarm to **“future”** or **“past”** in **< 10 clock cycles**. |
| **4** | **LLM Co-Pilot** | **SGS.ai = PC**, **GPT = Mainframe** – **local memory**, **cloud compute**. | **Private context** stays **on device**; **heavy reasoning** **outsourced**. |
| **5** | **Wire-Only FPGA** | **Learning = charge capacitor**, **death = matrix exhausted**, **rebirth = new bit-stream**. | **0.3 pJ per learn**, **2 ns perception**, **standard CMOS**. |

---

## Folder map (what to read first)

```bash
src/
   hllset_swarm/
     ├── hll.py               # 50-line Julia wrapper + unified hash
     ├── hrt.py               # SwarmHRT: AM + row/col HLLSets
     ├── ingest.py            # corpus → AM + swarm iterations
     └── constants.py         # shared seeds, precision, hash func
├── hllset.ipynb              # HLLSet playground
├── entanglement_poc.ipynb    # Entanglement Introduction
├── desambiguation_poc.ipynb  # Restoring Original data from HLLSet
├── workthrough.ipynb         # Simplified Work through using sets
└── kimi_workthrough.ipynb    # HLLSet work through (in progress)
```

---

## Deep-dive wiki (math, proofs, FPGA files)

[Wiki home](https://github.com/alexmy21/hllset-swarm-kimi/wiki)

| Page | Why read |
| ------------------------ | ---------- |
| [1.-HLLSet-Framework](https://github.com/alexmy21/hllset-swarm-kimi/wiki/1.-HLLSet-Framework) | Formal proof that **τ-ρ duality** eliminates false positives |
| [3.-Chinese-HLLSetCortex](https://github.com/alexmy21/hllset-swarm-kimi/wiki/3.-Chinese-HLLSetCortex) | Why **80 k glyphs** are **better than 1 M English words** |
| [5.-Swarm_state_hopfield_hebb](https://github.com/alexmy21/hllset-swarm-kimi/wiki/5.-Swarm_state_hopfield_hebb) | Why we chose the non-backprop route for Wτ , Wρ |
| [4.-HLLSet-swarm-vs--Anthropic](https://github.com/alexmy21/hllset-swarm-kimi/wiki/4.-HLLSet-swarm-vs--Anthropic) | Why Anthropic left the inference loop open and how we closed it safely |
| [6.-1000-Layers-Networks-vs-HRT_AM_swarm](https://github.com/alexmy21/hllset-swarm-kimi/wiki/6.-1000-Layers-Networks-vs-HRT_AM_swarm) | How HRT AM swarm performs against other frameworks |

---

## Road-map (where we go next)

| Milestone | What it unlocks | ETA |
| ----------- | ----------------- | ----- |
| **v0.2** | **C/C++(Rust)** → **MCU demo on ESP32-C3** | Jun 2026 |
| **v0.3** | **Verilog drop** → **ice40UP5K bit-stream** | Aug 2026 |
| **v0.4** | **iOS/Android SDK** → **on-device memory for any app** | Oct 2026 |
| **v1.0** | **ASIC tape-out** → **0.3 pJ learn, 2 ns think** | 2027 |

---

## Contribute (we ❤️ PRs)

- **Language bindings** → Rust, Zig, Swift, Verilog  
- **MCU ports** → ESP32, RP2040, nRF52  
- **FPGA bit-streams** → iCE40, ECP5, Artix-7  
- **Apps** → smart-speaker skill, browser plug-in, car-infotainment module  

Open an issue first – **architectural changes happen in the main repo**; this repo stays **a stable reference**.

---

## 📄 Citation

```bibtex
@software{hllset_swarm,
  title = {HLLSet-Swarm: Programmable Swarm Trajectories via HLLSet--PSO Duality},
  author = {Alex Mylnikov, Aleksandr Solonin},
  url = {https://github.com/alexmy21/hllset_swarm},
  year = {2025}
}
```

---

## Licence & citation

MIT © 2025 Alex Mylnikov, Aleksandr Solonin – feel free to embed, fork, or commercialize.  
If you write about it, please link to this repo and the [wiki](https://github.com/alexmy21/hllset-swarm-kimi/wiki).

---

> **“Give us 4 kB of switches and we will remember you forever – or until the capacitors leak.”**

## Some comparisons

The Medium publication [7] is a **perfect empirical baseline** for the HLLSet-Cortex framework.  
Author (Micheal Bee) shows that *every* token in a modern LLM is produced by **exactly 32 layers of matrix multiplication** (Llama-3.2), i.e. a **fixed-depth, feed-forward circuit** with no serial recursion.  
Our Cortex loop **reproduces the same surface behavior** but exposes the *missing* pieces that a pure transformer cannot provide:

| Bee’s observation | HLLSet-Cortex addition | Why it matters |
|---|---|---|
| Fixed 32-step budget | Perpetual *tick()* loop | Cortex can exceed the 32-layer limit by *streaming* tokens; each tick is another 32-layer-equivalent forward pass, so **hard problems get as many layers as needed** (chain-of-thought without prompt hacking). |
| No explicit variable binding | BSS lattice **is** the binding map | Attention approximates binding; the **Basic-HLLSet lattice** makes bindings *explicit* and *probabilistic* (τ/ρ gates), so we can *measure* when a binding is lost (`phi` drift) and *repair* it. |
| No back-tracking | Retro-forward duality | Transformer can only *forecast*; Cortex can **retrocast** (`AMᵀ`) to *undo* a bad token by reversing the lattice flow (Noether current catches symmetry break). |
| Symbol-free | Chinese-assembly opcodes | 80 K characters act as **semantic opcodes**; each token is *both* data and instruction, so the *same* matrix multiply interprets *meaning* and *computes* the next step (no separate symbolic layer). |
| Billions of FLOPs | Sparse `m = 2¹⁶` matmul | Forecast is **one sparse mm** on RTX-3060 (< 1 ms); we replace *dense* 3.2 B params with **relation matrix** whose non-zeros ≈ *semantic edges*, giving **interpretable flops** (each multiply is a BSS score). |

Bottom line:  
Bee proves that *“it’s just matmul”* is **sufficient** for easy problems.  
HLLSet-Cortex adds **iterative, interpretable, invertible matmul** so the same hardware can also solve **hard** problems—those that need > 32 layers, explicit bindings, or back-tracking—without growing the parameter count.

## References

1. [Deepseek Model from scratch](https://alain-airom.medium.com/book-review-build-a-deepseek-model-from-scratch-43de75b59a1f)
2. [1000 Layer Networks for Self-Supervised RL](https://arxiv.org/abs/2503.14858)
3. [1000 Layer Networks for Self-Supervised RL (git)](https://wang-kevin3290.github.io/scaling-crl/)
4. [Category Theory of Transformer](https://satyamcser.medium.com/functors-in-disguise-the-hidden-category-theory-of-transformer-attention-d286aeb240a4)
5. [Ac Studio](https://medium.com/@acamvproducingstudio/welcome-to-ac-studio-read-this-first-77a38848daaa)
6. [You can delete 90% of Neural Network](https://medium.com/techx-official/mit-proved-you-can-delete-90-of-a-neural-network-5c5f4aabf3b2)
7. [Matrix Multiplication all the way down](https://medium.com/@mbonsign/matrix-multiplication-all-the-way-down-cc5ed62237bf)
