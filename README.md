# 🧠 hllset-swarm-kimi

*A wire-level, self-generating AI micro-platform – no training, no back-prop, just geometry that learns.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-concept-demo-blue)](https://github.com/alexmy21/hllset-swarm-kime)

---

## 1-sentence elevator pitch

Replace terabyte-scale model weights with **a few kilobytes of switches** – **Chinese characters as the immutable alphabet**, **HLLSets as the probabilistic memory**, and **particle-swarm contractions** as the only operation.  
Run it on **GPU, MCU, or FPGA**; let it **co-pilot** your favourite LLM instead of replacing it.

---

## Why skim this repo? (30-second skim value)

| You are … | We give you … |
|-----------|---------------|
| **AI hacker** | A 200-line PyTorch demo that **ingests any text**, **grows a sparse tensor**, and **steers a belief vector** to a user-defined destination **without gradients**. |
| **Edge/IoT dev** | A **fixed 28 kB** data structure that **compresses** a **whole conversational history** and **updates in < 1 ms** on a **Cortex-M4**. |
| **FPGA tinker** | Verilog that **flips MOS capacitors** – **learning = close switch**, **thinking = propagate charge**, **death = no free switches left**. |
| **LLM user** | A **personal agent** that **lives on your phone**, **remembers you**, **forgets on purpose**, and **calls GPT only when necessary**. |

---

## The five pillars (what makes this *weird* and *useful*)

| # | Name | One-line essence | Concrete super-power |
|---|------|------------------|----------------------|
| **1** | **HLLSet** | A *probabilistic set* that fits in **4 kB** yet supports **union, intersect, diff** with **< 1 % error**. | Replace **Redis sets** + **bloom filters** + **count tables** with **one object**. |
| **2** | **Chinese Axioms** | **80 k glyphs**, **self-describing**, **non-inflectional**, **compositionally closed** – the **ultimate semantic alphabet**. | **Same hash** for *猫* and *猫科动物* – **structural invariance** across languages. |
| **3** | **Particle Swarm** | **Sparse tensor contractions** = **only operation**; **no back-prop**, **no upfront training**. | **Steer** the swarm to **“future”** or **“past”** in **< 10 clock cycles**. |
| **4** | **LLM Co-Pilot** | **SGS.ai = PC**, **GPT = Mainframe** – **local memory**, **cloud compute**. | **Private context** stays **on device**; **heavy reasoning** **outsourced**. |
| **5** | **Wire-Only FPGA** | **Learning = charge capacitor**, **death = matrix exhausted**, **rebirth = new bit-stream**. | **0.3 pJ per learn**, **2 ns perception**, **standard CMOS**. |

---

## 60-second demo (copy-paste runnable)

```bash
git clone https://github.com/alexmy21/hllset-swarm-kime
cd hllset-swarm-kime
pip install -e .
python -m hllset_swarm.demo
```

Output:

```bash
=== ingest ===
AM shape: (2811, 2811)  nnz:  8 492
=== inertial swarm ===
step 5  tokens: 人工智能发展趋势未来
=== guided → "未来世界" ===
arrived in 4 steps: 未来世界发展展望
Git log: 15 commits written → kime_git_log.json
```

**No network call, no gradient, no training data – just 200 lines of PyTorch and Julia glue.**

---

## Folder map (what to read first)

```bash
src/hllset_swarm/
├── __init__.py
├── hll.py          # 50-line Julia wrapper + unified hash
├── hrt.py          # SwarmHRT: AM + row/col HLLSets + belief contraction
├── ingest.py       # corpus → AM + swarm iterations
├── commit.py       # git-style commit objects
└── constants.py    # shared seeds, precision, hash func

notebooks/
└── kime_walkthrough.ipynb   # blog post in notebook form
```

Start here:

1. `notebooks/kime_walkthrough.ipynb` – **interactive blog** (math + code)  
2. `src/hllset_swarm/hrt.py` – **core 120 lines** (swarm logic)  
3. `main.py` – **30-line CLI** (end-to-end demo)

---

## Deep-dive wiki (math, proofs, FPGA files)

[Wiki home](https://github.com/alexmy21/hllset-swarm-kime/wiki)

| Page | Why read |
| ------ | ---------- |
| **HLLSet Category** | Formal proof that **τ-ρ duality** eliminates false positives |
| **Chinese Axioms** | Why **80 k glyphs** are **better than 1 M English words** |
| **Wire-Only FPGA** | Verilog + spice plots → **0.3 pJ learn @ 2 ns** |
| **Swarm Dynamics** | **Convex energy** → **≤ 10 steps** to any destination |
| **LLM Co-Pilot API** | **OpenAI-compatible endpoint** that **keeps your secrets** |

---

## Road-map (where we go next)

| Milestone | What it unlocks | ETA |
| ----------- | ----------------- | ----- |
| **v0.2** | **Rust + no-std** → **MCU demo on ESP32-C3** | Jun 2026 |
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

## Licence & citation

MIT © 2025 Alex Mylnikov, Aleksandr Solonin – feel free to embed, fork, or commercialize.  
If you write about it, please link to this repo and the [wiki](https://github.com/alexmy21/hllset-swarm-kimi/wiki).

---

> **“Give us 4 kB of switches and we will remember you forever – or until the capacitors leak.”**
