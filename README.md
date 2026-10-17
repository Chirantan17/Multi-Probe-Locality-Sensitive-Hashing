# High-Performance Multi-Probe LSH Vector Search Engine

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Hardware](https://img.shields.io/badge/Hardware-AVX2%20%2B%20FMA-orange.svg)](#2-key-architectural-features)
[![Parallel](https://img.shields.io/badge/Parallel-OpenMP-green.svg)](#2-key-architectural-features)
[![Coursework](https://img.shields.io/badge/Coursework-BITS%20Pilani%20CS%20G526-red.svg)](report/project_report.md)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An AVX2 SIMD-accelerated Approximate Nearest Neighbor (ANN) search engine implementing Locality-Sensitive Hashing (LSH) and margin-guided Multi-Probe perturbation routing in modern C++20, paired with an interactive 3D WebGL diagnostic workbench.

---

## 1. Executive Overview

High-dimensional vector similarity retrieval is the computational backbone of modern retrieval-augmented generation (RAG), vector databases, search recommendation engines, and high-throughput biometrics.

- **The Problem:** Exact brute-force search computes distances across every vector, scaling at $\mathcal{O}(N \cdot d)$ time per query. Traditional tree-based spatial indices ($k$-d trees, R-trees) break down due to the *Curse of Dimensionality* ($d > 20$), degrading to linear scans.
- **The Bottleneck in Classic LSH:** Standard random projection LSH suffers from boundary-miss errors. To reach high recall ($\ge 90\%$), standard LSH requires building dozens of independent hash tables ($L \gg 30$), resulting in excessive RAM consumption.
- **The Solution:** This engine implements **Multi-Probe LSH**. By sorting projection margins to identify the least-certain boundary hyperplanes, it systematically probes neighboring Hamming-1 and Hamming-2 buckets. This achieves $\ge 90\%$ recall with up to an **$80\%$ reduction in index memory footprint** compared to classic multi-table LSH.

---

## 2. Key Architectural Features

- **AVX2 Vectorized Kernels:** Hand-tuned `_mm256_fmadd_ps` fused multiply-add intrinsics executing 8 single-precision float operations per clock cycle for cosine distance reranking.
- **32-Byte Memory Alignment:** Allocations enforced with `alignas(32)` to enable direct aligned vector loads (`_mm256_load_ps`) with zero cache-line split penalties.
- **Sublinear Candidate Pruning:** Dynamically filters large vector corpora down to a small fraction of candidate items before exact scoring.
- **Multi-Probe Perturbation Router:** Intelligently constructs query-dependent probe sequences based on projection distance to hyperplane boundaries: $\vert{}w_i^T q\vert{}$.
- **Multi-Core Query Batching:** OpenMP multi-threading parallelizes query streams across CPU cores, measuring QPS and latency distributions (p50, p95, p99).
- **Dual Visual Diagnostics:**
  - Standalone 60 FPS WebGL / Three.js 3D boundary diagnostic playground (`index.html`).
  - Interactive Streamlit analytics dashboard (`app.py`).

---

## 3. End-to-End Pipeline Architecture

```text
[ High-Dim Dataset ] ---> [ Ingestion Engine ] ---> [ AVX2 Projection Engine ]
                                                          |
                                                          v
[ Query Vector ] ----> [ Multi-Probe Router ] ------> [ Compact Hash Tables ]
                                                          |
                                                          v
                                                [ Candidate Shortlist ]
                                                          |
                                                          v
                                          [ SIMD Reranker (Top-K) ]
                                                          |
                                                          v
                                          [ Visual Diagnostics Dashboard ]

4. Directory Structure
Locality-Sensitive Hashing (LSH)/
├── backend/
│   ├── include/
│   │   ├── types.hpp           # 32-byte aligned vectors, neighbor structs, .fvecs loader
│   │   ├── simd_metric.hpp     # AVX2 FMA cosine distance & dot-product kernels
│   │   ├── lsh_index.hpp       # Random projection tables and hash storage
│   │   └── multiprobe.hpp      # Boundary margin ranker & perturbation router
│   └── src/
│       └── main.cpp            # Multi-threaded OpenMP benchmark engine & telemetry exporter
├── benchmarks/
│   └── results.json            # Runtime telemetry data (latency, recall, QPS, RAM)
├── build/                      # CMake build artifacts (git-ignored)
├── data/                       # Datasets (.fvecs binary format / synthetic cache)
├── frontend/                   # Frontend workspace assets
├── report/
│   └── project_report.md       # Comprehensive academic project report
├── .gitignore                  # Git artifact and cache exclusion rules
├── app.py                      # Python Streamlit visual diagnostics dashboard
├── CMakeLists.txt              # CMake build configuration with AVX2/OpenMP/O3
├── index.html                  # Standalone 60 FPS WebGL 3D visual playground
├── README.md                   # System documentation
└── requirements.txt            # Python visualization dependencies


# High-Performance Multi-Probe LSH Vector Search Engine

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Hardware](https://img.shields.io/badge/Hardware-AVX2%20%2B%20FMA-orange.svg)](#2-key-architectural-features)
[![Parallel](https://img.shields.io/badge/Parallel-OpenMP-green.svg)](#2-key-architectural-features)
[![Coursework](https://img.shields.io/badge/Coursework-BITS%20Pilani%20CS%20G526-red.svg)](report/project_report.md)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An AVX2 SIMD-accelerated Approximate Nearest Neighbor (ANN) search engine implementing Locality-Sensitive Hashing (LSH) and margin-guided Multi-Probe perturbation routing in modern C++20, paired with an interactive 3D WebGL diagnostic workbench.

---

## 1. Executive Overview

High-dimensional vector similarity retrieval is the computational backbone of modern retrieval-augmented generation (RAG), vector databases, search recommendation engines, and high-throughput biometrics.

- **The Problem:** Exact brute-force search computes distances across every vector, scaling at $\mathcal{O}(N \cdot d)$ time per query. Traditional tree-based spatial indices ($k$-d trees, R-trees) break down due to the *Curse of Dimensionality* ($d > 20$), degrading to linear scans.
- **The Bottleneck in Classic LSH:** Standard random projection LSH suffers from boundary-miss errors. To reach high recall ($\ge 90\%$), standard LSH requires building dozens of independent hash tables ($L \gg 30$), resulting in excessive RAM consumption.
- **The Solution:** This engine implements **Multi-Probe LSH**. By sorting projection margins to identify the least-certain boundary hyperplanes, it systematically probes neighboring Hamming-1 and Hamming-2 buckets. This achieves $\ge 90\%$ recall with up to an **$80\%$ reduction in index memory footprint** compared to classic multi-table LSH.

---

## 2. Key Architectural Features

- **AVX2 Vectorized Kernels:** Hand-tuned `_mm256_fmadd_ps` fused multiply-add intrinsics executing 8 single-precision float operations per clock cycle for cosine distance reranking.
- **32-Byte Memory Alignment:** Allocations enforced with `alignas(32)` to enable direct aligned vector loads (`_mm256_load_ps`) with zero cache-line split penalties.
- **Sublinear Candidate Pruning:** Dynamically filters large vector corpora down to a small fraction of candidate items before exact scoring.
- **Multi-Probe Perturbation Router:** Intelligently constructs query-dependent probe sequences based on projection distance to hyperplane boundaries: $\vert{}w_i^T q\vert{}$.
- **Multi-Core Query Batching:** OpenMP multi-threading parallelizes query streams across CPU cores, measuring QPS and latency distributions (p50, p95, p99).
- **Dual Visual Diagnostics:**
  - Standalone 60 FPS WebGL / Three.js 3D boundary diagnostic playground (`index.html`).
  - Interactive Streamlit analytics dashboard (`app.py`).

---

## 3. End-to-End Pipeline Architecture

```text
[ High-Dim Dataset ] ---> [ Ingestion Engine ] ---> [ AVX2 Projection Engine ]
                                                          |
                                                          v
[ Query Vector ] ----> [ Multi-Probe Router ] ------> [ Compact Hash Tables ]
                                                          |
                                                          v
                                                [ Candidate Shortlist ]
                                                          |
                                                          v
                                          [ SIMD Reranker (Top-K) ]
                                                          |
                                                          v
                                          [ Visual Diagnostics Dashboard ]
```

---

## 4. Directory Structure

```text
Locality-Sensitive Hashing (LSH)/
├── backend/
│   ├── include/
│   │   ├── types.hpp           # 32-byte aligned vectors, neighbor structs, .fvecs loader
│   │   ├── simd_metric.hpp     # AVX2 FMA cosine distance & dot-product kernels
│   │   ├── lsh_index.hpp       # Random projection tables and hash storage
│   │   └── multiprobe.hpp      # Boundary margin ranker & perturbation router
│   └── src/
│       └── main.cpp            # Multi-threaded OpenMP benchmark engine & telemetry exporter
├── benchmarks/
│   └── results.json            # Runtime telemetry data (latency, recall, QPS, RAM)
├── build/                      # CMake build artifacts (git-ignored)
├── data/                       # Datasets (.fvecs binary format / synthetic cache)
├── frontend/                   # Frontend workspace assets
├── report/
│   └── project_report.md       # Comprehensive academic project report
├── .gitignore                  # Git artifact and cache exclusion rules
├── app.py                      # Python Streamlit visual diagnostics dashboard
├── CMakeLists.txt              # CMake build configuration with AVX2/OpenMP/O3
├── index.html                  # Standalone 60 FPS WebGL 3D visual playground
├── README.md                   # System documentation
└── requirements.txt            # Python visualization dependencies
```

---

## 5. Prerequisites

- **Build System:** CMake $\ge 3.20$
- **C++ Compiler:** C++20 compliant compiler supporting AVX2 intrinsics and OpenMP (MinGW GCC $\ge 11.0$, Clang $\ge 13.0$, or MSVC $\ge 2019$)
- **Python Environment:** Python 3.10+ (for optional visual dashboards)

---

## 6. Build & Execution Instructions

### A. Build the C++ Engine
From the project root directory:

```powershell
cmake -B build -S .
cmake --build build --config Release
```

### B. Run the Benchmark CLI
```powershell
# Default run (N=10000, D=128, L=6, K=8, Probes=6)
.\build\lsh_bench.exe

# Custom scale run (N=25000, D=128, K=8, L=6, Probes=8)
.\build\lsh_bench.exe -n 25000 -d 128 -k 8 -l 6 -p 8
```

#### CLI Parameters:
- `-n` : Total dataset vector count (e.g., `10000`, `25000`)
- `-d` : Vector dimension (e.g., `64`, `128`, `256`)
- `-k` : Number of hyperplanes per table ($K$)
- `-l` : Number of hash tables ($L$)
- `-p` : Probes evaluated per table

### C. Launch the Visual Analytics Dashboards
- **Standalone WebGL 3D Playground (Recommended):**
  ```powershell
  python -m http.server 8000
  # Open in browser: http://localhost:8000/index.html
  ```
- **Streamlit Analytics Dashboard:**
  ```powershell
  pip install -r requirements.txt
  python -m streamlit run app.py
  ```

---

## 7. Empirical Benchmark Results

Evaluated across clustered 128-dimensional synthetic and real feature embeddings ($L = 6$ tables, $K = 8$ bits, Top-$K = 10$):

| Dataset Size ($N$) | Search Method | Latency ($\mu$s) | Recall (%) | Speedup vs BF | Candidates Scored |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **10,000** | Exact Brute-Force | 301.76 $\mu$s | 100.0% | 1.00x | 10,000 (100%) |
| | Standard LSH (1-Probe) | 83.75 $\mu$s | 99.0% | 3.60x | ~780 (~8%) |
| | **Multi-Probe LSH (6-Probe)** | **194.98 $\mu$s** | **100.0%** | **1.55x** | **~1,840 (~18%)** |
| **25,000** | Exact Brute-Force | 1151.86 $\mu$s | 100.0% | 1.00x | 25,000 (100%) |
| | Standard LSH (1-Probe) | 281.29 $\mu$s | 99.0% | 4.09x | ~1,950 (~8%) |
| | **Multi-Probe LSH (8-Probe)** | **737.95 $\mu$s** | **100.0%** | **1.56x** | **~4,600 (~18%)** |

---

## 8. Mathematical Foundations

### Random Hyperplane Projections (SimHash)
For vectors $u, v \in \mathbb{R}^d$ and a random Gaussian hyperplane vector $w \sim \mathcal{N}(0, I_d)$, the collision probability for the hash function $h_w(v) = \text{sign}(w^T v)$ is directly proportional to their angular similarity:

$$\Pr[h_w(u) = h_w(v)] = 1 - \frac{\theta(u, v)}{\pi}$$

Where $\theta(u, v) = \arccos\left(\frac{u \cdot v}{\|u\| \|v\|}\right)$.

### S-Curve Amplification
Combining $K$ independent hyperplanes per table across $L$ distinct tables creates an S-curve filter function. Two vectors with cosine similarity $s$ collide in at least one table with probability:

$$\Pr[\text{Collision}] = 1 - (1 - s^K)^L$$

- **Increasing $K$ (Bits):** Sharpens candidate filtering (fewer false positives to rerank), but increases boundary misses.
- **Increasing $L$ (Tables):** Boosts recall, but scales index memory linearly ($\mathcal{O}(L \cdot N)$).

### Multi-Probe Perturbation Mechanics
Instead of evaluating only the base bucket $g(q) = (h_1(q), \dots, h_K(q))$, Multi-Probe computes the margin to each hyperplane:

$$\delta_i(q) = |w_i^T q|$$

Hyperplanes with the smallest margins $\delta_i(q)$ represent coordinates where the query lies closest to the boundary. The router flips these low-confidence bits first, inspecting neighboring buckets ordered by increasing boundary distance penalty:

1. **Hamming Radius 1:** Flips individual bits corresponding to the smallest single-margin coordinate.
2. **Hamming Radius 2:** Flips pairs of bits whose summed margin penalties are minimal.

This enables a compact index ($L = 6$) to match or exceed the recall of an index with $L = 30+$, preventing index memory explosion.