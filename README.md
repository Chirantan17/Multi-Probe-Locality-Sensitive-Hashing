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
- **Visual Diagnostics:** Standalone 60 FPS WebGL / Three.js 3D boundary diagnostic playground (`index.html`).

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