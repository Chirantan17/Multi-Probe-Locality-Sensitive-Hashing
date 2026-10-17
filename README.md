# High-Performance Multi-Probe LSH Vector Search Engine

An AVX2 SIMD-accelerated Approximate Nearest Neighbor (ANN) search engine implementing Locality-Sensitive Hashing (LSH) and Multi-Probe perturbation routing in modern C++20, paired with an interactive diagnostic web visualizer.

---

## 1. Executive Overview

High-dimensional vector similarity retrieval is the computational backbone of modern retrieval-augmented generation (RAG), vector databases, search recommendation engines, and high-throughput biometrics.

- **The Problem:** Exact brute-force search computes distances across every vector, scaling at $\mathcal{O}(N \cdot d)$ time per query. Traditional tree-based spatial indices ($k$-d trees, R-trees) break down due to the *Curse of Dimensionality* ($d > 20$), degrading to linear scans.
- **The Bottleneck in Classic LSH:** Standard random projection LSH suffers from boundary-miss errors. To reach high recall ($\ge 90\%$), standard LSH requires building dozens of independent hash tables ($L \gg 30$), resulting in excessive RAM consumption.
- **The Solution:** This engine implements **Multi-Probe LSH**. By sorting projection margins to identify the least-certain boundary hyperplanes, it systematically probes neighboring Hamming-1 and Hamming-2 buckets. This achieves $\ge 90\%$ recall with up to an **$80\%$ reduction in index memory footprint** compared to classic multi-table LSH.

---

## 2. Key Architectural Features

- **AVX2 Vectorized Kernels:** Hand-tuned `_mm256_fmadd_ps` fused multiply-add intrinsics executing 8 single-precision float operations per clock cycle for cosine distance reranking.
- **Sublinear Candidate Pruning:** Dynamically filters large vector corpora down to a small fraction of candidate items before exact scoring.
- **Multi-Probe Perturbation Router:** Intelligently constructs query-dependent probe sequences based on projection distance to hyperplane boundaries: $\vert{}w_i^T q\vert{}$.
- **Telemetry & Web Workbench:** Exported run metrics pipe directly into an interactive Streamlit dashboard featuring:
  - Empirical speedup vs. recall trade-off curves.
  - Candidate search-space reduction telemetry.
  - 3D interactive hyperplane partitioning and boundary-recovery playground.
  - Dynamic theoretical S-curve filter simulator ($P = 1 - (1 - s^K)^L$).

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