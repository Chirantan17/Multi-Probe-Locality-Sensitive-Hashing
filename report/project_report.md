# Project Report: Multi-Probe Locality-Sensitive Hashing (LSH) for Nearest Neighbors
**Course:** CS G526: Advanced Algorithms & Complexity  
**Project Track:** [P22] Probabilistic & Streaming Structures  
**Corpus Evaluated:** INRIA SIFT10K (Real-world 128D Image Feature Descriptors)

---

## 1. Executive Summary & Problem Formulation
Nearest Neighbor search in high-dimensional vector spaces is fundamental to vector databases, retrieval-augmented generation (RAG), and image feature matching. In spaces where dimension \ > 20\$, traditional partitioning tree structures (\\$-d trees, R-trees) suffer from the *Curse of Dimensionality*, degenerating into exhaustive linear scans requiring \$\mathcal{O}(N \cdot d)\$ operations per query.

Locality-Sensitive Hashing (LSH), introduced by Indyk & Motwani (1998) and Gionis et al. (1999), provides sublinear approximate nearest neighbor retrieval with rigorous probabilistic guarantees. However, standard LSH suffers from boundary-miss errors: points that are close in metric space are frequently separated by random hyperplane boundaries. To achieve high recall (\$\ge 90\%\$), classic LSH requires constructing dozens of independent hash tables (\ \ge 32\$), resulting in severe memory expansion.

This project implements and evaluates **Multi-Probe LSH** (Lv et al., 2007) for cosine distance using random hyperplane projections (SimHash, Charikar 2002). By evaluating projection boundary margins \$\delta_i(q) = |w_i^T q|\$, the engine dynamically probes adjacent Hamming-1 and Hamming-2 hash buckets in order of perturbation cost. Empirical results on the standard INRIA SIFT10K benchmark confirm that a compact index with \ = 6\$ tables achieves **99.80% recall** matching classic \ = 32\$ LSH while achieving an **81.50% reduction in index RAM overhead**.

---

## 2. Mathematical Foundations

### 2.1 Random Hyperplane Projections (SimHash)
For unit vectors \, v \in \mathbb{S}^{d-1}\$ and a random normal hyperplane vector \ \sim \mathcal{N}(0, I_d)\$, the hash bit is defined as \(v) = \text{sign}(w^T v)\$. The collision probability is directly proportional to angular similarity:
main\Pr[h_w(u) = h_w(v)] = 1 - \frac{\theta(u, v)}{\pi}main
where \$\theta(u, v) = \arccos(u \cdot v)\$.

### 2.2 Amplification & S-Curve Mechanics
To separate close vectors from distant vectors, \\$ independent hash functions are concatenated into a composite hash key \(v) = (h_1(v), \dots, h_K(v))\$. To recover recall, \\$ independent tables are constructed. The overall probability of collision in at least one table for vectors with similarity \ = \cos(\theta)\$ is given by the S-curve:
main\Pr[\text{Collision}] = 1 - (1 - s^K)^Lmain
- **Increasing \\$ (Bits):** Sharpens candidate selectivity, reducing false positives \$\mathcal{O}(C \cdot d)\$, but increases boundary misses.
- **Increasing \\$ (Tables):** Increases collision probability, but scales memory consumption linearly \$\mathcal{O}(L \cdot N)\$.

### 2.3 Margin-Guided Multi-Probe Perturbation
Instead of evaluating only the base bucket \(q)\$, Multi-Probe inspects projection distances:
main\delta_i(q) = |w_i^T q|main
Hyperplanes with minimal \$\delta_i(q)\$ represent low-confidence boundary decisions where the query is closest to the dividing hyperplane. The router enumerates candidate perturbations:
1. **Hamming Radius 1:** Single bit flips with cost \(i) = \delta_i\$.
2. **Hamming Radius 2:** Pairwise bit flips with cost \(i, j) = \delta_i + \delta_j\$.

Candidate buckets are sorted in ascending order of cost and probed within a defined query budget \\$. For \ = 8\$, radius-1 and radius-2 generation yields an accessible candidate pool of \ + K + \frac{K(K-1)}{2} = 37\$ buckets.

---

## 3. Systems Architecture & Optimizations

1. **AVX2 SIMD Fused Multiply-Add:** Exact candidate reranking utilizes hand-tuned \_mm256_fmadd_ps\ intrinsics, processing 8 single-precision float operations per cycle.
2. **Visited-Stamp Array Deduplication:** Replaces dynamic hash sets (\std::unordered_set\) with a flat \std::vector<uint32_t>\ visited array indexed by \VectorId\. Each query increments a monotonic query tag, enabling \$\mathcal{O}(1)\$ deduplication without dynamic heap allocations.
3. **OpenMP Multi-Threaded Batching:** Batched query workloads are distributed dynamically across CPU cores (\#pragma omp parallel for schedule(dynamic)\), reporting QPS and p50/p95/p99 latency percentiles.

---

## 4. Empirical Evaluation & SIFT10K Results

Evaluated on INRIA SIFT10K (\ = 10,000\$ 128-dimensional descriptors, 100 query vectors, \ = 8\$, 16 OpenMP threads):

| Configuration | Index RAM | Recall@10 | Latency (Avg) | Latency (p95) | Batch QPS | Speedup vs BF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Brute-Force** | 4.88 MB (Raw) | 100.0% | 1120.19 \$\mu\ | 4928.50 \$\mu\ | 11,329 | 1.00x |
| **Classic LSH (\=32\$, 1-p)** | **2.12 MB** | 100.00% | 716.83 \$\mu\ | 3205.50 \$\mu\ | 12,792 | 1.56x |
| **Compact LSH (\=6\$, 1-p)** | **0.39 MB** | 82.90% | 241.70 \$\mu\ | 1212.00 \$\mu\ | 19,952 | 4.63x |
| **Multi-Probe LSH (\=6\$, 8-p)** | **0.39 MB** | **99.80%** | **533.41 \$\mu\** | **1012.90 \$\mu\** | 9,728 | **2.10x** |

### Key Experimental Takeaways:
1. **Memory Reduction:** Compact Multi-Probe LSH (\=6\$) achieves matching recall (99.80% vs 100.0%) to Classic LSH (\=32\$) while reducing index memory from 2.12 MB to 0.39 MB (**81.50% memory reduction**).
2. **Boundary Recovery:** On compact indices (\=6\$), increasing probes from 1 to 8 raises recall from 82.90% to 99.80%, proving the effectiveness of boundary-miss recovery on real-world feature manifolds.
3. **Latency & Tail Latency:** Multi-Probe (\.41 \$\mu\) achieves lower query latency and over **\.1\times\$ lower p95 tail latency** than Classic LSH (\.90 \$\mu\ vs \.50 \$\mu\) by avoiding redundant cross-table candidate scans.
