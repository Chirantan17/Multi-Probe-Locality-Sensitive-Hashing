# Project Report: Multi-Probe Locality-Sensitive Hashing (LSH) for Nearest Neighbors
**Course:** CS G526: Advanced Algorithms & Complexity  
**Project Track:** [P22] Probabilistic & Streaming Structures  

---

## 1. Executive Summary & Problem Formulation
Nearest Neighbor search in high-dimensional vector spaces is fundamental to vector databases, retrieval-augmented generation (RAG), and similarity retrieval. In spaces where dimension  > 20$, traditional partitioning tree data structures ($-d trees, R-trees) suffer from the *Curse of Dimensionality*, degenerating into exhaustive linear scans requiring $\mathcal{O}(N \cdot d)$ operations per query.

Locality-Sensitive Hashing (LSH), introduced by Indyk & Motwani (1998) and Gionis et al. (1999), provides sublinear approximate nearest neighbor retrieval with rigorous probabilistic guarantees. However, standard LSH suffers from boundary-miss errors: points that are close in metric space are frequently separated by random hyperplane boundaries. To achieve high recall ($\ge 90\%$), classic LSH requires constructing dozens of independent hash tables ( \ge 30$), resulting in severe memory expansion.

This project implements and evaluates **Multi-Probe LSH** (Lv et al., 2007) for cosine distance using random hyperplane projections (SimHash, Charikar 2002). By evaluating projection boundary margins $\delta_i(q) = |w_i^T q|$, the engine dynamically probes adjacent Hamming-1 and Hamming-2 hash buckets in order of perturbation cost. Empirical results confirm that a compact index with  = 6$ tables matches or exceeds the recall of classic  = 32$ LSH while achieving an **~81% reduction in index RAM overhead**.

---

## 2. Mathematical Foundations

### 2.1 Random Hyperplane Projections (SimHash)
For unit vectors , v \in \mathbb{S}^{d-1}$ and a random normal hyperplane vector  \sim \mathcal{N}(0, I_d)$, the hash bit is defined as (v) = \text{sign}(w^T v)$. The collision probability is directly proportional to angular similarity:
}\Pr[h_w(u) = h_w(v)] = 1 - \frac{\theta(u, v)}{\pi}}
where $\theta(u, v) = \arccos(u \cdot v)$.

### 2.2 Amplification & S-Curve Mechanics
To separate close vectors from distant vectors, $ independent hash functions are concatenated into a composite hash key (v) = (h_1(v), \dots, h_K(v))$. To recover recall, $ independent tables are constructed. The overall probability of collision in at least one table for vectors with similarity  = \cos(\theta)$ is given by the S-curve:
}\Pr[\text{Collision}] = 1 - (1 - s^K)^L}
- **Increasing $ (Bits):** Sharpens candidate selectivity, reducing false positives $\mathcal{O}(C \cdot d)$, but increases boundary misses.
- **Increasing $ (Tables):** Increases collision probability, but scales memory consumption linearly $\mathcal{O}(L \cdot N)$.

### 2.3 Margin-Guided Multi-Probe Perturbation
Instead of evaluating only the base bucket (q)$, Multi-Probe inspects projection distances:
}\delta_i(q) = |w_i^T q|}
Hyperplanes with minimal $\delta_i(q)$ represent low-confidence boundary decisions where the query is closest to the dividing hyperplane. The router enumerates candidate perturbations:
1. **Hamming Radius 1:** Single bit flips with cost (i) = \delta_i$.
2. **Hamming Radius 2:** Pairwise bit flips with cost (i, j) = \delta_i + \delta_j$.

Candidate buckets are sorted in ascending order of cost and probed within a defined query budget $. For  = 8$, radius-1 and radius-2 generation yields an accessible candidate pool of  + K + \frac{K(K-1)}{2} = 37$ buckets.

---

## 3. Systems Architecture & Optimizations

1. **AVX2 SIMD Fused Multiply-Add:** Exact candidate reranking utilizes hand-tuned _mm256_fmadd_ps intrinsics, processing 8 single-precision float operations per cycle.
2. **Visited-Stamp Array Deduplication:** Replaces dynamic hash sets (std::unordered_set) with a flat std::vector<uint32_t> visited array indexed by VectorId. Each query increments a monotonic query tag, enabling $\mathcal{O}(1)$ deduplication without heap allocations.
3. **OpenMP Multi-Threaded Batching:** Batched query workloads are distributed dynamically across CPU cores (#pragma omp parallel for schedule(dynamic)), reporting QPS and p50/p95/p99 latency percentiles.

---

## 4. Empirical Evaluation & Results

Evaluated on  = 10,000$ 128-dimensional vectors across 500 held-out test queries ( = 8$):

| Configuration | Index RAM | Recall@10 | Latency (Avg) | Latency (p95) | Speedup vs BF |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exact Brute-Force** | 4.88 MB (Raw) | 100.0% | ~410 $\mu | ~490 $\mu | 1.00x |
| **Classic LSH (=32$, 1-p)** | **3.82 MB** | 98.4% | ~115 $\mu | ~140 $\mu | 3.56x |
| **Compact LSH (=6$, 1-p)** | **0.72 MB** | 68.2% | ~35 $\mu | ~45 $\mu | 11.7x |
| **Multi-Probe LSH (=6$, 8-p)** | **0.72 MB** | **98.8%** | ~145 $\mu | ~185 $\mu | **2.83x** |

### Key Experimental Takeaways:
1. **Memory Reduction:** Compact Multi-Probe LSH (=6$) achieves matching recall to Classic LSH (=32$) while reducing index memory from 3.82 MB to 0.72 MB (**81.1% memory reduction**).
2. **Boundary Recovery:** On compact indices (=6$), increasing probes from 1 to 8 raises recall from 68.2% to 98.8%, proving the effectiveness of boundary-miss recovery.
3. **Latency Trade-off:** Multi-Probe incurs a modest query latency overhead compared to 1-probe standard LSH due to probing multiple buckets, but maintains a ~2.8x speedup over exact brute-force search.

---

## 5. Limitations & Future Extensions
- **Radius Truncation:** Restricting enumeration to radius 1 and 2 caps the maximum probe budget at 37 for =8$. Higher budgets require priority-queue dynamic heap traversal.
- **Extreme High Dimensionality:** In spaces where  \gg 1000$ with uniform spherical dispersion, margin variance diminishes, requiring data-dependent projection techniques such as PCA-Tree or Cross-Polytope LSH.
