# Multi-Probe Locality-Sensitive Hashing for High-Dimensional Nearest-Neighbor Search

**Course:** CS G526 — Advanced Algorithms and Complexity (BITS Pilani, Hyderabad Campus)  
**Track:** [P22] Probabilistic & Streaming Data Structures  
**Authors:** Vatsal Chauhan (2026H1030064H), Chirantan Ghosh (2026H1030034H), Prithviraj Kapil (2026H1030047H)  
**Benchmark Corpus:** INRIA SIFT10K ($N = 10,000, d = 128, Q = 100$) + Scaling up to $N = 100,000$

---

## 1. Executive Summary & Scope of Claims

Approximate Nearest Neighbor (ANN) search in high-dimensional spaces ($d = 128$) faces the *Curse of Dimensionality*, where spatial partitioning trees ($k$-d trees, R-trees) degrade to $\mathcal{O}(N \cdot d)$ exhaustive scans. Locality-Sensitive Hashing (LSH) provides sublinear theoretical guarantees, but standard random-hyperplane LSH (SimHash) requires a large number of hash tables ($L \ge 32$) to overcome boundary-miss errors, inflating index memory linearly with $L$.

This project implements and audits **Margin-Guided Multi-Probe LSH** (Lv et al., 2007) in C++20. All empirical tables in this report are generated directly from `benchmarks/results.json` under a deterministic seed protocol (`seed = 777`, plus 5-seed variability analysis). Specifically, we demonstrate:
1. **Index Memory Reduction vs. Recall Recovery:** Reducing $L$ from $32$ to $6$ cuts index structural RAM by **81.50%** (2.12 MB $\rightarrow$ 0.39 MB; total footprint including the 4.88 MB raw dataset drops by **24.65%**, with process RSS measured at **25.09 MB**). Single-probe recall on the compact $L=6$ index drops to **82.90%** (uncentered) / **64.40%** (centered), whereas $P=8$ multi-probe routing recovers **99.80%** (uncentered) / **97.40%** (centered).
2. **SIFT Non-Negativity & Bucket Entropy:** Raw SIFT descriptors lie strictly in the non-negative orthant ($\mathbb{R}_+^{128}$), causing origin-centered hyperplanes to populate only **168/256** buckets with a maximum bucket size of **1610** (Shannon entropy **5.45 bits**). Mean-centering the projection space populates **255/256** buckets, reduces the max bucket to **317** (entropy **7.16 bits**), and cuts the candidate fraction scanned by more than half (from **67.44%** down to **27.45%**).
3. **Honest Latency & Scaling Regime:** At $N = 10,000$, the 4.88 MiB dataset is completely L3-cache resident, allowing contiguous AVX2 brute-force scans to complete in **180.87 $\mu$s**. At this small scale, hash-table indirection and candidate gathering make LSH slower or comparable to brute force. However, our scaling study proves that at $N = 100,000$ ($K=12, L=6, P=8$), Multi-Probe LSH evaluates only **6.80%** of the corpus and achieves a **8.67$\times$ single-thread latency speedup** over AVX2 brute force (**615.09 $\mu$s** vs. **5329.88 $\mu$s** at **91.50%** recall).

---

## 2. Theoretical Foundations & Complexity Analysis

### 2.1 Random Hyperplane LSH (Charikar SimHash)
Given unit vectors $u, v \in \mathbb{S}^{d-1}$ separated by angle $\theta(u, v) = \arccos(u^T v) \in [0, \pi]$ and a Gaussian hyperplane normal $w \sim \mathcal{N}(0, I_d)$, the hash function $h_w(x) = \mathbb{I}[w^T x \ge 0]$ separates $u$ and $v$ if and only if the projection of $w$ onto $\text{span}(u, v)$ falls within the two opposite angular wedges of measure $\theta$. By rotational invariance (Goemans & Williamson, 1995):
$$\Pr[h_w(u) = h_w(v)] = 1 - \frac{\theta(u, v)}{\pi}$$

### 2.2 $(K, L)$ Amplification, S-Curve, and Sublinear Exponent $\rho$
Concatenating $K$ independent hyperplanes yields anAND-construction with single-table collision probability $p(\theta) = (1 - \theta/\pi)^K$. Constructing $L$ independent tables yields an OR-construction whose probability of colliding in at least one table is the exact SimHash S-curve:
$$\Pr[\text{Collision}(u, v)] = 1 - \left[1 - \left(1 - \frac{\theta(u, v)}{\pi}\right)^K\right]^L$$

In the $(r_1, r_2, p_1, p_2)$-LSH framework (Indyk & Motwani, 1998), with $p_1 = 1 - \theta_1/\pi$ for near neighbors and $p_2 = 1 - \theta_2/\pi$ for distant points ($p_1 > p_2$), setting $K = \lceil \log_{1/p_2} N \rceil$ bounds expected false positives per table to $\mathcal{O}(1)$, while $L = \Theta(N^\rho)$ tables guarantee constant recall, where:
$$\rho = \frac{\ln(1/p_1)}{\ln(1/p_2)} < 1$$
Leading to sublinear query time $\mathcal{O}(d N^\rho)$ and space $\mathcal{O}(N^{1+\rho} + N d)$.

### 2.3 Margin-Guided Multi-Probe Routing: Gaussian Squared vs. Linear Heuristic
When a true neighbor $v$ differs from query $q$ by a small angle, $v$ lands in a bucket whose code differs from $g(q)$ only on bits where $q$ lies close to the hyperplane boundary. The distance from $q$ to hyperplane $w_i$ is the projection margin $\delta_i(q) = |w_i^T q|$.

Under Lv et al. (2007), the projection $w_i^T(v - q)$ follows a Gaussian distribution, so the log-likelihood of flipping a subset of bits $M \subseteq \{1, \dots, K\}$ is proportional to the **sum of squared margins**:
$$C_{\text{sq}}(M) = \sum_{i \in M} \delta_i(q)^2$$
An alternative Laplace-like heuristic uses the **linear sum** $C_{\text{lin}}(M) = \sum_{i \in M} \delta_i(q)$. Notice that $C_{\text{sq}}$ favors flipping two very small margins over one moderate margin whenever $\delta_1^2 + \delta_2^2 < \delta_3^2$, even if $\delta_1 + \delta_2 > \delta_3$. Our engine implements both cost models (`CostMode::Squared` and `CostMode::Linear`) and restricts enumeration to Hamming radii $r \in \{1, 2\}$, giving an explicit pool of $1 + K + \binom{K}{2}$ candidate buckets per table ($37$ buckets for $K=8$). Enumerating and sorting these candidates requires $\mathcal{O}(L K^2 \log K)$ time per query, whereas Lv et al.'s min-heap shift/expand generator requires $\mathcal{O}(L P \log P)$.

---

## 3. Systems Architecture & Verification Invariants

1. **32-Byte Aligned Vector Storage:** `std::vector` header alignment (`alignas(32)`) does not align heap buffers. We implement a custom C++20 `aligned_allocator<float, 32>` using `_aligned_malloc` / `posix_memalign`, guaranteeing `reinterpret_cast<uintptr_t>(v.raw()) % 32 == 0` on every vector and enabling hardware-aligned `_mm256_load_ps` + `_mm256_fmadd_ps` execution.
2. **Zero-Allocation Hot Loop (`QueryScratchpad`):** All intermediate query structures (`visited_stamps`, `bit_margins`, `candidates`, `selected_probes`, `candidate_ids`, `neighbors`, and `centered_q`) are pre-allocated once per worker thread inside `QueryScratchpad` prior to entering the timed query loop.
3. **Visited-Stamp Invariant & $2^{32}-1$ Rollover:** Each thread maintains an independent `visited_stamps` array of size $N$ and a monotonic `tag`. Tag `0` is strictly reserved for the unvisited state. When `tag` reaches `0xFFFFFFFFU`, `advance_tag()` zeroes the entire `visited_stamps` array and resets `tag = 1U`, preventing silent candidate drops on wrap-around.

---

## 4. Empirical Evaluation on INRIA SIFT10K

### 4.1 Hyperplane Partition Entropy (Raw vs. Mean-Centered SIFT10K)

| Partitioning Mode ($L=1, K=8, N=10,000$) | Non-Empty Buckets (of 256) | Max Bucket Size | Shannon Entropy (Max 8.00 bits) |
| :--- | :---: | :---: | :---: |
| **Uncentered SIFT10K** | 168 / 256 | 1610 | 5.45 bits |
| **Mean-Centered SIFT10K** | 255 / 256 | 317 | 7.16 bits |

### 4.2 Four-Way Core Ablation — Uncentered SIFT10K ($K=8$, 5-Trial Medians)

| Configuration | Index RAM | Recall@10 | ST Avg ($\mu$s) | ST p50 ($\mu$s) | ST p95 ($\mu$s) | ST p99 ($\mu$s) | Cand Frac | Buckets | OpenMP QPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Brute-Force** | 0.00 MB | 100.00% | 180.87 | 169.55 | 258.40 | 316.29 | 100.00% | 0 | 33333 |
| **Classic LSH (L=32, P=1)** | 2.12 MB | 100.00% | 274.46 | 242.70 | 438.03 | 552.27 | 51.14% | 32 | 27180 |
| **Compact LSH (L=6, P=1)** | 0.39 MB | 82.90% | 164.75 | 155.10 | 359.62 | 472.78 | 22.11% | 6 | 59068 |
| **Multi-Probe LSH (L=6, P=8 Sq)** | 0.39 MB | 99.80% | 381.74 | 365.80 | 652.90 | 793.60 | 67.44% | 48 | 23559 |
| **Multi-Probe LSH (L=6, P=8 Lin)** | 0.39 MB | 99.80% | 245.25 | 244.00 | 316.95 | 336.65 | 67.44% | 48 | 26743 |

> **Note on Candidate Fraction (Uncentered):** On uncentered SIFT10K, Multi-Probe ($L=6, P=8$) probes $48$ buckets and evaluates **67.44%** of the corpus compared to **51.14%** for Classic LSH ($L=32, P=1$). Thus, Multi-Probe trades ~16% more candidate evaluations to save **81.5% index RAM** while recovering recall from **82.90%** to **99.80%**.

### 4.3 Four-Way Core Ablation — Mean-Centered Hashing + Raw Cosine Reranking ($K=8$)

| Configuration | Index RAM | Recall@10 | ST Avg ($\mu$s) | ST p50 ($\mu$s) | ST p95 ($\mu$s) | ST p99 ($\mu$s) | Cand Frac | Buckets | OpenMP QPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Brute-Force** | 0.00 MB | 100.00% | 180.87 | 169.55 | 258.40 | 316.29 | 100.00% | 0 | 33333 |
| **Centered Classic (L=32, P=1)** | 2.17 MB | 98.00% | 88.01 | 87.25 | 128.29 | 143.31 | 25.19% | 32 | 53282 |
| **Centered Compact (L=6, P=1)** | 0.41 MB | 64.40% | 24.70 | 24.40 | 41.51 | 49.36 | 6.59% | 6 | 217803 |
| **Centered Multi-Probe (L=6, P=8)** | 0.41 MB | 97.40% | 92.15 | 90.05 | 143.81 | 158.82 | 27.45% | 48 | 54977 |

### 4.4 Multi-Seed Variability ($L=6, K=8, P=8$ Centered across 5 Independent Seeds)
- **Seeds Evaluated:** `101, 202, 303, 404, 777`
- **Recall@10 (Mean $\pm$ SD):** **97.96% $\pm$ 0.49%**
- **Candidate Fraction (Mean $\pm$ SD):** **28.70% $\pm$ 0.74%**

### 4.5 Controlled Parameter Sweep ($K, L, P$ on Mean-Centered SIFT10K)

| $K$ (Bits) | $L$ (Tables) | $P$ (Probes) | Index RAM (MB) | Build (ms) | Recall@10 | Cand Frac (%) | ST Avg ($\mu$s) | OpenMP QPS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 8 | 6 | 1 | 0.41 | 21.67 | 64.40% | 6.59% | 75.64 | 112109 |
| 8 | 6 | 2 | 0.41 | 13.79 | 81.50% | 11.34% | 41.51 | 120393 |
| 8 | 6 | 4 | 0.41 | 12.66 | 92.30% | 18.29% | 121.87 | 76296 |
| 8 | 6 | 8 | 0.41 | 10.96 | 97.40% | 27.45% | 93.39 | 41945 |
| 8 | 6 | 16 | 0.41 | 15.75 | 99.40% | 38.89% | 155.52 | 28553 |
| 8 | 16 | 1 | 1.08 | 30.12 | 89.60% | 15.15% | 44.77 | 86627 |
| 8 | 32 | 1 | 2.17 | 93.62 | 98.00% | 25.19% | 89.28 | 52991 |
| 12 | 6 | 1 | 0.82 | 37.58 | 37.70% | 1.45% | 10.15 | 385804 |
| 12 | 6 | 8 | 0.82 | 15.27 | 81.20% | 7.12% | 39.80 | 135807 |
| 12 | 6 | 16 | 0.82 | 14.20 | 91.70% | 11.18% | 52.35 | 103168 |
| 12 | 16 | 4 | 2.20 | 41.44 | 90.70% | 9.83% | 82.92 | 84196 |
| 16 | 8 | 8 | 2.10 | 38.38 | 63.10% | 2.53% | 62.35 | 137508 |
| 16 | 16 | 8 | 4.01 | 57.95 | 80.40% | 4.99% | 128.36 | 72853 |

### 4.6 Corpus Scaling Study ($N = 10\text{K} \rightarrow 100\text{K}$ at $K=12, L=6, P=8$)

| Corpus Size ($N$) | Brute-Force ST ($\mu$s) | Multi-Probe ST ($\mu$s) | ST Latency Speedup | Recall@10 | Candidate Fraction |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 10000 | 157.21 $\mu$s | 58.25 $\mu$s | **2.70$\times$** | 81.60% | 7.04% |
| 25000 | 912.15 $\mu$s | 116.41 $\mu$s | **7.84$\times$** | 86.30% | 6.98% |
| 100000 | 5329.88 $\mu$s | 615.09 $\mu$s | **8.67$\times$** | 91.50% | 6.80% |

---

## 5. References

1. P. Indyk and R. Motwani, "Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality," *Proc. 30th ACM STOC*, pp. 604–613, 1998.
2. A. Gionis, P. Indyk, and R. Motwani, "Similarity Search in High Dimensions via Hashing," *Proc. 25th VLDB*, pp. 518–529, 1999.
3. M. S. Charikar, "Similarity Estimation Techniques from Rounding Algorithms," *Proc. 34th ACM STOC*, pp. 380–388, 2002.
4. M. X. Goemans and D. P. Williamson, "Improved Approximation Algorithms for Maximum Cut and Satisfiability Problems Using Semidefinite Programming," *Journal of the ACM*, vol. 42, no. 6, pp. 1115–1145, 1995.
5. Q. Lv, W. Josephson, Z. Wang, M. Charikar, and K. Li, "Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search," *Proc. 33rd VLDB*, pp. 950–961, 2007.
