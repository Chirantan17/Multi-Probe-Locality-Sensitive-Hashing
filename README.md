# High-Performance Multi-Probe LSH Vector Search Engine (CS G526)

An AVX2 SIMD-accelerated Approximate Nearest Neighbor (ANN) engine in C++20 implementing Random-Hyperplane LSH (SimHash) and Margin-Guided Multi-Probe LSH (Lv et al., 2007), evaluated on the real-world **INRIA SIFT10K** benchmark and scaled up to $N = 100,000$ vectors.

## Key Engineering & Algorithmic Highlights

- **True 32-Byte Heap Alignment:** Custom `aligned_allocator<float, 32>` guarantees 32-byte alignment for every vector allocation, enabling hardware-aligned `_mm256_load_ps` + `_mm256_fmadd_ps` kernels.
- **Zero-Allocation Hot Query Path:** Per-thread `QueryScratchpad` pre-allocates visited stamps, margins, candidate perturbations, and neighbor buffers outside the timed loop.
- **Hardened Visited-Stamp Rollover:** Monotonic 32-bit query tags reject `0` and automatically zero the stamp buffer upon `0xFFFFFFFF -> 1` wrap-around.
- **Mean-Centered Hyperplane Partitioning:** Eliminates SIFT's non-negative orthant skew, raising 8-bit bucket Shannon entropy from **5.45 bits** to **7.16 bits** and cutting candidate scans from **67.44%** to **27.45%**.

## Empirical Summary on INRIA SIFT10K ($K=8$, 5-Trial Medians)

| Configuration | Index RAM | Recall@10 | Single-Thread Avg ($\mu$s) | Candidate Fraction | OpenMP Batch QPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exact Brute-Force** | 0.00 MB | 100.00% | 170.47 $\mu$s | 100.00% | 14285 |
| **Uncentered Classic LSH ($L=32, P=1$)** | 2.12 MB | 100.00% | 220.72 $\mu$s | 51.14% | 12815 |
| **Uncentered Compact LSH ($L=6, P=1$)** | 0.39 MB | 82.90% | 77.98 $\mu$s | 22.11% | 39166 |
| **Uncentered Multi-Probe ($L=6, P=8$)** | 0.39 MB | **99.80%** | 344.12 $\mu$s | 67.44% | 12957 |
| **Centered Classic LSH ($L=32, P=1$)** | 2.17 MB | 98.00% | 124.68 $\mu$s | 25.19% | 32910 |
| **Centered Compact LSH ($L=6, P=1$)** | 0.41 MB | 64.40% | 24.61 $\mu$s | 6.59% | 129315 |
| **Centered Multi-Probe ($L=6, P=8$)** | 0.41 MB | **97.40%** | **99.28 $\mu$s** | **27.45%** | 47673 |

See [`report/project_report.md`](report/project_report.md) for full theoretical derivations, parameter sweeps, and scaling benchmarks up to $N = 100,000$.

## Build & Verify

```powershell
cmake -B build -S .
cmake --build build --config Release
ctest --test-dir build --output-on-failure
.\build\lsh_bench.exe
```
