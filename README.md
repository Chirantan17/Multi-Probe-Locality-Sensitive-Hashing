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
| **Exact Brute-Force** | 0.00 MB | 100.00% | 180.87 $\mu$s | 100.00% | 33333 |
| **Uncentered Classic LSH ($L=32, P=1$)** | 2.12 MB | 100.00% | 274.46 $\mu$s | 51.14% | 27180 |
| **Uncentered Compact LSH ($L=6, P=1$)** | 0.39 MB | 82.90% | 164.75 $\mu$s | 22.11% | 59068 |
| **Uncentered Multi-Probe ($L=6, P=8$)** | 0.39 MB | **99.80%** | 381.74 $\mu$s | 67.44% | 23559 |
| **Centered Classic LSH ($L=32, P=1$)** | 2.17 MB | 98.00% | 88.01 $\mu$s | 25.19% | 53282 |
| **Centered Compact LSH ($L=6, P=1$)** | 0.41 MB | 64.40% | 24.70 $\mu$s | 6.59% | 217803 |
| **Centered Multi-Probe ($L=6, P=8$)** | 0.41 MB | **97.40%** | **92.15 $\mu$s** | **27.45%** | 54977 |

See [`report/project_report.md`](report/project_report.md) for full theoretical derivations, parameter sweeps, and scaling benchmarks up to $N = 100,000$.

## Build & Verify

```powershell
cmake -B build -S .
cmake --build build --config Release
ctest --test-dir build --output-on-failure
.\build\lsh_bench.exe
```
