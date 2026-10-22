#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <initializer_list>
#include <chrono>
#include <immintrin.h>

namespace lsh {

using VectorId = uint32_t;
using HashCode = uint64_t;

// 1. Memory-Aligned Vector
struct AlignedVector {
    alignas(32) std::vector<float> data;

    explicit AlignedVector(size_t dim = 0, float init = 0.0f) : data(dim, init) {}
    AlignedVector(std::initializer_list<float> list) : data(list) {}

    size_t size() const { return data.size(); }
    const float* raw() const { return data.data(); }
    float* raw() { return data.data(); }

    float operator[](size_t idx) const { return data[idx]; }
    float& operator[](size_t idx) { return data[idx]; }
};

struct Neighbor {
    VectorId id;
    float distance;

    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

struct QueryResult {
    std::vector<Neighbor> neighbors;
    size_t candidates_evaluated;
    size_t buckets_probed;
    double latency_us;
};

// 2. SIMD Vectorized Distance Kernels
namespace simd {

inline float dot_product(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    float result = 0.0f;
#if defined(__AVX2__)
    __m256 sum256 = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum256 = _mm256_fmadd_ps(va, vb, sum256);
    }
    __m128 lo = _mm256_castps256_ps128(sum256);
    __m128 hi = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    result = _mm_cvtss_f32(sum128);
#endif
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

inline float norm(const float* a, size_t dim) {
    return std::sqrt(std::max(0.0f, dot_product(a, a, dim)));
}

inline float cosine_distance(const AlignedVector& a, const AlignedVector& b) {
    size_t dim = a.size();
    float dot = dot_product(a.raw(), b.raw(), dim);
    float norm_a = norm(a.raw(), dim);
    float norm_b = norm(b.raw(), dim);
    if (norm_a <= 1e-8f || norm_b <= 1e-8f) return 1.0f;
    float sim = dot / (norm_a * norm_b);
    sim = std::clamp(sim, -1.0f, 1.0f);
    return 1.0f - sim;
}

} // namespace simd

// 3. Multi-Probe Margin Router
struct PerturbationCandidate {
    HashCode bucket_code;
    float cost;
    uint8_t radius;

    bool operator<(const PerturbationCandidate& other) const {
        return cost < other.cost;
    }
};

class MultiProbeRouter {
public:
    explicit MultiProbeRouter(size_t max_bits = 64) : max_bits_(max_bits) {}

    std::vector<HashCode> generate_probes(
        const AlignedVector& query,
        const std::vector<AlignedVector>& hyperplanes,
        size_t probe_budget) const 
    {
        size_t k = hyperplanes.size();
        if (k > 64 || k > max_bits_) {
            throw std::invalid_argument("Hyperplane count K exceeds supported 64-bit integer limit.");
        }
        if (probe_budget == 0) return {};

        struct BitMargin {
            size_t bit_index;
            float margin;
        };

        std::vector<BitMargin> margins(k);
        HashCode primary_hash = 0;

        for (size_t i = 0; i < k; ++i) {
            float proj = simd::dot_product(query.raw(), hyperplanes[i].raw(), query.size());
            if (proj >= 0.0f) {
                primary_hash |= (1ULL << i);
            }
            margins[i] = {i, std::abs(proj)};
        }

        std::vector<HashCode> selected_probes;
        selected_probes.reserve(probe_budget);
        selected_probes.push_back(primary_hash);

        if (probe_budget == 1) return selected_probes;

        std::sort(margins.begin(), margins.end(), [](const BitMargin& a, const BitMargin& b) {
            return a.margin < b.margin;
        });

        std::vector<PerturbationCandidate> candidates;

        // Radius 1: Single bit flips
        for (size_t i = 0; i < k; ++i) {
            HashCode perturbed = primary_hash ^ (1ULL << margins[i].bit_index);
            candidates.push_back({perturbed, margins[i].margin, 1});
        }

        // Radius 2: Pairwise bit flips
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = i + 1; j < k; ++j) {
                HashCode perturbed = primary_hash ^ (1ULL << margins[i].bit_index) ^ (1ULL << margins[j].bit_index);
                float combined_cost = margins[i].margin + margins[j].margin;
                candidates.push_back({perturbed, combined_cost, 2});
            }
        }

        std::sort(candidates.begin(), candidates.end());

        for (const auto& cand : candidates) {
            if (selected_probes.size() >= probe_budget) break;
            if (std::find(selected_probes.begin(), selected_probes.end(), cand.bucket_code) == selected_probes.end()) {
                selected_probes.push_back(cand.bucket_code);
            }
        }

        return selected_probes;
    }

    static std::vector<HashCode> get_probe_buckets(
        const AlignedVector& query,
        const std::vector<AlignedVector>& hyperplanes,
        size_t probe_budget) 
    {
        MultiProbeRouter router;
        return router.generate_probes(query, hyperplanes, probe_budget);
    }

    static std::vector<HashCode> get_probe_buckets(
        HashCode /*base_hash*/,
        const AlignedVector& query,
        const std::vector<AlignedVector>& hyperplanes,
        size_t probe_budget) 
    {
        MultiProbeRouter router;
        return router.generate_probes(query, hyperplanes, probe_budget);
    }

private:
    size_t max_bits_;
};

// 4. LSH Index Engine
class LSHIndex {
public:
    LSHIndex(size_t dim, size_t num_tables, size_t num_bits, uint32_t seed = 42)
        : dim_(dim), L_(num_tables), K_(num_bits)
    {
        if (K_ > 64) throw std::invalid_argument("K cannot exceed 64 bits.");
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        hyperplanes_.resize(L_);
        tables_.resize(L_);

        for (size_t l = 0; l < L_; ++l) {
            hyperplanes_[l].resize(K_);
            for (size_t k = 0; k < K_; ++k) {
                hyperplanes_[l][k] = AlignedVector(dim_);
                float sq_sum = 0.0f;
                for (size_t d = 0; d < dim_; ++d) {
                    float val = dist(rng);
                    hyperplanes_[l][k][d] = val;
                    sq_sum += val * val;
                }
                float inv_norm = 1.0f / (std::sqrt(sq_sum) + 1e-8f);
                for (size_t d = 0; d < dim_; ++d) {
                    hyperplanes_[l][k][d] *= inv_norm;
                }
            }
        }
    }

    HashCode hash_vector(size_t table_idx, const AlignedVector& vec) const {
        HashCode h = 0;
        for (size_t k = 0; k < K_; ++k) {
            float proj = simd::dot_product(vec.raw(), hyperplanes_[table_idx][k].raw(), dim_);
            if (proj >= 0.0f) {
                h |= (1ULL << k);
            }
        }
        return h;
    }

    void insert(VectorId id, const AlignedVector& vec) {
        for (size_t l = 0; l < L_; ++l) {
            HashCode h = hash_vector(l, vec);
            tables_[l][h].push_back(id);
        }
    }

    void build(const std::vector<AlignedVector>& dataset) {
        for (VectorId id = 0; id < static_cast<VectorId>(dataset.size()); ++id) {
            insert(id, dataset[id]);
        }
    }

    QueryResult query(
        const AlignedVector& q,
        const std::vector<AlignedVector>& dataset,
        size_t top_k,
        size_t probes_per_table = 1) const
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        std::unordered_set<VectorId> candidate_set;
        candidate_set.reserve(1024);
        size_t total_buckets = 0;

        for (size_t l = 0; l < L_; ++l) {
            std::vector<HashCode> probes = MultiProbeRouter::get_probe_buckets(q, hyperplanes_[l], probes_per_table);
            total_buckets += probes.size();

            for (HashCode code : probes) {
                auto it = tables_[l].find(code);
                if (it != tables_[l].end()) {
                    for (VectorId id : it->second) {
                        candidate_set.insert(id);
                    }
                }
            }
        }

        std::vector<Neighbor> candidates;
        candidates.reserve(candidate_set.size());

        for (VectorId id : candidate_set) {
            float dist = simd::cosine_distance(q, dataset[id]);
            candidates.push_back({id, dist});
        }

        size_t k = std::min(top_k, candidates.size());
        if (k > 0) {
            std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end());
            candidates.resize(k);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double lat_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

        return {candidates, candidate_set.size(), total_buckets, lat_us};
    }

    size_t memory_bytes() const {
        size_t bytes = 0;
        for (const auto& table : tables_) {
            bytes += table.size() * sizeof(HashCode);
            for (const auto& kv : table) {
                bytes += kv.second.capacity() * sizeof(VectorId);
            }
        }
        return bytes;
    }

    size_t L() const { return L_; }
    size_t K() const { return K_; }
    size_t dim() const { return dim_; }
    const std::vector<AlignedVector>& get_hyperplanes(size_t l) const { return hyperplanes_[l]; }

private:
    size_t dim_;
    size_t L_;
    size_t K_;
    std::vector<std::vector<AlignedVector>> hyperplanes_;
    std::vector<std::unordered_map<HashCode, std::vector<VectorId>>> tables_;
};

} // namespace lsh