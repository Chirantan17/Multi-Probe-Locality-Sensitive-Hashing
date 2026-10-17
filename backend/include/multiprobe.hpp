#pragma once
#include "types.hpp"
#include "lsh_index.hpp"
#include "simd_metric.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace lsh {

struct BitPerturbation {
    size_t bit_index;
    float margin; // Distance to hyperplane boundary

    bool operator<(const BitPerturbation& other) const {
        return margin < other.margin; // Smaller margin = higher priority to flip
    }
};

class MultiProbeRouter {
public:
    // Generate buckets to probe: primary bucket + perturbation buckets (Hamming radius 1 and 2)
    static std::vector<HashCode> get_probe_buckets(
        const LSHIndex& index,
        size_t table_idx,
        const float* query_data,
        size_t max_probes) {

        std::vector<HashCode> buckets;
        HashCode base_code = 0;

        std::vector<BitPerturbation> perturbations(index.hash_bits);

        for (size_t k = 0; k < index.hash_bits; ++k) {
            float projection = simd::dot_product(index.hyperplanes[table_idx][k].raw(), query_data, index.dimension);
            if (projection >= 0.0f) {
                base_code |= (1ULL << k);
            }
            perturbations[k] = { k, std::abs(projection) };
        }

        // Primary bucket (Hamming distance 0)
        buckets.push_back(base_code);
        if (max_probes <= 1) return buckets;

        // Sort bits by margin ascending (least certain bits first)
        std::sort(perturbations.begin(), perturbations.end());

        // Radius 1 probes: flip single least-certain bits
        for (size_t i = 0; i < perturbations.size() && buckets.size() < max_probes; ++i) {
            size_t bit = perturbations[i].bit_index;
            buckets.push_back(base_code ^ (1ULL << bit));
        }

        // Radius 2 probes: flip pair of least-certain bits
        for (size_t i = 0; i < perturbations.size() && buckets.size() < max_probes; ++i) {
            for (size_t j = i + 1; j < perturbations.size() && buckets.size() < max_probes; ++j) {
                size_t bit_i = perturbations[i].bit_index;
                size_t bit_j = perturbations[j].bit_index;
                buckets.push_back(base_code ^ (1ULL << bit_i) ^ (1ULL << bit_j));
            }
        }

        return buckets;
    }
};

} // namespace lsh
