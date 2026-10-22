#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include "types.hpp"

namespace lsh {

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
            float proj = 0.0f;
            for (size_t d = 0; d < query.size(); ++d) {
                proj += query[d] * hyperplanes[i][d];
            }

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

        // Radius 2: Pairwise bit flips sorted by combined margin cost
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

    // Static overload expected by main.cpp
    static std::vector<HashCode> get_probe_buckets(
        const AlignedVector& query,
        const std::vector<AlignedVector>& hyperplanes,
        size_t probe_budget)
    {
        MultiProbeRouter router;
        return router.generate_probes(query, hyperplanes, probe_budget);
    }

    // Secondary static overload if base_hash is passed
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

} // namespace lsh