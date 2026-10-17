#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>

namespace lsh {

using VectorId = uint32_t;
using HashCode = uint64_t;

struct AlignedVector {
    alignas(32) std::vector<float> data;

    explicit AlignedVector(size_t dim = 0, float init = 0.0f) : data(dim, init) {}
    
    size_t size() const { return data.size(); }
    const float* raw() const { return data.data(); }
    float* raw() { return data.data(); }
    
    float& operator[](size_t i) { return data[i]; }
    const float& operator[](size_t i) const { return data[i]; }
};

struct Neighbor {
    VectorId id;
    float distance; // Cosine distance: 1.0 - similarity

    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

struct QueryResult {
    std::vector<Neighbor> neighbors;
    size_t candidates_evaluated = 0;
    size_t buckets_probed = 0;
    double search_latency_us = 0.0;
};

} // namespace lsh