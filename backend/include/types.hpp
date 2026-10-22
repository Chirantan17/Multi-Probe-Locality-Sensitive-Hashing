#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <initializer_list>

namespace lsh {

using VectorId = uint32_t;
using HashCode = uint64_t;

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

} // namespace lsh