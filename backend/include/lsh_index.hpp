#pragma once
#include "types.hpp"
#include "simd_metric.hpp"
#include <vector>
#include <unordered_map>
#include <random>
#include <cmath>

namespace lsh {

class LSHIndex {
public:
    size_t num_tables;      // L
    size_t hash_bits;       // K (number of hyperplanes per table)
    size_t dimension;       // d

    // hyperplanes[table_idx][bit_idx] -> AlignedVector of size 'dimension'
    std::vector<std::vector<AlignedVector>> hyperplanes;

    // tables[table_idx][hash_code] -> list of VectorId
    std::vector<std::unordered_map<HashCode, std::vector<VectorId>>> tables;

    LSHIndex(size_t L, size_t K, size_t dim, uint32_t seed = 42)
        : num_tables(L), hash_bits(K), dimension(dim) {
        
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal_dist(0.0f, 1.0f);

        hyperplanes.resize(num_tables);
        tables.resize(num_tables);

        for (size_t l = 0; l < num_tables; ++l) {
            hyperplanes[l].resize(hash_bits);
            for (size_t k = 0; k < hash_bits; ++k) {
                hyperplanes[l][k] = AlignedVector(dimension);
                float norm_sq = 0.0f;
                for (size_t d = 0; d < dimension; ++d) {
                    float val = normal_dist(rng);
                    hyperplanes[l][k][d] = val;
                    norm_sq += val * val;
                }
                // Normalize hyperplane normal vector
                float inv_norm = 1.0f / std::sqrt(norm_sq);
                for (size_t d = 0; d < dimension; ++d) {
                    hyperplanes[l][k][d] *= inv_norm;
                }
            }
        }
    }

    HashCode hash_vector(size_t table_idx, const float* vec_data) const {
        HashCode code = 0;
        for (size_t k = 0; k < hash_bits; ++k) {
            float projection = simd::dot_product(hyperplanes[table_idx][k].raw(), vec_data, dimension);
            if (projection >= 0.0f) {
                code |= (1ULL << k);
            }
        }
        return code;
    }

    void insert(VectorId id, const float* vec_data) {
        for (size_t l = 0; l < num_tables; ++l) {
            HashCode code = hash_vector(l, vec_data);
            tables[l][code].push_back(id);
        }
    }
};

} // namespace lsh