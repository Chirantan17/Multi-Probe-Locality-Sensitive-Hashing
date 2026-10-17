#pragma once
#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace lsh::simd {

inline float dot_product(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    __m256 sum256 = _mm256_setzero_ps();

    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum256 = _mm256_fmadd_ps(va, vb, sum256);
    }

    alignas(32) float buffer[8];
    _mm256_store_ps(buffer, sum256);
    float total = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                  buffer[4] + buffer[5] + buffer[6] + buffer[7];

    for (; i < dim; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

inline float cosine_distance(const float* a, const float* b, size_t dim) {
    float dot = dot_product(a, b, dim);
    float norm_a = dot_product(a, a, dim);
    float norm_b = dot_product(b, b, dim);

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-9f) return 1.0f;

    float sim = dot / denom;
    sim = std::clamp(sim, -1.0f, 1.0f);
    return 1.0f - sim;
}

} // namespace lsh::simd