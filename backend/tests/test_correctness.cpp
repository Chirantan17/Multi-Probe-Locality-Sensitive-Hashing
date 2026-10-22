#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "types.hpp"
#include "simd_metric.hpp"
#include "multiprobe.hpp"

void test_metrics() {
    std::cout << "[Test 1/3] Testing SIMD Cosine Distance & Dot Products...\n";
    lsh::AlignedVector a(128, 0.0f);
    lsh::AlignedVector b(128, 0.0f);
    a[0] = 1.0f;
    b[0] = 1.0f;

    // Identical unit vectors -> Cosine distance ~ 0.0
    float d_identical = lsh::simd::cosine_distance(a, b);
    assert(std::abs(d_identical) < 1e-4f);

    // Opposite unit vectors -> Cosine distance ~ 2.0
    b[0] = -1.0f;
    float d_opposite = lsh::simd::cosine_distance(a, b);
    assert(std::abs(d_opposite - 2.0f) < 1e-4f);

    // Orthogonal unit vectors -> Cosine distance ~ 1.0
    b[0] = 0.0f;
    b[1] = 1.0f;
    float d_orthogonal = lsh::simd::cosine_distance(a, b);
    assert(std::abs(d_orthogonal - 1.0f) < 1e-4f);

    std::cout << "  --> Passed!\n";
}

void test_multiprobe_ordering() {
    std::cout << "[Test 2/3] Testing Multi-Probe Cost-Based Ordering...\n";
    lsh::MultiProbeRouter router;

    lsh::AlignedVector query = {1.0f, 0.0f, 0.0f};
    std::vector<lsh::AlignedVector> planes = {
        {0.05f, 0.0f, 0.0f},
        {0.10f, 0.0f, 0.0f},
        {0.90f, 0.0f, 0.0f}
    };

    auto probes = router.generate_probes(query, planes, 4);

    assert(probes.size() == 4);
    // Primary bucket is first
    assert(probes[0] == 0b111);
    // Next lowest cost is flipping bit 0 (cost 0.05)
    assert(probes[1] == 0b110);
    // Next lowest is flipping bit 1 (cost 0.10)
    assert(probes[2] == 0b101);
    // Next lowest is flipping bit 0 and 1 together (cost 0.05 + 0.10 = 0.15 < 0.90)
    assert(probes[3] == 0b100);

    std::cout << "  --> Passed! Multi-probe correctly ranked radius-2 before high-margin radius-1.\n";
}

void test_probe_budget_and_uniqueness() {
    std::cout << "[Test 3/3] Testing Probe Uniqueness & Budget...\n";
    lsh::MultiProbeRouter router;
    lsh::AlignedVector query = {0.5f, 0.2f};
    std::vector<lsh::AlignedVector> planes = {{1.0f, 0.0f}, {0.0f, 1.0f}};

    // Budget = 1 -> Primary bucket only
    auto p1 = router.generate_probes(query, planes, 1);
    assert(p1.size() == 1);

    // Large budget -> Cannot exceed total unique combinations (2^2 = 4)
    auto p_all = router.generate_probes(query, planes, 16);
    assert(p_all.size() == 4);

    std::cout << "  --> Passed!\n";
}

int main() {
    std::cout << "=== Running Multi-Probe LSH Correctness Tests ===\n";
    test_metrics();
    test_multiprobe_ordering();
    test_probe_budget_and_uniqueness();
    std::cout << "=== All Unit Tests Passed Successfully! ===\n";
    return 0;
}
