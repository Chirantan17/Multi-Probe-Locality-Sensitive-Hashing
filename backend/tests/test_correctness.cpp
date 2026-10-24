#include "lsh_engine.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>

#define TEST_CHECK(cond, msg) \
    do { if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; std::exit(1); } } while(0)

using namespace lsh;

void test_simd_vs_scalar() {
    AlignedVector a(131, 0.1f); 
    AlignedVector b(131, 0.2f);
    a[130] = 0.9f; b[130] = 0.5f;
    float simd_dist = simd::cosine_distance_unit(a, b);
    float scal_dot = simd::scalar_dot_product(a.raw(), b.raw(), 131);
    float scal_dist = 1.0f - std::clamp(scal_dot, -1.0f, 1.0f);
    TEST_CHECK(std::abs(simd_dist - scal_dist) < 1e-5f, "SIMD and Scalar distance mismatch.");
    std::cout << "[Test 1/5] SIMD Tail Vector Agreement Passed.\n";
}

void test_dimension_mismatch() {
    LSHIndex idx(128, 1, 8);
    AlignedVector bad_vec(120);
    bool caught = false;
    try { idx.hash_vector(0, bad_vec); } catch (const std::invalid_argument&) { caught = true; }
    TEST_CHECK(caught, "Failed to catch dimension mismatch.");
    
    bool caught_k = false;
    try { LSHIndex idx_bad(128, 1, 65); } catch (const std::invalid_argument&) { caught_k = true; }
    TEST_CHECK(caught_k, "Failed to catch K > 64 constructor violation.");
    std::cout << "[Test 2/5] Dimension & K Bounds Exceptions Passed.\n";
}

void test_visited_stamp_reuse() {
    LSHIndex idx(128, 2, 4, 42); // 2 tables, K=4
    AlignedVector v(128, 0.1f);
    
    // Dataset with duplicate vectors to guarantee they map to identical buckets
    std::vector<AlignedVector> ds = {v, v};
    idx.build(ds);
    std::vector<uint32_t> visited(ds.size(), 0);
    
    // Tag = 1: Despite 2 tables, engine should dedupe and evaluate exactly 2 unique candidate IDs
    auto res1 = idx.query_fast(v, ds, 10, 1, visited, 1);
    TEST_CHECK(res1.candidates_evaluated == 2, "Failed to evaluate exactly 2 unique candidates.");
    
    // Tag = 2: Ensure engine successfully resets state against the identical array
    auto res2 = idx.query_fast(v, ds, 10, 1, visited, 2);
    TEST_CHECK(res2.candidates_evaluated == 2, "Failed to reuse visited array on next tag.");
    
    // Wrap-around Vulnerability Test (Testing near UINT32_MAX)
    visited[0] = 0xFFFFFFFE; visited[1] = 0xFFFFFFFE;
    auto res3 = idx.query_fast(v, ds, 10, 1, visited, 0xFFFFFFFF);
    TEST_CHECK(res3.candidates_evaluated == 2, "Failed deduplication near UINT32_MAX boundary.");
    std::cout << "[Test 3/5] Engine Visited-Stamp Deduplication Passed.\n";
}

void test_multiprobe_ordering() {
    std::vector<AlignedVector> hyperplanes(4, AlignedVector(4));
    hyperplanes[0] = {1.0f, 0.0f, 0.0f, 0.0f}; hyperplanes[1] = {0.0f, 1.0f, 0.0f, 0.0f};
    hyperplanes[2] = {0.0f, 0.0f, 1.0f, 0.0f}; hyperplanes[3] = {0.0f, 0.0f, 0.0f, 1.0f};
    AlignedVector query = {0.9f, 0.1f, 0.2f, 0.3f};
    auto probes = MultiProbeRouter::get_probe_buckets(query, hyperplanes, 3);
    TEST_CHECK(probes.size() == 3, "Probe budget mismatch.");
    TEST_CHECK(probes[0] == 15, "Primary hash failure.");
    TEST_CHECK(probes[1] == (15 ^ (1ULL << 1)), "Margin perturbation ranking failure.");
    std::cout << "[Test 4/5] Multi-Probe Margin Perturbation Passed.\n";
}

void test_minhash_jaccard() {
    MinHash minhash(64, 42);
    auto sig1 = minhash.compute_signature({1, 2, 3, 4, 5, 6, 7, 8});
    auto sig2 = minhash.compute_signature({1, 2, 3, 4, 9, 10, 11, 12});
    float est_sim = MinHash::jaccard_similarity(sig1, sig2);
    TEST_CHECK(est_sim > 0.15f && est_sim < 0.55f, "MinHash Jaccard bounds failed.");
    std::cout << "[Test 5/5] MinHash Set Shingling Passed.\n";
}

int main() {
    std::cout << "=== Running Production Correctness Suite (NDEBUG Proof) ===\n";
    test_simd_vs_scalar(); test_dimension_mismatch(); test_visited_stamp_reuse();
    test_multiprobe_ordering(); test_minhash_jaccard();
    std::cout << "=== All Unit Tests Passed Successfully! ===\n";
    return 0;
}
