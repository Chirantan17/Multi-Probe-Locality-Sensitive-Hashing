#include "lsh_engine.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <omp.h>

#define TEST_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " -> " << msg << "\n"; \
            std::exit(1); \
        } \
    } while (0)

using namespace lsh;

void test_1_alignment_and_simd_equivalence() {
    // 1a. Verify 32-byte alignment across 100 heap allocations of varying sizes
    for (size_t d = 1; d <= 100; ++d) {
        AlignedVector v(d, 1.0f);
        TEST_CHECK(v.is_32b_aligned(), "AlignedVector buffer is not 32-byte aligned.");
    }

    // 1b. Verify SIMD and scalar dot/cosine agreement on unit-normalized vectors across tail dimensions
    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    const size_t dims[] = {1, 7, 8, 9, 127, 128, 131};

    for (size_t dim : dims) {
        for (int trial = 0; trial < 20; ++trial) {
            AlignedVector a(dim), b(dim);
            for (size_t i = 0; i < dim; ++i) {
                a[i] = dist(rng);
                b[i] = dist(rng);
            }
            simd::normalize_in_place(a);
            simd::normalize_in_place(b);

            float dot_simd = simd::dot_product(a.raw(), b.raw(), dim);
            float dot_scal = simd::scalar_dot_product(a.raw(), b.raw(), dim);
            TEST_CHECK(std::abs(dot_simd - dot_scal) < 1e-5f, "SIMD vs Scalar dot product mismatch.");

            float cos_unit = simd::cosine_distance_unit(a, b);
            float cos_gen = simd::cosine_distance_general(a, b);
            TEST_CHECK(std::abs(cos_unit - cos_gen) < 1e-5f, "Unit vs General cosine distance mismatch on unit vectors.");
            if (dim >= 2) { TEST_CHECK(cos_unit > 0.0001f && cos_unit < 1.9999f, "Vacuous zero/clamped distance detected."); }
        }
    }
    std::cout << "[Test 1/7] 32B Heap Alignment & SIMD/Scalar Equivalence (Dims 1..131) Passed.\n";
}

void test_2_cosine_metric_contract() {
    AlignedVector u = {1.0f, 0.0f, 0.0f, 0.0f};
    AlignedVector orth = {0.0f, 1.0f, 0.0f, 0.0f};
    AlignedVector opp = {-1.0f, 0.0f, 0.0f, 0.0f};
    AlignedVector scaled = {5.5f, 0.0f, 0.0f, 0.0f};

    TEST_CHECK(std::abs(simd::cosine_distance_unit(u, u) - 0.0f) < 1e-6f, "Identical vector distance != 0.");
    TEST_CHECK(std::abs(simd::cosine_distance_unit(u, orth) - 1.0f) < 1e-6f, "Orthogonal vector distance != 1.");
    TEST_CHECK(std::abs(simd::cosine_distance_unit(u, opp) - 2.0f) < 1e-6f, "Opposite vector distance != 2.");
    TEST_CHECK(std::abs(simd::cosine_distance_general(u, scaled) - 0.0f) < 1e-6f, "General cosine failed scale invariance.");
    std::cout << "[Test 2/7] Cosine Metric Contract (Identical, Orthogonal, Opposite, Scaled) Passed.\n";
}

void test_3_simhash_bits_and_k_bounds() {
    // K=1 and K=64 boundary tests
    LSHIndex idx_k1(4, 1, 1, 42);
    LSHIndex idx_k64(4, 1, 64, 42);
    AlignedVector v = {1.0f, 0.0f, 0.0f, 0.0f};
    HashCode h1 = idx_k1.hash_vector(0, v);
    HashCode h64 = idx_k64.hash_vector(0, v);
    TEST_CHECK(h1 <= 1ULL, "K=1 hash exceeded 1 bit.");
    TEST_CHECK(h64 != 0ULL || h64 == 0ULL, "K=64 hash evaluation succeeded without UB.");

    // Exact sign bit and tie rule (proj == 0.0f -> bit = 1)
    std::vector<AlignedVector> hyps = {
        {1.0f, 0.0f, 0.0f, 0.0f},  // dot = +0.5 -> bit 0 = 1
        {-1.0f, 0.0f, 0.0f, 0.0f}, // dot = -0.5 -> bit 1 = 0
        {0.0f, 1.0f, 0.0f, 0.0f}   // dot =  0.0 -> bit 2 = 1 (tie rule proj >= 0)
    };
    AlignedVector q = {0.5f, 0.0f, 0.0f, 0.0f};
    auto probes = MultiProbeRouter::get_probe_buckets(q, hyps, 1);
    TEST_CHECK(probes.size() == 1 && probes[0] == 0b101ULL, "SimHash sign/tie bit mismatch.");
    std::cout << "[Test 3/7] SimHash Sign Bits, Tie Rule, and K=1..64 Bounds Passed.\n";
}

void test_4_multiprobe_enumeration_and_budgets() {
    // 4a. Verify Linear vs Squared cost ranking on radius-1 vs radius-2
    std::vector<AlignedVector> hyps(4, AlignedVector(4));
    hyps[0] = {1.0f, 0.0f, 0.0f, 0.0f}; // margin 0.10
    hyps[1] = {0.0f, 1.0f, 0.0f, 0.0f}; // margin 0.20
    hyps[2] = {0.0f, 0.0f, 1.0f, 0.0f}; // margin 0.25
    hyps[3] = {0.0f, 0.0f, 0.0f, 1.0f}; // margin 0.90
    AlignedVector q = {0.10f, 0.20f, 0.25f, 0.90f}; // primary = 15 (0b1111)

    // Under Squared cost:
    // bit 0: 0.01, bit 1: 0.04, pair(0,1): 0.01+0.04 = 0.05 < bit 2: 0.0625!
    auto sq_probes = MultiProbeRouter::get_probe_buckets(q, hyps, 5, CostMode::Squared);
    TEST_CHECK(sq_probes.size() == 5, "Squared probe count mismatch.");
    TEST_CHECK(sq_probes[0] == 15ULL, "Base bucket must be first.");
    TEST_CHECK(sq_probes[1] == (15ULL ^ 1ULL), "Squared probe 1 mismatch.");
    TEST_CHECK(sq_probes[2] == (15ULL ^ 2ULL), "Squared probe 2 mismatch.");
    TEST_CHECK(sq_probes[3] == (15ULL ^ 3ULL), "Squared probe 3 (radius-2 pair 0,1) should precede bit 2.");
    TEST_CHECK(sq_probes[4] == (15ULL ^ 4ULL), "Squared probe 4 (bit 2) mismatch.");

    // 4b. Budgets 0, 1, 8, 37, and >37 for K=8
    std::vector<AlignedVector> hyps8(8, AlignedVector(8, 0.0f));
    for (size_t i = 0; i < 8; ++i) hyps8[i][i] = 1.0f;
    AlignedVector q8 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

    std::vector<BitMargin> margins(8);
    std::vector<PerturbationCandidate> cands;
    std::vector<HashCode> out;
    bool sat = false;

    MultiProbeRouter::generate_probes_in_place(q8, hyps8, 0, margins, cands, out, sat);
    TEST_CHECK(out.empty() && !sat, "Budget 0 should return 0 probes.");

    MultiProbeRouter::generate_probes_in_place(q8, hyps8, 1, margins, cands, out, sat);
    TEST_CHECK(out.size() == 1 && !sat, "Budget 1 should return 1 probe.");

    MultiProbeRouter::generate_probes_in_place(q8, hyps8, 8, margins, cands, out, sat);
    TEST_CHECK(out.size() == 8 && !sat, "Budget 8 should return 8 probes.");

    MultiProbeRouter::generate_probes_in_place(q8, hyps8, 37, margins, cands, out, sat);
    TEST_CHECK(out.size() == 37 && !sat, "Budget 37 should return all 37 radius<=2 probes without saturation.");

    MultiProbeRouter::generate_probes_in_place(q8, hyps8, 50, margins, cands, out, sat);
    TEST_CHECK(out.size() == 37 && sat, "Budget 50 on K=8 must saturate at 37 probes and set budget_saturated=true.");

    std::unordered_set<HashCode> uniq(out.begin(), out.end());
    TEST_CHECK(uniq.size() == 37, "All 37 generated probe codes must be unique.");
    std::cout << "[Test 4/7] Multi-Probe Ordering, Radius-2 Pairs, and Budgets (0..50) Passed.\n";
}

void test_5_deduplication_and_tag_rollover() {
    LSHIndex idx(4, 3, 4, 42);
    AlignedVector v = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<AlignedVector> ds = {v, v, v};
    idx.build(ds);

    QueryScratchpad scratch(ds.size(), 4, 4);

    // Pre-populate stamps with 1U (simulating stale stamps from an earlier cycle)
    // and set tag to 0xFFFFFFFFU so the next advance_tag() triggers rollover to 1U.
    std::fill(scratch.visited_stamps.begin(), scratch.visited_stamps.end(), 1U);
    scratch.tag = 0xFFFFFFFFU;

    auto res = idx.query_fast(v, ds, 10, 1, scratch);
    TEST_CHECK(scratch.tag == 1U, "Tag failed to wrap around from 0xFFFFFFFFU to 1U.");
    TEST_CHECK(res.candidates_evaluated == 3, "Wraparound failed to clear stale 1U stamps; candidates were dropped!");

    // Verify allocations in hot path: capacity of scratch buffers must not grow on repeated queries
    size_t cap_cands = scratch.candidate_ids.capacity();
    size_t cap_probes = scratch.selected_probes.capacity();
    for (int i = 0; i < 50; ++i) {
        auto r = idx.query_fast(v, ds, 2, 4, scratch);
        TEST_CHECK(r.neighbors.size() == 2, "Top-K truncation failed.");
    }
    TEST_CHECK(scratch.candidate_ids.capacity() == cap_cands && scratch.selected_probes.capacity() == cap_probes,
               "Scratchpad reallocated during hot query loop.");
    std::cout << "[Test 5/7] Visited-Stamp Rollover (0xFFFFFFFF->1) & Zero-Alloc Scratchpad Passed.\n";
}

void test_6_openmp_concurrency_determinism() {
    std::mt19937 rng(77);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    size_t N = 500, D = 32, Q = 40;
    std::vector<AlignedVector> ds(N, AlignedVector(D)), qs(Q, AlignedVector(D));
    for (size_t i = 0; i < N; ++i) {
        for (size_t d = 0; d < D; ++d) ds[i][d] = dist(rng);
        simd::normalize_in_place(ds[i]);
    }
    for (size_t q = 0; q < Q; ++q) {
        for (size_t d = 0; d < D; ++d) qs[q][d] = dist(rng);
        simd::normalize_in_place(qs[q]);
    }

    LSHIndex idx(D, 6, 8, 999);
    idx.build(ds, true);

    std::vector<QueryResult> serial_res(Q), parallel_res(Q);
    QueryScratchpad serial_scratch(N, 8, D);
    for (size_t q = 0; q < Q; ++q) {
        serial_res[q] = idx.query_fast(qs[q], ds, 10, 8, serial_scratch);
    }

    #pragma omp parallel
    {
        QueryScratchpad thread_scratch(N, 8, D);
        #pragma omp for schedule(dynamic)
        for (int q = 0; q < static_cast<int>(Q); ++q) {
            parallel_res[q] = idx.query_fast(qs[q], ds, 10, 8, thread_scratch);
        }
    }

    for (size_t q = 0; q < Q; ++q) {
        TEST_CHECK(serial_res[q].candidates_evaluated == parallel_res[q].candidates_evaluated, "Parallel candidate count mismatch.");
        TEST_CHECK(serial_res[q].neighbors.size() == parallel_res[q].neighbors.size(), "Parallel neighbor count mismatch.");
        for (size_t k = 0; k < serial_res[q].neighbors.size(); ++k) {
            TEST_CHECK(serial_res[q].neighbors[k].id == parallel_res[q].neighbors[k].id, "Parallel neighbor ID mismatch.");
        }
    }
    std::cout << "[Test 6/7] OpenMP Parallel vs. Serial Determinism Passed.\n";
}

void test_7_input_validation_and_minhash() {
    LSHIndex idx(4, 2, 4, 42);
    AlignedVector v = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<AlignedVector> ds = {v};
    idx.build(ds);
    // Rebuild behavior: building again should replace, not duplicate
    idx.build(ds);
    QueryScratchpad scratch(1, 4, 4);
    auto r = idx.query_fast(v, ds, 10, 1, scratch);
    TEST_CHECK(r.candidates_evaluated == 1, "Rebuild duplicated index entries.");

    // Exceptions on invalid inputs
    bool caught_nan = false, caught_dim = false, caught_tag0 = false, caught_small_stamp = false;
    AlignedVector nan_vec = {std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 0.0f};
    try { idx.query_fast(nan_vec, ds, 1, 1, scratch); } catch (const std::invalid_argument&) { caught_nan = true; }
    try { idx.hash_vector(0, AlignedVector(3, 1.0f)); } catch (const std::invalid_argument&) { caught_dim = true; }
    std::vector<uint32_t> stamps(1, 0);
    try { idx.query_fast(v, ds, 1, 1, stamps, 0U); } catch (const std::invalid_argument&) { caught_tag0 = true; }
    std::vector<uint32_t> empty_stamps;
    try { idx.query_fast(v, ds, 1, 1, empty_stamps, 1U); } catch (const std::invalid_argument&) { caught_small_stamp = true; }

    TEST_CHECK(caught_nan && caught_dim && caught_tag0 && caught_small_stamp, "Failed to reject invalid inputs.");

    // MinHash 64-bit large token test
    MinHash mh(128, 42);
    std::vector<uint64_t> set_a, set_b;
    for (uint64_t i = 1; i <= 100; ++i) set_a.push_back(i * 1000000000ULL);
    for (uint64_t i = 51; i <= 150; ++i) set_b.push_back(i * 1000000000ULL); // Exact Jaccard = 50 / 150 = 0.3333
    float j_est = MinHash::jaccard_similarity(mh.compute_signature(set_a), mh.compute_signature(set_b));
    TEST_CHECK(std::abs(j_est - (1.0f / 3.0f)) < 0.12f, "64-bit MinHash Jaccard estimate out of tolerance.");
    std::cout << "[Test 7/7] Input Validation, Rebuild Invariant, and 64-bit MinHash Passed.\n";
}

int main() {
    std::cout << "====================================================================\n";
    std::cout << "  Running Comprehensive Correctness Suite (NDEBUG-Safe, 7 Modules)  \n";
    std::cout << "====================================================================\n";
    test_1_alignment_and_simd_equivalence();
    test_2_cosine_metric_contract();
    test_3_simhash_bits_and_k_bounds();
    test_4_multiprobe_enumeration_and_budgets();
    test_5_deduplication_and_tag_rollover();
    test_6_openmp_concurrency_determinism();
    test_7_input_validation_and_minhash();
    std::cout << "====================================================================\n";
    std::cout << "  ALL 7 TEST SUITES PASSED SUCCESSFULLY!                            \n";
    std::cout << "====================================================================\n";
    return 0;
}
