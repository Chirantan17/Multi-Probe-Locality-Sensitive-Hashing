#include "lsh_engine.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <unordered_set>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <string>
#include <filesystem>
#include <omp.h>

using namespace lsh;

std::vector<AlignedVector> load_fvecs(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return {};
    std::vector<AlignedVector> vectors;
    int dim = 0;
    while (file.read(reinterpret_cast<char*>(&dim), sizeof(int))) {
        if (dim <= 0 || dim > 4096) break;
        AlignedVector vec(dim);
        file.read(reinterpret_cast<char*>(vec.raw()), dim * sizeof(float));
        float sq = 0.0f;
        for (size_t d = 0; d < static_cast<size_t>(dim); ++d) sq += vec[d] * vec[d];
        float inv = 1.0f / (std::sqrt(sq) + 1e-8f);
        for (size_t d = 0; d < static_cast<size_t>(dim); ++d) vec[d] *= inv;
        vectors.push_back(std::move(vec));
    }
    return vectors;
}

std::vector<Neighbor> brute_force_search(const std::vector<AlignedVector>& dataset, const AlignedVector& query, size_t top_k) {
    std::vector<Neighbor> candidates;
    candidates.reserve(dataset.size());
    for (VectorId id = 0; id < dataset.size(); ++id) candidates.push_back({id, simd::cosine_distance_unit(dataset[id], query)});
    std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end());
    candidates.resize(top_k);
    return candidates;
}

float compute_recall(const std::vector<Neighbor>& ground_truth, const std::vector<Neighbor>& approx) {
    if (ground_truth.empty() || approx.empty()) return 0.0f;
    std::unordered_set<VectorId> gt_set;
    for (const auto& n : ground_truth) gt_set.insert(n.id);
    size_t matches = 0;
    for (const auto& n : approx) if (gt_set.count(n.id)) matches++;
    return static_cast<float>(matches) / static_cast<float>(ground_truth.size());
}

struct LatencyStats { double avg, p50, p95, p99; };
LatencyStats compute_stats(std::vector<double>& latencies) {
    if (latencies.empty()) return {0,0,0,0};
    std::sort(latencies.begin(), latencies.end());
    double sum = 0.0;
    for (double lat : latencies) sum += lat;
    return { sum / latencies.size(), latencies[latencies.size() * 0.50], latencies[latencies.size() * 0.95], latencies[latencies.size() * 0.99] };
}

int main() {
    size_t TOP_K = 10, K = 8, L_COMPACT = 6, L_CLASSIC = 32, MP_PROBES = 8;
    std::vector<AlignedVector> dataset = load_fvecs("data/siftsmall/siftsmall_base.fvecs");
    std::vector<AlignedVector> queries = load_fvecs("data/siftsmall/siftsmall_query.fvecs");
    if(dataset.empty()) { std::cerr << "Dataset missing. Download to data/siftsmall!\n"; return 1; }
    
    size_t NUM_VECTORS = dataset.size(), DIMENSION = dataset[0].size(), NUM_QUERIES = queries.size();
    
    LSHIndex index_classic(DIMENSION, L_CLASSIC, K, 777); index_classic.build(dataset);
    LSHIndex index_compact(DIMENSION, L_COMPACT, K, 777); index_compact.build(dataset);
    
    double classic_ram_mb = index_classic.memory_bytes() / 1048576.0;
    double compact_ram_mb = index_compact.memory_bytes() / 1048576.0;
    double ram_saved_pct = (1.0 - (compact_ram_mb / classic_ram_mb)) * 100.0;

    std::vector<std::vector<Neighbor>> ground_truth(NUM_QUERIES);
    std::vector<double> bf_lats(NUM_QUERIES);
    
    // Warmup
    for(size_t i=0; i<10; ++i) brute_force_search(dataset, queries[i], TOP_K);
    
    // Brute Force (Parallelized for QPS baseline)
    auto t_bf_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic)
    for (size_t q = 0; q < NUM_QUERIES; ++q) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ground_truth[q] = brute_force_search(dataset, queries[q], TOP_K);
        bf_lats[q] = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count();
    }
    double bf_qps = NUM_QUERIES / std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_bf_start).count();
    LatencyStats bf_stats = compute_stats(bf_lats);

    auto run_benchmark = [&](const LSHIndex& idx, size_t probes, float& recall, LatencyStats& stats, double& qps, float& frac) {
        std::vector<double> lats(NUM_QUERIES);
        float r_sum = 0.0f; size_t cand_sum = 0;
        
        auto t_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel
        {
            std::vector<uint32_t> visited(dataset.size(), 0);
            uint32_t tag = 1;
            
            #pragma omp for schedule(dynamic) reduction(+:r_sum, cand_sum)
            for (size_t q = 0; q < NUM_QUERIES; ++q) {
                auto res = idx.query_fast(queries[q], dataset, TOP_K, probes, visited, tag++);
                lats[q] = res.latency_us;
                r_sum += compute_recall(ground_truth[q], res.neighbors);
                cand_sum += res.candidates_evaluated;
            }
        }
        qps = NUM_QUERIES / std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();
        stats = compute_stats(lats);
        recall = r_sum / NUM_QUERIES;
        frac = (static_cast<float>(cand_sum) / NUM_QUERIES) / NUM_VECTORS;
    };

    float c_rec, s6_rec, mp_rec, c_frac, s6_frac, mp_frac;
    LatencyStats c_st, s6_st, mp_st;
    double c_qps, s6_qps, mp_qps;

    run_benchmark(index_classic, 1, c_rec, c_st, c_qps, c_frac);
    run_benchmark(index_compact, 1, s6_rec, s6_st, s6_qps, s6_frac);
    run_benchmark(index_compact, MP_PROBES, mp_rec, mp_st, mp_qps, mp_frac);
    
    std::cout << "Configuration                   | Recall@10 | Latency (Avg) | Latency (p99) | Cand Frac | OpenMP QPS\n";
    std::cout << "------------------------------------------------------------------------------------------------------\n";
    std::cout << "Exact Brute-Force (Linear Scan) |    100.0% | " << std::setw(10) << bf_stats.avg << " us | " << std::setw(10) << bf_stats.p99 << " us |    100.0% | " << std::setw(10) << static_cast<int>(bf_qps) << "\n";
    std::cout << "Classic LSH (L=32, 1-Probe)     | " << std::setw(9) << (c_rec * 100.0f) << "% | " << std::setw(10) << c_st.avg << " us | " << std::setw(10) << c_st.p99 << " us | " << std::setw(8) << (c_frac * 100.0f) << "% | " << std::setw(10) << static_cast<int>(c_qps) << "\n";
    std::cout << "Multi-Probe LSH (L=6, 8-Probe)  | " << std::setw(9) << (mp_rec * 100.0f) << "% | " << std::setw(10) << mp_st.avg << " us | " << std::setw(10) << mp_st.p99 << " us | " << std::setw(8) << (mp_frac * 100.0f) << "% | " << std::setw(10) << static_cast<int>(mp_qps) << "\n";

    std::filesystem::create_directories("benchmarks");
    std::ofstream out("benchmarks/results.json");
    out << "{\n  \"corpus\": \"SIFT10K\",\n  \"memory\": {\"classic_mb\": " << classic_ram_mb << ", \"compact_mb\": " << compact_ram_mb << ", \"reduction_pct\": " << ram_saved_pct << "},\n";
    out << "  \"brute_force\": {\"avg_us\": " << bf_stats.avg << ", \"p99_us\": " << bf_stats.p99 << ", \"qps\": " << bf_qps << "},\n";
    out << "  \"classic_lsh\": {\"recall\": " << c_rec << ", \"avg_us\": " << c_st.avg << ", \"p99_us\": " << c_st.p99 << ", \"qps\": " << c_qps << ", \"frac\": " << c_frac << "},\n";
    out << "  \"multiprobe_lsh\": {\"recall\": " << mp_rec << ", \"avg_us\": " << mp_st.avg << ", \"p99_us\": " << mp_st.p99 << ", \"qps\": " << mp_qps << ", \"frac\": " << mp_frac << "}\n}\n";
    
    return 0;
}
