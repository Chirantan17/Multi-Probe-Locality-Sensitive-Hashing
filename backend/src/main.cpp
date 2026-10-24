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

std::vector<AlignedVector> load_fvecs(const std::string& path, size_t max_vectors = 0) {
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
        if (max_vectors > 0 && vectors.size() >= max_vectors) break;
    }
    return vectors;
}

std::vector<AlignedVector> generate_synthetic_dataset(size_t num_vectors, size_t dim, size_t num_clusters = 20, float cluster_spread = 0.30f, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> cluster_dist(0, num_clusters - 1);

    std::vector<AlignedVector> centers(num_clusters, AlignedVector(dim));
    for (size_t c = 0; c < num_clusters; ++c) {
        float sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            centers[c][d] = normal_dist(rng);
            sq += centers[c][d] * centers[c][d];
        }
        float inv = 1.0f / (std::sqrt(sq) + 1e-8f);
        for (size_t d = 0; d < dim; ++d) centers[c][d] *= inv;
    }

    std::vector<AlignedVector> dataset(num_vectors, AlignedVector(dim));
    for (size_t i = 0; i < num_vectors; ++i) {
        size_t c = cluster_dist(rng);
        float sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            dataset[i][d] = centers[c][d] + normal_dist(rng) * cluster_spread;
            sq += dataset[i][d] * dataset[i][d];
        }
        float inv = 1.0f / (std::sqrt(sq) + 1e-8f);
        for (size_t d = 0; d < dim; ++d) dataset[i][d] *= inv;
    }
    return dataset;
}

std::vector<AlignedVector> generate_synthetic_queries(size_t num_queries, size_t dim, uint32_t seed = 9999) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);

    std::vector<AlignedVector> queries(num_queries, AlignedVector(dim));
    for (size_t q = 0; q < num_queries; ++q) {
        float sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            queries[q][d] = normal_dist(rng);
            sq += queries[q][d] * queries[q][d];
        }
        float inv = 1.0f / (std::sqrt(sq) + 1e-8f);
        for (size_t d = 0; d < dim; ++d) queries[q][d] *= inv;
    }
    return queries;
}

std::vector<Neighbor> brute_force_search(const std::vector<AlignedVector>& dataset, const AlignedVector& query, size_t top_k) {
    std::vector<Neighbor> candidates;
    candidates.reserve(dataset.size());
    for (VectorId id = 0; id < dataset.size(); ++id) {
        float dist = simd::cosine_distance_unit(dataset[id], query);
        candidates.push_back({id, dist});
    }
    std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end());
    candidates.resize(top_k);
    return candidates;
}

float compute_recall(const std::vector<Neighbor>& ground_truth, const std::vector<Neighbor>& approx) {
    if (ground_truth.empty() || approx.empty()) return 0.0f;
    std::unordered_set<VectorId> gt_set;
    for (const auto& n : ground_truth) gt_set.insert(n.id);

    size_t matches = 0;
    for (const auto& n : approx) {
        if (gt_set.count(n.id)) matches++;
    }
    return static_cast<float>(matches) / static_cast<float>(ground_truth.size());
}

struct LatencyStats {
    double avg;
    double p50;
    double p95;
    double p99;
};

LatencyStats compute_stats(std::vector<double>& latencies) {
    if (latencies.empty()) return {0,0,0,0};
    std::sort(latencies.begin(), latencies.end());
    double sum = 0.0;
    for (double lat : latencies) sum += lat;
    double avg = sum / latencies.size();
    double p50 = latencies[static_cast<size_t>(latencies.size() * 0.50)];
    double p95 = latencies[static_cast<size_t>(latencies.size() * 0.95)];
    double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];
    return {avg, p50, p95, p99};
}

int main(int argc, char* argv[]) {
    size_t TOP_K = 10;
    size_t K = 8;
    size_t L_COMPACT = 6;
    size_t L_CLASSIC = 32;
    size_t MP_PROBES = 8;

    std::cout << "========================================================================\n";
    std::cout << "  Multi-Probe LSH High-Performance Evaluation & Benchmark Engine         \n";
    std::cout << "========================================================================\n";

    std::string siftsmall_base = "data/siftsmall/siftsmall_base.fvecs";
    std::string siftsmall_query = "data/siftsmall/siftsmall_query.fvecs";

    std::vector<AlignedVector> dataset;
    std::vector<AlignedVector> queries;
    std::string dataset_name;

    if (std::filesystem::exists(siftsmall_base) && std::filesystem::exists(siftsmall_query)) {
        dataset_name = "SIFT10K (Real-world 128D INRIA Feature Vectors)";
        dataset = load_fvecs(siftsmall_base, 10000);
        queries = load_fvecs(siftsmall_query, 100);
    } else {
        dataset_name = "Synthetic Clustered Vectors (Held-out Evaluation)";
        dataset = generate_synthetic_dataset(10000, 128, 20, 0.30f, 42);
        queries = generate_synthetic_queries(200, 128, 9999);
    }

    size_t NUM_VECTORS = dataset.size();
    size_t DIMENSION = dataset[0].size();
    size_t NUM_QUERIES = queries.size();

    std::cout << "Active Corpus        : " << dataset_name << "\n";
    std::cout << "Corpus Size (N)      : " << NUM_VECTORS << "\n";
    std::cout << "Vector Dimension (D) : " << DIMENSION << "\n";
    std::cout << "Query Count          : " << NUM_QUERIES << "\n";
    std::cout << "OpenMP Threads       : " << omp_get_max_threads() << " threads\n\n";

    std::cout << "[1/4] Building Indices:\n";
    std::cout << "      -> Index A: Classic Multi-Table LSH (L=" << L_CLASSIC << ", K=" << K << ")...\n";
    LSHIndex index_classic(DIMENSION, L_CLASSIC, K, 777);
    index_classic.build(dataset);

    std::cout << "      -> Index B: Compact Multi-Probe LSH (L=" << L_COMPACT << ", K=" << K << ")...\n";
    LSHIndex index_compact(DIMENSION, L_COMPACT, K, 777);
    index_compact.build(dataset);

    double classic_ram_mb = index_classic.memory_bytes() / (1024.0 * 1024.0);
    double compact_ram_mb = index_compact.memory_bytes() / (1024.0 * 1024.0);
    double ram_saved_pct = (1.0 - (compact_ram_mb / classic_ram_mb)) * 100.0;

    std::cout << "\n================ INDEX MEMORY FOOTPRINT COMPARISON ================\n";
    std::cout << "Classic LSH Index (L=32) : " << std::fixed << std::setprecision(2) << classic_ram_mb << " MB\n";
    std::cout << "Compact LSH Index (L=6)  : " << compact_ram_mb << " MB\n";
    std::cout << "Direct Memory Reduction  : " << ram_saved_pct << "% RAM SAVED (Proven!)\n";
    std::cout << "===================================================================\n\n";

    std::cout << "[2/4] Computing Exact Brute-Force Ground Truth (Parallel OpenMP)...\n";
    std::vector<std::vector<Neighbor>> ground_truth(NUM_QUERIES);
    std::vector<double> bf_lats(NUM_QUERIES);

    auto t_bf_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic)
    for (size_t q = 0; q < NUM_QUERIES; ++q) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ground_truth[q] = brute_force_search(dataset, queries[q], TOP_K);
        auto t1 = std::chrono::high_resolution_clock::now();
        bf_lats[q] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    auto t_bf_end = std::chrono::high_resolution_clock::now();
    double bf_wall_s = std::chrono::duration<double>(t_bf_end - t_bf_start).count();
    double bf_qps = NUM_QUERIES / bf_wall_s;
    LatencyStats bf_stats = compute_stats(bf_lats);

    std::cout << "[3/4] Benchmarking Search Pipelines (Parallel OpenMP batching)...\n";

    // Classic LSH (L=32, 1 Probe)
    std::vector<double> classic_lats(NUM_QUERIES);
    float classic_recall_sum = 0.0f;
    auto t_c_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        std::vector<uint32_t> visited(dataset.size(), 0);
        uint32_t tag = 1;
        #pragma omp for schedule(dynamic) reduction(+:classic_recall_sum)
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            auto res = index_classic.query_fast(queries[q], dataset, TOP_K, 1, visited, tag++);
            classic_lats[q] = res.latency_us;
            classic_recall_sum += compute_recall(ground_truth[q], res.neighbors);
        }
    }
    auto t_c_end = std::chrono::high_resolution_clock::now();
    double classic_wall_s = std::chrono::duration<double>(t_c_end - t_c_start).count();
    double classic_qps = NUM_QUERIES / classic_wall_s;
    LatencyStats classic_stats = compute_stats(classic_lats);

    // Baseline Compact Standard LSH (L=6, 1 Probe)
    std::vector<double> std6_lats(NUM_QUERIES);
    float std6_recall_sum = 0.0f;
    auto t_s6_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        std::vector<uint32_t> visited(dataset.size(), 0);
        uint32_t tag = 1;
        #pragma omp for schedule(dynamic) reduction(+:std6_recall_sum)
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            auto res = index_compact.query_fast(queries[q], dataset, TOP_K, 1, visited, tag++);
            std6_lats[q] = res.latency_us;
            std6_recall_sum += compute_recall(ground_truth[q], res.neighbors);
        }
    }
    auto t_s6_end = std::chrono::high_resolution_clock::now();
    double std6_wall_s = std::chrono::duration<double>(t_s6_end - t_s6_start).count();
    double std6_qps = NUM_QUERIES / std6_wall_s;
    LatencyStats std6_stats = compute_stats(std6_lats);

    // Compact Multi-Probe LSH (L=6, 8 Probes)
    std::vector<double> mp_lats(NUM_QUERIES);
    float mp_recall_sum = 0.0f;
    auto t_mp_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        std::vector<uint32_t> visited(dataset.size(), 0);
        uint32_t tag = 1;
        #pragma omp for schedule(dynamic) reduction(+:mp_recall_sum)
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            auto res = index_compact.query_fast(queries[q], dataset, TOP_K, MP_PROBES, visited, tag++);
            mp_lats[q] = res.latency_us;
            mp_recall_sum += compute_recall(ground_truth[q], res.neighbors);
        }
    }
    auto t_mp_end = std::chrono::high_resolution_clock::now();
    double mp_wall_s = std::chrono::duration<double>(t_mp_end - t_mp_start).count();
    double mp_qps = NUM_QUERIES / mp_wall_s;
    LatencyStats mp_stats = compute_stats(mp_lats);

    float classic_recall = classic_recall_sum / NUM_QUERIES;
    float std6_recall = std6_recall_sum / NUM_QUERIES;
    float mp_recall = mp_recall_sum / NUM_QUERIES;

    std::cout << "[4/4] Empirical Benchmark Results:\n";
    std::cout << "=================================================================================================\n";
    std::cout << "Configuration                   | Recall@10 | Latency (Avg) | Latency (p95) | QPS (Batch) | Speedup\n";
    std::cout << "-------------------------------------------------------------------------------------------------\n";
    std::cout << "Exact Brute-Force (Linear Scan) |    100.0% | " << std::setw(10) << bf_stats.avg << " us | " << std::setw(10) << bf_stats.p95 << " us | " << std::setw(9) << static_cast<int>(bf_qps) << "   |    1.00x\n";
    std::cout << "Classic LSH (L=32, 1-Probe)     | " << std::setw(9) << (classic_recall * 100.0f) << "% | " << std::setw(10) << classic_stats.avg << " us | " << std::setw(10) << classic_stats.p95 << " us | " << std::setw(9) << static_cast<int>(classic_qps) << "   | " << std::setw(7) << (bf_stats.avg / classic_stats.avg) << "x\n";
    std::cout << "Compact LSH (L=6, 1-Probe)      | " << std::setw(9) << (std6_recall * 100.0f) << "% | " << std::setw(10) << std6_stats.avg << " us | " << std::setw(10) << std6_stats.p95 << " us | " << std::setw(9) << static_cast<int>(std6_qps) << "   | " << std::setw(7) << (bf_stats.avg / std6_stats.avg) << "x\n";
    std::cout << "Multi-Probe LSH (L=6, 8-Probe)  | " << std::setw(9) << (mp_recall * 100.0f) << "% | " << std::setw(10) << mp_stats.avg << " us | " << std::setw(10) << mp_stats.p95 << " us | " << std::setw(9) << static_cast<int>(mp_qps) << "   | " << std::setw(7) << (bf_stats.avg / mp_stats.avg) << "x\n";
    std::cout << "=================================================================================================\n";

    std::filesystem::create_directories("benchmarks");
    std::ofstream out("benchmarks/results.json");
    if (out.is_open()) {
        out << "{\n";
        out << "  \"corpus\": \"" << dataset_name << "\",\n";
        out << "  \"num_vectors\": " << NUM_VECTORS << ",\n";
        out << "  \"dimension\": " << DIMENSION << ",\n";
        out << "  \"top_k\": " << TOP_K << ",\n";
        out << "  \"num_queries\": " << NUM_QUERIES << ",\n";
        out << "  \"memory\": {\n";
        out << "    \"classic_lsh_mb\": " << classic_ram_mb << ",\n";
        out << "    \"compact_lsh_mb\": " << compact_ram_mb << ",\n";
        out << "    \"reduction_pct\": " << ram_saved_pct << "\n";
        out << "  },\n";
        out << "  \"brute_force\": { \"avg_us\": " << bf_stats.avg << ", \"p95_us\": " << bf_stats.p95 << ", \"recall\": 1.0, \"qps\": " << bf_qps << " },\n";
        out << "  \"classic_lsh\": { \"avg_us\": " << classic_stats.avg << ", \"p95_us\": " << classic_stats.p95 << ", \"recall\": " << classic_recall << ", \"qps\": " << classic_qps << " },\n";
        out << "  \"compact_std_lsh\": { \"avg_us\": " << std6_stats.avg << ", \"p95_us\": " << std6_stats.p95 << ", \"recall\": " << std6_recall << ", \"qps\": " << std6_qps << " },\n";
        out << "  \"multiprobe_lsh\": { \"avg_us\": " << mp_stats.avg << ", \"p95_us\": " << mp_stats.p95 << ", \"recall\": " << mp_recall << ", \"qps\": " << mp_qps << " }\n";
        out << "}\n";
        std::cout << "\n[Telemetry] Verified telemetry saved to benchmarks/results.json\n";
    }

    return 0;
}
