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
#include <omp.h>

using namespace lsh;

std::vector<AlignedVector> generate_dataset(size_t num_vectors, size_t dim, size_t num_clusters = 20, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> cluster_dist(0, num_clusters - 1);

    std::vector<AlignedVector> centers(num_clusters, AlignedVector(dim));
    for (size_t c = 0; c < num_clusters; ++c) {
        for (size_t d = 0; d < dim; ++d) {
            centers[c][d] = normal_dist(rng);
        }
    }

    std::vector<AlignedVector> dataset(num_vectors, AlignedVector(dim));
    for (size_t i = 0; i < num_vectors; ++i) {
        size_t c = cluster_dist(rng);
        for (size_t d = 0; d < dim; ++d) {
            dataset[i][d] = centers[c][d] + normal_dist(rng) * 0.20f;
        }
    }
    return dataset;
}

std::vector<AlignedVector> generate_queries(const std::vector<AlignedVector>& dataset, size_t num_queries, uint32_t seed = 999) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> pick_dist(0, dataset.size() - 1);
    std::normal_distribution<float> noise(0.0f, 0.05f);

    size_t dim = dataset[0].size();
    std::vector<AlignedVector> queries(num_queries, AlignedVector(dim));
    for (size_t q = 0; q < num_queries; ++q) {
        size_t base_idx = pick_dist(rng);
        for (size_t d = 0; d < dim; ++d) {
            queries[q][d] = dataset[base_idx][d] + noise(rng);
        }
    }
    return queries;
}

std::vector<Neighbor> brute_force_search(const std::vector<AlignedVector>& dataset, const AlignedVector& query, size_t top_k) {
    std::vector<Neighbor> candidates;
    candidates.reserve(dataset.size());

    for (VectorId id = 0; id < dataset.size(); ++id) {
        float dist = simd::cosine_distance(dataset[id], query);
        candidates.push_back({id, dist});
    }

    std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end());
    candidates.resize(top_k);
    return candidates;
}

float compute_recall(const std::vector<Neighbor>& ground_truth, const std::vector<Neighbor>& approx) {
    if (ground_truth.empty()) return 0.0f;
    std::unordered_set<VectorId> gt_set;
    for (const auto& n : ground_truth) gt_set.insert(n.id);

    size_t matches = 0;
    for (const auto& n : approx) {
        if (gt_set.count(n.id)) matches++;
    }
    return static_cast<float>(matches) / static_cast<float>(ground_truth.size());
}

double estimate_index_memory_mb(const LSHIndex& index) {
    return static_cast<double>(index.memory_bytes()) / (1024.0 * 1024.0);
}

int main(int argc, char* argv[]) {
    size_t NUM_VECTORS = 10000;
    size_t DIMENSION = 128;
    size_t TOP_K = 10;
    size_t NUM_QUERIES = 20;
    size_t L = 6;
    size_t K = 8;
    size_t PROBES = 6;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) NUM_VECTORS = std::stoul(argv[++i]);
        else if (arg == "-d" && i + 1 < argc) DIMENSION = std::stoul(argv[++i]);
        else if (arg == "-k" && i + 1 < argc) K = std::stoul(argv[++i]);
        else if (arg == "-l" && i + 1 < argc) L = std::stoul(argv[++i]);
        else if (arg == "-p" && i + 1 < argc) PROBES = std::stoul(argv[++i]);
    }

    std::cout << "========================================================\n";
    std::cout << "  Multi-Probe LSH High-Performance Benchmark Engine    \n";
    std::cout << "========================================================\n";
    std::cout << "Vectors: " << NUM_VECTORS << " | Dim: " << DIMENSION 
              << " | L: " << L << " | K: " << K << " | Probes: " << PROBES << "\n\n";

    auto dataset = generate_dataset(NUM_VECTORS, DIMENSION);
    auto queries = generate_queries(dataset, NUM_QUERIES);

    LSHIndex index(DIMENSION, L, K);
    index.build(dataset);

    double index_memory_mb = estimate_index_memory_mb(index);
    double raw_data_mb = (NUM_VECTORS * DIMENSION * sizeof(float)) / (1024.0 * 1024.0);

    double bf_total_latency = 0.0;
    double std_total_latency = 0.0;
    double mp_total_latency = 0.0;

    float std_total_recall = 0.0f;
    float mp_total_recall = 0.0f;

    size_t std_candidates_avg = 0;
    size_t mp_candidates_avg = 0;

    for (size_t q = 0; q < NUM_QUERIES; ++q) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto ground_truth = brute_force_search(dataset, queries[q], TOP_K);
        auto t1 = std::chrono::high_resolution_clock::now();
        bf_total_latency += std::chrono::duration<double, std::micro>(t1 - t0).count();

        auto std_res = index.query(queries[q], dataset, TOP_K, 1);
        std_total_latency += std_res.latency_us;
        std_total_recall += compute_recall(ground_truth, std_res.neighbors);
        std_candidates_avg += std_res.candidates_evaluated;

        auto mp_res = index.query(queries[q], dataset, TOP_K, PROBES);
        mp_total_latency += mp_res.latency_us;
        mp_total_recall += compute_recall(ground_truth, mp_res.neighbors);
        mp_candidates_avg += mp_res.candidates_evaluated;
    }

    double avg_bf_lat = bf_total_latency / NUM_QUERIES;
    double avg_std_lat = std_total_latency / NUM_QUERIES;
    double avg_mp_lat = mp_total_latency / NUM_QUERIES;

    float avg_std_rec = std_total_recall / NUM_QUERIES;
    float avg_mp_rec = mp_total_recall / NUM_QUERIES;

    std::cout << "\n================ BENCHMARK RESULTS ================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Method              | Latency (us) | Recall (%) | Speedup vs BF\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Exact Brute-Force   | " << std::setw(12) << avg_bf_lat  << " |     100.0% |   1.0x\n";
    std::cout << "Standard LSH (1-p)  | " << std::setw(12) << avg_std_lat << " | " << std::setw(9) << (avg_std_rec * 100.0f) << "% |  " << std::setw(4) << (avg_bf_lat / avg_std_lat) << "x\n";
    std::cout << "Multi-Probe LSH     | " << std::setw(12) << avg_mp_lat  << " | " << std::setw(9) << (avg_mp_rec * 100.0f) << "% |  " << std::setw(4) << (avg_bf_lat / avg_mp_lat) << "x\n";
    std::cout << "========================================================\n";

    std::ofstream out("benchmarks/results.json");
    if (out.is_open()) {
        out << "{\n";
        out << "  \"num_vectors\": " << NUM_VECTORS << ",\n";
        out << "  \"dimension\": " << DIMENSION << ",\n";
        out << "  \"top_k\": " << TOP_K << ",\n";
        out << "  \"raw_data_mb\": " << raw_data_mb << ",\n";
        out << "  \"index_memory_mb\": " << index_memory_mb << ",\n";
        out << "  \"brute_force\": { \"latency_us\": " << avg_bf_lat << ", \"recall\": 1.0 },\n";
        out << "  \"standard_lsh\": { \"latency_us\": " << avg_std_lat << ", \"recall\": " << avg_std_rec << ", \"candidates\": " << (std_candidates_avg / NUM_QUERIES) << " },\n";
        out << "  \"multiprobe_lsh\": { \"latency_us\": " << avg_mp_lat << ", \"recall\": " << avg_mp_rec << ", \"candidates\": " << (mp_candidates_avg / NUM_QUERIES) << " }\n";
        out << "}\n";
        std::cout << "\n[4/4] Telemetry exported to 'benchmarks/results.json' successfully.\n";
    }
    
    return 0;
}