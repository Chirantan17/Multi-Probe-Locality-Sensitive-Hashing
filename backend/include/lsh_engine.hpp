#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <stdexcept>
#include <initializer_list>
#include <chrono>
#include <limits>
#include <numeric>
#include <memory>
#include <immintrin.h>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace lsh {

using VectorId = uint32_t;
using HashCode = uint64_t;

template <typename T, std::size_t Alignment>
struct aligned_allocator {
    using value_type = T;
    constexpr aligned_allocator() noexcept = default;
    template <class U>
    constexpr aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();
        void* ptr = nullptr;
#if defined(_WIN32)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) ptr = nullptr;
#endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        if (!p) return;
#if defined(_WIN32)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    template <typename U>
    struct rebind { using other = aligned_allocator<U, Alignment>; };

    bool operator==(const aligned_allocator&) const noexcept { return true; }
    bool operator!=(const aligned_allocator&) const noexcept { return false; }
};

struct AlignedVector {
    std::vector<float, aligned_allocator<float, 32>> data;

    explicit AlignedVector(size_t dim = 0, float init = 0.0f) : data(dim, init) {}
    AlignedVector(std::initializer_list<float> list) : data(list.begin(), list.end()) {}

    size_t size() const noexcept { return data.size(); }
    bool empty() const noexcept { return data.empty(); }
    const float* raw() const noexcept { return data.data(); }
    float* raw() noexcept { return data.data(); }

    float operator[](size_t idx) const noexcept { return data[idx]; }
    float& operator[](size_t idx) noexcept { return data[idx]; }

    bool is_32b_aligned() const noexcept {
        return (reinterpret_cast<uintptr_t>(data.data()) & 31U) == 0U;
    }

    bool has_non_finite() const noexcept {
        for (float v : data) {
            if (!std::isfinite(v)) return true;
        }
        return false;
    }
};

struct Neighbor {
    VectorId id{0};
    float distance{0.0f};

    bool operator<(const Neighbor& other) const noexcept {
        if (std::abs(distance - other.distance) > 1e-7f) return distance < other.distance;
        return id < other.id;
    }
};

struct QueryResult {
    std::vector<Neighbor> neighbors;
    size_t candidates_evaluated{0};
    size_t buckets_probed{0};
    double latency_us{0.0};
    bool budget_saturated{false};
};

namespace simd {

inline float scalar_dot_product(const float* a, const float* b, size_t dim) noexcept {
    float result = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

inline float dot_product(const float* a, const float* b, size_t dim) noexcept {
    size_t i = 0;
    float result = 0.0f;
#if defined(__AVX2__)
    __m256 sum256 = _mm256_setzero_ps();
    const bool aligned = ((reinterpret_cast<uintptr_t>(a) & 31U) == 0U) &&
                         ((reinterpret_cast<uintptr_t>(b) & 31U) == 0U);
    if (aligned) {
        for (; i + 8 <= dim; i += 8) {
            __m256 va = _mm256_load_ps(a + i);
            __m256 vb = _mm256_load_ps(b + i);
            sum256 = _mm256_fmadd_ps(va, vb, sum256);
        }
    } else {
        for (; i + 8 <= dim; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            sum256 = _mm256_fmadd_ps(va, vb, sum256);
        }
    }
    __m128 lo = _mm256_castps256_ps128(sum256);
    __m128 hi = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    result = _mm_cvtss_f32(sum128);
#endif
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

inline float norm(const float* a, size_t dim) noexcept {
    return std::sqrt(std::max(0.0f, dot_product(a, a, dim)));
}

inline void validate_pair(const AlignedVector& a, const AlignedVector& b) {
    if (a.empty() || a.size() != b.size()) {
        throw std::invalid_argument("Vector dimension mismatch or empty vector.");
    }
}

inline float cosine_distance_unit(const AlignedVector& a, const AlignedVector& b) {
    validate_pair(a, b);
    float dot = dot_product(a.raw(), b.raw(), a.size());
    return 1.0f - std::clamp(dot, -1.0f, 1.0f);
}

inline float cosine_distance_general(const AlignedVector& a, const AlignedVector& b) {
    validate_pair(a, b);
    if (a.has_non_finite() || b.has_non_finite()) {
        throw std::invalid_argument("Non-finite elements in cosine_distance_general.");
    }
    float na = norm(a.raw(), a.size());
    float nb = norm(b.raw(), b.size());
    if (na <= 1e-12f || nb <= 1e-12f) {
        throw std::invalid_argument("Zero-norm vector in cosine_distance_general.");
    }
    float cos_sim = dot_product(a.raw(), b.raw(), a.size()) / (na * nb);
    return 1.0f - std::clamp(cos_sim, -1.0f, 1.0f);
}

inline void normalize_in_place(AlignedVector& v) {
    if (v.empty() || v.has_non_finite()) {
        throw std::invalid_argument("Cannot normalize empty or non-finite vector.");
    }
    float n = norm(v.raw(), v.size());
    if (n <= 1e-12f) {
        throw std::invalid_argument("Cannot normalize zero-norm vector.");
    }
    float inv = 1.0f / n;
    for (size_t i = 0; i < v.size(); ++i) v[i] *= inv;
}

} // namespace simd

enum class CostMode {
    Squared, // Lv et al. (2007) sum of squared margins: m_i^2 + m_j^2
    Linear   // Laplace heuristic sum of linear margins: m_i + m_j
};

struct BitMargin {
    size_t bit_index{0};
    float margin{0.0f};
    float weight{0.0f};

    bool operator<(const BitMargin& other) const noexcept {
        if (std::abs(weight - other.weight) > 1e-7f) return weight < other.weight;
        return bit_index < other.bit_index;
    }
};

struct PerturbationCandidate {
    HashCode bucket_code{0};
    float cost{0.0f};
    uint8_t radius{0};

    bool operator<(const PerturbationCandidate& other) const noexcept {
        if (std::abs(cost - other.cost) > 1e-7f) return cost < other.cost;
        if (radius != other.radius) return radius < other.radius;
        return bucket_code < other.bucket_code;
    }
};

struct QueryScratchpad {
    std::vector<uint32_t> visited_stamps;
    uint32_t tag{1};
    std::vector<BitMargin> bit_margins;
    std::vector<PerturbationCandidate> candidates;
    std::vector<HashCode> selected_probes;
    std::vector<VectorId> candidate_ids;
    std::vector<Neighbor> neighbors;
    AlignedVector centered_q;

    QueryScratchpad(size_t corpus_size, size_t max_k, size_t dim = 128)
        : visited_stamps(corpus_size, 0U),
          tag(1U),
          bit_margins(max_k),
          centered_q(dim, 0.0f)
    {
        size_t max_cands = 1 + max_k + (max_k * (max_k - 1)) / 2;
        candidates.reserve(max_cands);
        selected_probes.reserve(max_cands);
        candidate_ids.reserve(corpus_size > 0 ? corpus_size : 1024);
        neighbors.reserve(corpus_size > 0 ? corpus_size : 1024);
    }

    void ensure_capacity(size_t corpus_size, size_t k, size_t dim) {
        if (visited_stamps.size() < corpus_size) {
            visited_stamps.assign(corpus_size, 0U);
            tag = 1U;
        }
        if (bit_margins.size() < k) bit_margins.resize(k);
        size_t max_cands = 1 + k + (k * (k - 1)) / 2;
        if (candidates.capacity() < max_cands) candidates.reserve(max_cands);
        if (selected_probes.capacity() < max_cands) selected_probes.reserve(max_cands);
        if (candidate_ids.capacity() < corpus_size) candidate_ids.reserve(corpus_size);
        if (neighbors.capacity() < corpus_size) neighbors.reserve(corpus_size);
        if (centered_q.size() != dim) centered_q = AlignedVector(dim, 0.0f);
    }

    void advance_tag() noexcept {
        if (tag == 0U || tag == 0xFFFFFFFFU) {
            std::fill(visited_stamps.begin(), visited_stamps.end(), 0U);
            tag = 1U;
        } else {
            ++tag;
        }
    }
};

class MultiProbeRouter {
public:
    static constexpr size_t max_radius2_probes(size_t k) noexcept {
        return 1 + k + (k * (k - 1)) / 2;
    }

    static void generate_probes_in_place(
        const AlignedVector& query,
        const std::vector<AlignedVector>& hyperplanes,
        size_t probe_budget,
        std::vector<BitMargin>& margins,
        std::vector<PerturbationCandidate>& candidates,
        std::vector<HashCode>& selected_probes,
        bool& budget_saturated,
        CostMode mode = CostMode::Squared)
    {
        size_t k = hyperplanes.size();
        if (k == 0 || k > 64) {
            throw std::invalid_argument("Hyperplane count K must be in [1, 64].");
        }
        selected_probes.clear();
        candidates.clear();
        budget_saturated = false;

        if (probe_budget == 0) return;

        if (margins.size() < k) margins.resize(k);
        HashCode primary_hash = 0ULL;

        for (size_t i = 0; i < k; ++i) {
            if (query.size() != hyperplanes[i].size()) {
                throw std::invalid_argument("Query dimension does not match hyperplane dimension.");
            }
            float proj = simd::dot_product(query.raw(), hyperplanes[i].raw(), query.size());
            if (proj >= 0.0f) {
                primary_hash |= (1ULL << i);
            }
            float m = std::abs(proj);
            float w = (mode == CostMode::Squared) ? (m * m) : m;
            margins[i] = {i, m, w};
        }

        size_t max_avail = max_radius2_probes(k);
        if (probe_budget > max_avail) {
            budget_saturated = true;
        }
        size_t target = std::min(probe_budget, max_avail);
        selected_probes.push_back(primary_hash);
        if (target == 1) return;

        std::sort(margins.begin(), margins.begin() + k);

        // Radius 1: K single-bit flips
        for (size_t i = 0; i < k; ++i) {
            HashCode mask = (1ULL << margins[i].bit_index);
            candidates.push_back({primary_hash ^ mask, margins[i].weight, 1});
        }

        // Radius 2: K*(K-1)/2 pairwise flips
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = i + 1; j < k; ++j) {
                HashCode mask = (1ULL << margins[i].bit_index) ^ (1ULL << margins[j].bit_index);
                candidates.push_back({primary_hash ^ mask, margins[i].weight + margins[j].weight, 2});
            }
        }

        std::sort(candidates.begin(), candidates.end());

        for (const auto& cand : candidates) {
            if (selected_probes.size() >= target) break;
            selected_probes.push_back(cand.bucket_code);
        }
    }

    static std::vector<HashCode> get_probe_buckets(
        const AlignedVector& query,
        const std::vector<AlignedVector>& hyperplanes,
        size_t probe_budget,
        CostMode mode = CostMode::Squared)
    {
        std::vector<BitMargin> margins(hyperplanes.size());
        std::vector<PerturbationCandidate> candidates;
        std::vector<HashCode> probes;
        bool saturated = false;
        generate_probes_in_place(query, hyperplanes, probe_budget, margins, candidates, probes, saturated, mode);
        return probes;
    }
};

struct BucketStats {
    size_t non_empty_buckets{0};
    size_t max_bucket_size{0};
    double shannon_entropy_bits{0.0};
};

class LSHIndex {
public:
    LSHIndex(size_t dim, size_t num_tables, size_t num_bits, uint32_t seed = 42, CostMode mode = CostMode::Squared)
        : dim_(dim), L_(num_tables), K_(num_bits), cost_mode_(mode), use_centering_(false), mean_vec_(dim, 0.0f)
    {
        if (dim_ == 0) throw std::invalid_argument("Dimension must be >= 1.");
        if (L_ == 0) throw std::invalid_argument("Number of tables L must be >= 1.");
        if (K_ == 0 || K_ > 64) throw std::invalid_argument("Number of bits K must be in [1, 64].");

        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        hyperplanes_.resize(L_);
        tables_.resize(L_);

        for (size_t l = 0; l < L_; ++l) {
            hyperplanes_[l].reserve(K_);
            for (size_t k = 0; k < K_; ++k) {
                AlignedVector h(dim_);
                float sq_sum = 0.0f;
                for (size_t d = 0; d < dim_; ++d) {
                    float val = dist(rng);
                    h[d] = val;
                    sq_sum += val * val;
                }
                float inv_norm = 1.0f / std::sqrt(std::max(sq_sum, 1e-12f));
                for (size_t d = 0; d < dim_; ++d) {
                    h[d] *= inv_norm;
                }
                hyperplanes_[l].push_back(std::move(h));
            }
        }
    }

    void set_cost_mode(CostMode mode) noexcept { cost_mode_ = mode; }
    CostMode cost_mode() const noexcept { return cost_mode_; }

    HashCode hash_vector(size_t table_idx, const AlignedVector& vec) const {
        if (table_idx >= L_) throw std::out_of_range("Table index out of range.");
        if (vec.size() != dim_ || vec.has_non_finite()) {
            throw std::invalid_argument("Invalid vector dimension or non-finite values in hash_vector.");
        }
        HashCode h = 0ULL;
        for (size_t k = 0; k < K_; ++k) {
            float proj = simd::dot_product(vec.raw(), hyperplanes_[table_idx][k].raw(), dim_);
            if (proj >= 0.0f) {
                h |= (1ULL << k);
            }
        }
        return h;
    }

    void clear() {
        for (auto& t : tables_) t.clear();
        corpus_size_ = 0;
    }

    void insert(VectorId id, const AlignedVector& vec) {
        if (vec.size() != dim_ || vec.has_non_finite()) {
            throw std::invalid_argument("Invalid vector in insert.");
        }
        for (size_t l = 0; l < L_; ++l) {
            HashCode h = hash_vector(l, vec);
            tables_[l][h].push_back(id);
        }
        if (static_cast<size_t>(id) + 1 > corpus_size_) {
            corpus_size_ = static_cast<size_t>(id) + 1;
        }
    }

    void build(const std::vector<AlignedVector>& dataset, bool center_for_hashing = false) {
        if (dataset.empty()) throw std::invalid_argument("Cannot build LSHIndex on empty dataset.");
        clear();
        use_centering_ = center_for_hashing;
        mean_vec_ = AlignedVector(dim_, 0.0f);

        if (use_centering_) {
            for (const auto& v : dataset) {
                if (v.size() != dim_ || v.has_non_finite()) {
                    throw std::invalid_argument("Dataset vector dimension mismatch or non-finite value.");
                }
                for (size_t d = 0; d < dim_; ++d) mean_vec_[d] += v[d];
            }
            float inv_n = 1.0f / static_cast<float>(dataset.size());
            for (size_t d = 0; d < dim_; ++d) mean_vec_[d] *= inv_n;

            AlignedVector centered(dim_);
            for (VectorId id = 0; id < static_cast<VectorId>(dataset.size()); ++id) {
                center_and_normalize(dataset[id], centered);
                for (size_t l = 0; l < L_; ++l) {
                    tables_[l][hash_vector(l, centered)].push_back(id);
                }
            }
            corpus_size_ = dataset.size();
        } else {
            for (VectorId id = 0; id < static_cast<VectorId>(dataset.size()); ++id) {
                insert(id, dataset[id]);
            }
        }
    }

    QueryResult query_fast(
        const AlignedVector& q,
        const std::vector<AlignedVector>& dataset,
        size_t top_k,
        size_t probes_per_table,
        QueryScratchpad& scratch) const
    {
        if (q.size() != dim_ || q.has_non_finite()) {
            throw std::invalid_argument("Invalid query vector dimension or non-finite values.");
        }
        if (dataset.empty() || top_k == 0) {
            return {};
        }
        if (scratch.visited_stamps.size() < dataset.size()) {
            throw std::invalid_argument("QueryScratchpad visited_stamps array is smaller than dataset size.");
        }
        if (scratch.bit_margins.size() < K_) {
            scratch.bit_margins.resize(K_);
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        scratch.advance_tag();
        const uint32_t current_tag = scratch.tag;
        scratch.candidate_ids.clear();
        scratch.neighbors.clear();

        const AlignedVector* hash_q = &q;
        if (use_centering_) {
            center_and_normalize(q, scratch.centered_q);
            hash_q = &scratch.centered_q;
        }

        size_t total_buckets = 0;
        bool any_saturated = false;

        for (size_t l = 0; l < L_; ++l) {
            bool table_saturated = false;
            MultiProbeRouter::generate_probes_in_place(
                *hash_q,
                hyperplanes_[l],
                probes_per_table,
                scratch.bit_margins,
                scratch.candidates,
                scratch.selected_probes,
                table_saturated,
                cost_mode_);

            if (table_saturated) any_saturated = true;
            total_buckets += scratch.selected_probes.size();

            for (HashCode code : scratch.selected_probes) {
                auto it = tables_[l].find(code);
                if (it != tables_[l].end()) {
                    for (VectorId id : it->second) {
                        if (id >= dataset.size()) {
                            throw std::out_of_range("Candidate ID exceeds dataset bounds.");
                        }
                        if (scratch.visited_stamps[id] != current_tag) {
                            scratch.visited_stamps[id] = current_tag;
                            scratch.candidate_ids.push_back(id);
                        }
                    }
                }
            }
        }

        for (VectorId id : scratch.candidate_ids) {
            float dist = simd::cosine_distance_unit(q, dataset[id]);
            scratch.neighbors.push_back({id, dist});
        }

        size_t k = std::min(top_k, scratch.neighbors.size());
        if (k > 0) {
            std::partial_sort(scratch.neighbors.begin(), scratch.neighbors.begin() + k, scratch.neighbors.end());
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double lat_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

        QueryResult result;
        result.neighbors.assign(scratch.neighbors.begin(), scratch.neighbors.begin() + k);
        result.candidates_evaluated = scratch.candidate_ids.size();
        result.buckets_probed = total_buckets;
        result.latency_us = lat_us;
        result.budget_saturated = any_saturated;
        return result;
    }

    // Legacy compatibility overload with strict validation
    QueryResult query_fast(
        const AlignedVector& q,
        const std::vector<AlignedVector>& dataset,
        size_t top_k,
        size_t probes_per_table,
        std::vector<uint32_t>& visited_stamps,
        uint32_t query_tag) const
    {
        if (query_tag == 0U) {
            throw std::invalid_argument("Query tag 0 is reserved for unvisited state.");
        }
        if (visited_stamps.size() < dataset.size()) {
            throw std::invalid_argument("visited_stamps buffer smaller than dataset.");
        }
        QueryScratchpad scratch(dataset.size(), K_, dim_);
        scratch.visited_stamps = std::move(visited_stamps);
        scratch.tag = query_tag - 1U;
        QueryResult res = query_fast(q, dataset, top_k, probes_per_table, scratch);
        visited_stamps = std::move(scratch.visited_stamps);
        return res;
    }

    BucketStats table_bucket_stats(size_t table_idx = 0) const {
        if (table_idx >= L_) throw std::out_of_range("Table index out of range.");
        BucketStats st;
        st.non_empty_buckets = tables_[table_idx].size();
        size_t total_items = 0;
        for (const auto& kv : tables_[table_idx]) {
            size_t sz = kv.second.size();
            if (sz > st.max_bucket_size) st.max_bucket_size = sz;
            total_items += sz;
        }
        if (total_items > 0) {
            double h = 0.0;
            for (const auto& kv : tables_[table_idx]) {
                if (!kv.second.empty()) {
                    double p = static_cast<double>(kv.second.size()) / static_cast<double>(total_items);
                    h -= p * std::log2(p);
                }
            }
            st.shannon_entropy_bits = h;
        }
        return st;
    }

    size_t memory_bytes() const noexcept {
        size_t bytes = L_ * K_ * dim_ * sizeof(float);
        for (const auto& table : tables_) {
            bytes += sizeof(table) + (table.bucket_count() * sizeof(void*));
            for (const auto& kv : table) {
                bytes += sizeof(HashCode) + sizeof(std::vector<VectorId>) + (kv.second.capacity() * sizeof(VectorId));
            }
        }
        return bytes;
    }

    size_t L() const noexcept { return L_; }
    size_t K() const noexcept { return K_; }
    size_t dim() const noexcept { return dim_; }

private:
    void center_and_normalize(const AlignedVector& in, AlignedVector& out) const {
        if (out.size() != dim_) out = AlignedVector(dim_);
        float sq = 0.0f;
        for (size_t d = 0; d < dim_; ++d) {
            float diff = in[d] - mean_vec_[d];
            out[d] = diff;
            sq += diff * diff;
        }
        float inv = 1.0f / std::sqrt(std::max(sq, 1e-12f));
        for (size_t d = 0; d < dim_; ++d) {
            out[d] *= inv;
        }
    }

    size_t dim_;
    size_t L_;
    size_t K_;
    CostMode cost_mode_;
    bool use_centering_;
    AlignedVector mean_vec_;
    size_t corpus_size_{0};
    std::vector<std::vector<AlignedVector>> hyperplanes_;
    std::vector<std::unordered_map<HashCode, std::vector<VectorId>>> tables_;
};

// 64-bit Carter-Wegman MinHash over Mersenne Prime 2^31 - 1 with 128-bit intermediate products
class MinHash {
public:
    static constexpr uint64_t MERSENNE_PRIME = 2147483647ULL; // 2^31 - 1

    explicit MinHash(size_t num_hashes, uint32_t seed = 42) : num_hashes_(num_hashes) {
        if (num_hashes_ == 0) throw std::invalid_argument("MinHash num_hashes must be >= 1.");
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist_a(1ULL, MERSENNE_PRIME - 1ULL);
        std::uniform_int_distribution<uint64_t> dist_b(0ULL, MERSENNE_PRIME - 1ULL);
        a_params_.resize(num_hashes_);
        b_params_.resize(num_hashes_);
        for (size_t i = 0; i < num_hashes_; ++i) {
            a_params_[i] = dist_a(rng);
            b_params_[i] = dist_b(rng);
        }
    }

    std::vector<uint64_t> compute_signature(const std::vector<uint64_t>& tokens) const {
        std::vector<uint64_t> sig(num_hashes_, MERSENNE_PRIME);
        for (uint64_t tok : tokens) {
            uint64_t x = tok % MERSENNE_PRIME;
            for (size_t i = 0; i < num_hashes_; ++i) {
                uint64_t h = static_cast<uint64_t>(
                    (static_cast<unsigned __int128>(a_params_[i]) * x + b_params_[i]) % MERSENNE_PRIME
                );
                if (h < sig[i]) sig[i] = h;
            }
        }
        return sig;
    }

    static float jaccard_similarity(const std::vector<uint64_t>& sig_a, const std::vector<uint64_t>& sig_b) {
        if (sig_a.empty() || sig_a.size() != sig_b.size()) {
            throw std::invalid_argument("MinHash signature size mismatch or empty.");
        }
        size_t matches = 0;
        for (size_t i = 0; i < sig_a.size(); ++i) {
            if (sig_a[i] == sig_b[i]) ++matches;
        }
        return static_cast<float>(matches) / static_cast<float>(sig_a.size());
    }

private:
    size_t num_hashes_;
    std::vector<uint64_t> a_params_;
    std::vector<uint64_t> b_params_;
};

} // namespace lsh
