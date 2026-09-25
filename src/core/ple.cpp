#include "core/ple.hpp"

#include <chrono>

#include "core/common.hpp"
#include "core/json.hpp"

namespace qw {

namespace {

constexpr uint64_t GAMMA = 0x9E3779B97F4A7C15ull;

uint64_t splitmix64(uint64_t v) {
    v += GAMMA;
    v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
    v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
    return v ^ (v >> 31);
}

uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    return uint64_t((unsigned __int128)a * b % m);
}

uint64_t powmod(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r = 1;
    b %= m;
    while (e) {
        if (e & 1) r = mulmod(r, b, m);
        b = mulmod(b, b, m);
        e >>= 1;
    }
    return r;
}

// Deterministic Miller-Rabin for 64-bit inputs, same witnesses as the reference.
bool is_prime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull})
        if (n % p == 0) return n == p;
    uint64_t d = n - 1;
    int s = 0;
    while (d % 2 == 0) {
        d /= 2;
        ++s;
    }
    for (uint64_t a : {2ull, 325ull, 9375ull, 28178ull, 450775ull, 9780504ull, 1795265022ull}) {
        if (a % n == 0) continue;
        uint64_t x = powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (int r = 1; r < s; ++r) {
            x = mulmod(x, x, n);
            if (x == n - 1) {
                comp = false;
                break;
            }
        }
        if (comp) return false;
    }
    return true;
}

}  // namespace

NgramHasher::NgramHasher() {
    const uint64_t max_mult = ((uint64_t(1) << 63) - 1) / uint64_t(cfg::VOCAB);
    const uint64_t half = std::max<uint64_t>(1, max_mult / 2);
    const uint64_t base_seed = cfg::PLE_SEED + 10007ull * 0;  // dense PLE layer id 0
    for (int i = 0; i < cfg::NGRAM; ++i)
        mult_[size_t(i)] = 2 * (splitmix64(base_seed + GAMMA * uint64_t(i + 1)) % half) + 1;
    // Head h's table is the (h+1)-th prime above base-1.
    uint64_t p = uint64_t(cfg::NGRAM_VOCAB_BASE - 1);
    int64_t off = 0;
    for (int h = 0; h < cfg::NGRAM_HEADS; ++h) {
        uint64_t c = p + 1;
        if (c % 2 == 0) ++c;
        while (!is_prime(c)) c += 2;
        p = c;
        sizes_[size_t(h)] = int64_t(p);
        offsets_[size_t(h)] = off;
        off += int64_t(p);
    }
}

NgramIds NgramHasher::ids(const int32_t *ctx, int n, int prev_eos) const {
    // tok[s] = ctx[n-1-s] when it exists and comes after the latest EOS before
    // the current token, else EOS.
    uint64_t tok[cfg::NGRAM];
    for (int s = 0; s < cfg::NGRAM; ++s) {
        int idx = n - 1 - s;
        bool valid = s == 0 || (idx >= 0 && idx > prev_eos);
        tok[s] = uint64_t(valid ? ctx[idx] : cfg::EOS);
    }
    NgramIds out{};
    const int per = cfg::NGRAM_HEADS / (cfg::NGRAM - 1);
    for (int order = 2; order <= cfg::NGRAM; ++order) {
        uint64_t mixed = tok[0] * mult_[0];
        for (int i = 1; i < order; ++i) mixed ^= tok[i] * mult_[size_t(i)];
        for (int j = 0; j < per; ++j) {
            int h = (order - 2) * per + j;
            out[size_t(h)] = int64_t(mixed % uint64_t(sizes_[size_t(h)])) + offsets_[size_t(h)];
        }
    }
    return out;
}

std::vector<NgramIds> NgramHasher::ids_for(const std::vector<int32_t> &history,
                                           const std::vector<int32_t> &tokens) const {
    std::vector<int32_t> ctx(history.end() - std::min<ptrdiff_t>(history.size(), cfg::NGRAM - 1), history.end());
    int hist = int(ctx.size());
    ctx.insert(ctx.end(), tokens.begin(), tokens.end());
    std::vector<NgramIds> out;
    out.reserve(tokens.size());
    int prev_eos = -1;  // latest EOS strictly before the current position
    for (int i = 0; i < hist; ++i)
        if (ctx[size_t(i)] == cfg::EOS) prev_eos = i;
    for (size_t t = 0; t < tokens.size(); ++t) {
        int pos = hist + int(t);
        out.push_back(ids(ctx.data(), pos + 1, prev_eos));
        if (ctx[size_t(pos)] == cfg::EOS) prev_eos = pos;
    }
    return out;
}

PleTable::PleTable(const std::string &dir) {
    Json meta = Json::parse(read_file(dir + "/META.json"));
    QW_CHECK(meta["layout"].as_str() == "group16_int4_fp16scale_lownibblefirst",
             "PLE sidecar: unsupported layout " + meta["layout"].as_str());
    QW_CHECK(meta["width"].as_int() == cfg::NGRAM_DIM, "PLE sidecar: width");
    rows_ = meta["rows"].as_int();
    int64_t n_shards = meta["shards"].as_int();
    rows_per_shard_ = (rows_ + n_shards - 1) / n_shards;
    for (int64_t s = 0; s < n_shards; ++s) {
        auto st = std::make_unique<SafeTensors>();
        st->add_file(dir + "/shard_" + std::to_string(s) + ".safetensors");
        const TensorView &q = st->get("weight_i4");
        const TensorView &sc = st->get("weight_scale");
        QW_CHECK(q.dtype == DType::U8 && q.dim(1) == cfg::NGRAM_DIM / 2, "PLE shard layout");
        QW_CHECK(sc.dtype == DType::F16 && sc.dim(1) == cfg::NGRAM_DIM / 16, "PLE scale layout");
        QW_CHECK(q.dim(0) == rows_per_shard_ || s == n_shards - 1, "PLE shard row count");
        q_.push_back(q.u8());
        scale_.push_back(sc.u16());
        shards_.push_back(std::move(st));
    }
    NgramHasher h;
    int64_t total = h.sizes().back();
    for (int i = 0; i + 1 < cfg::NGRAM_HEADS; ++i) total += h.sizes()[size_t(i)];
    QW_CHECK(total <= rows_, "PLE sidecar has fewer rows than the n-gram vocabulary");
}

void PleTable::row(int64_t id, float *out) const {
    QW_CHECK(id >= 0 && id < rows_, "PLE row out of range");
    int64_t s = id / rows_per_shard_, r = id % rows_per_shard_;
    const uint8_t *q = q_[size_t(s)] + r * (cfg::NGRAM_DIM / 2);
    const uint16_t *sc = scale_[size_t(s)] + r * (cfg::NGRAM_DIM / 16);
    for (int g = 0; g < cfg::NGRAM_DIM / 16; ++g) {
        float scale = f16_to_f32(sc[g]);
        for (int j = 0; j < 8; ++j) {
            uint8_t b = q[g * 8 + j];
            out[g * 16 + 2 * j] = float(int(b & 0xf) - 8) * scale;
            out[g * 16 + 2 * j + 1] = float(int(b >> 4) - 8) * scale;
        }
    }
}

void PleTable::gather(const NgramIds &ids, float *out) const {
    for (int h = 0; h < cfg::NGRAM_HEADS; ++h) row(ids[size_t(h)], out + h * cfg::NGRAM_DIM);
}

void PleTable::prefetch() const {
    for (auto &s : shards_)
        for (auto &f : s->files()) f->prefetch();
}

void PleTable::pin_in_background(bool lock) {
    pin_thread_ = std::thread([this, lock] {
        const auto t0 = std::chrono::steady_clock::now();
        size_t bytes = 0;
        bool pinned = lock;
        for (const auto &st : shards_)
            for (const auto &f : st->files()) {
                if (stop_pin_) return;
                if (!pinned || !f->lock()) {
                    pinned = false;
                    f->touch();
                }
                bytes += f->size();
            }
        log("PLE table: %.1f GB %s in RAM in %.0f s", double(bytes) / 1e9,
            pinned ? "pinned"
            : lock ? "read (mlock refused)"
                   : "read (page cache)",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    });
}

PleTable::~PleTable() {
    stop_pin_ = true;
    if (pin_thread_.joinable()) pin_thread_.join();
}

}  // namespace qw
