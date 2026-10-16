#include "core/ple.hpp"

#include <sys/mman.h>

#include <chrono>
#include <fstream>
#include <map>
#include <set>

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
    for (auto &t : ctx) t = cfg::real_token(t);  // vision tokens hash as their pad token
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

void PleTable::open_checkpoint(const std::string &dir) {
    const Json idx = Json::parse(read_file(dir + "/model.safetensors.index.json"));
    std::map<int64_t, std::string> weights;  // shard -> its weight tensor
    std::string global;
    std::set<std::string> files;
    for (const auto &kv : idx["weight_map"].obj) {
        const std::string &n = kv.first;
        const size_t at = n.find("ngram_embedding.");
        if (at == std::string::npos) continue;
        files.insert(kv.second.as_str());
        if (n.compare(at, 22, "ngram_embedding.shard_") == 0 && n.size() > 7 && n.compare(n.size() - 7, 7, ".weight") == 0)
            weights[std::stoll(n.substr(at + 22))] = n;
        else if (n.size() > 15 && n.compare(n.size() - 15, 15, ".weight_scale_2") == 0)
            global = n;
    }
    QW_CHECK(!weights.empty() && !global.empty(), dir + ": no n-gram table in the checkpoint (and no META.json)");
    auto st = std::make_unique<SafeTensors>();
    for (const auto &f : files) st->add_file(dir + "/" + f);
    const TensorView &g = st->get(global);
    QW_CHECK(g.dtype == DType::F32 && g.numel() == 1, global + ": one FP32 scale");
    layout_ = Layout::F4;
    for (int b = 0; b < 256; ++b) f8_[size_t(b)] = f8e4m3_to_f32(uint8_t(b));
    rows_per_shard_ = st->get(weights.begin()->second).dim(0);
    rows_ = 0;
    for (const auto &[s, name] : weights) {
        QW_CHECK(s == int64_t(q_.size()), dir + ": n-gram shards not numbered 0, 1, ...");
        const TensorView &q = st->get(name, DType::U8, {st->get(name).dim(0), cfg::NGRAM_DIM / 2});
        const TensorView &sc = st->get(name + "_scale", DType::F8E4M3, {q.dim(0), cfg::NGRAM_DIM / 16});
        QW_CHECK(q.dim(0) == rows_per_shard_ || s == int64_t(weights.size()) - 1, "PLE shard row count");
        q_.push_back(q.u8());
        scale8_.push_back(sc.u8());
        global_.push_back(g.f32()[0]);
        ranges_.emplace_back(q.u8(), q.nbytes);
        ranges_.emplace_back(sc.u8(), sc.nbytes);
        rows_ += q.dim(0);
    }
    shards_.push_back(std::move(st));
}

PleTable::PleTable(const std::string &dir) {
    if (!std::ifstream(dir + "/META.json") && std::ifstream(dir + "/model.safetensors.index.json")) {
        open_checkpoint(dir);
        NgramHasher h;
        int64_t total = h.sizes().back();
        for (int i = 0; i + 1 < cfg::NGRAM_HEADS; ++i) total += h.sizes()[size_t(i)];
        QW_CHECK(total <= rows_, "PLE table has fewer rows than the n-gram vocabulary");
        return;
    }
    Json meta = Json::parse(read_file(dir + "/META.json"));
    const std::string layout = meta["layout"].as_str();
    if (layout == "group16_int4_fp16scale_lownibblefirst")
        layout_ = Layout::Int4;
    else if (layout == "group16_int8_fp16scale")
        layout_ = Layout::Int8;
    else if (layout == "bf16")
        layout_ = Layout::Bf16;
    else if (layout == "f8e4m3_tensorscale")
        layout_ = Layout::F8;
    else
        fail("PLE sidecar: unsupported layout " + layout);
    QW_CHECK(meta["width"].as_int() == cfg::NGRAM_DIM, "PLE sidecar: width");
    if (layout_ == Layout::F8) {
        const float scale = float(meta["scale"].as_double());
        for (int b = 0; b < 256; ++b) f8_[size_t(b)] = f8e4m3_to_f32(uint8_t(b)) * scale;
    }
    rows_ = meta["rows"].as_int();
    int64_t n_shards = meta["shards"].as_int();
    rows_per_shard_ = (rows_ + n_shards - 1) / n_shards;
    for (int64_t s = 0; s < n_shards; ++s) {
        auto st = std::make_unique<SafeTensors>();
        st->add_file(dir + "/shard_" + std::to_string(s) + ".safetensors");
        const char *name = layout_ == Layout::Int4 ? "weight_i4"
                           : layout_ == Layout::Int8 ? "weight_i8"
                           : layout_ == Layout::F8   ? "weight_f8"
                                                     : "weight";
        const TensorView &q = st->get(name);
        const DType want = layout_ == Layout::Int4 || layout_ == Layout::F8 ? DType::U8
                           : layout_ == Layout::Int8                        ? DType::I8
                                                                            : DType::BF16;
        const int64_t width = layout_ == Layout::Int4 ? cfg::NGRAM_DIM / 2 : cfg::NGRAM_DIM;
        QW_CHECK(q.dtype == want && q.dim(1) == width, "PLE shard layout");
        QW_CHECK(q.dim(0) == rows_per_shard_ || s == n_shards - 1, "PLE shard row count");
        q_.push_back(q.u8());
        if (layout_ == Layout::Int4 || layout_ == Layout::Int8) {
            const TensorView &sc = st->get("weight_scale");
            QW_CHECK(sc.dtype == DType::F16 && sc.dim(1) == cfg::NGRAM_DIM / 16, "PLE scale layout");
            scale_.push_back(sc.u16());
        }
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
    if (layout_ == Layout::Bf16) {
        const uint16_t *v = reinterpret_cast<const uint16_t *>(q_[size_t(s)]) + r * cfg::NGRAM_DIM;
        for (int j = 0; j < cfg::NGRAM_DIM; ++j) out[j] = bf16_to_f32(v[j]);
        return;
    }
    if (layout_ == Layout::F8) {
        const uint8_t *v = q_[size_t(s)] + r * cfg::NGRAM_DIM;
        for (int j = 0; j < cfg::NGRAM_DIM; ++j) out[j] = f8_[v[j]];
        return;
    }
    if (layout_ == Layout::F4) {
        static constexpr float E2M1[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
        const uint8_t *q = q_[size_t(s)] + r * (cfg::NGRAM_DIM / 2), *sc = scale8_[size_t(s)] + r * (cfg::NGRAM_DIM / 16);
        for (int g = 0; g < cfg::NGRAM_DIM / 16; ++g) {
            const float scale = f8_[sc[g]] * global_[size_t(s)];
            for (int j = 0; j < 8; ++j) {
                const uint8_t b = q[g * 8 + j];
                out[g * 16 + 2 * j] = E2M1[b & 0xf] * scale;
                out[g * 16 + 2 * j + 1] = E2M1[b >> 4] * scale;
            }
        }
        return;
    }
    const uint16_t *sc = scale_[size_t(s)] + r * (cfg::NGRAM_DIM / 16);
    if (layout_ == Layout::Int8) {
        const int8_t *q = reinterpret_cast<const int8_t *>(q_[size_t(s)]) + r * cfg::NGRAM_DIM;
        for (int g = 0; g < cfg::NGRAM_DIM / 16; ++g) {
            const float scale = f16_to_f32(sc[g]);
            for (int j = 0; j < 16; ++j) out[g * 16 + j] = float(q[g * 16 + j]) * scale;
        }
        return;
    }
    const uint8_t *q = q_[size_t(s)] + r * (cfg::NGRAM_DIM / 2);
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
    for (const auto &[p, n] : ranges_) {  // checkpoint mode: the table's bytes only, not the files around them
        const uintptr_t a = reinterpret_cast<uintptr_t>(p) & ~uintptr_t(4095);
        ::madvise(reinterpret_cast<void *>(a), n + (reinterpret_cast<uintptr_t>(p) - a), MADV_WILLNEED);
    }
    if (!ranges_.empty()) return;
    for (auto &s : shards_)
        for (auto &f : s->files()) f->prefetch();
}

void PleTable::pin_in_background(bool lock) {
    pin_thread_ = std::thread([this, lock] {
        const auto t0 = std::chrono::steady_clock::now();
        size_t bytes = 0;
        bool pinned = lock;
        for (const auto &[p, n] : ranges_) {  // checkpoint mode: the table's bytes only, not the files around them
            if (stop_pin_) return;
            const uintptr_t a = reinterpret_cast<uintptr_t>(p) & ~uintptr_t(4095);
            void *at = reinterpret_cast<void *>(a);
            const size_t len = n + (reinterpret_cast<uintptr_t>(p) - a);
            if (!pinned || ::mlock(at, len) != 0) {
                pinned = false;
                volatile uint8_t sink = 0;
                for (size_t i = 0; i < n; i += 4096) sink = sink + p[i];
                (void)sink;
            }
            bytes += n;
        }
        for (const auto &st : shards_) {
            if (!ranges_.empty()) break;
            for (const auto &f : st->files()) {
                if (stop_pin_) return;
                if (!pinned || !f->lock()) {
                    pinned = false;
                    f->touch();
                }
                bytes += f->size();
            }
        }
        log("PLE table: %.1f GB %s in RAM in %.0f s", double(bytes) / 1e9,
            pinned ? "pinned"
            : lock ? "read (mlock refused)"
                   : "read (page cache)",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    });
}

void PleTable::wait_pinned() {
    if (pin_thread_.joinable()) pin_thread_.join();
}

PleTable::~PleTable() {
    stop_pin_ = true;
    if (pin_thread_.joinable()) pin_thread_.join();
}

}  // namespace qw
