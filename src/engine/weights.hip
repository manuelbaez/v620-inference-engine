// Weight loading: converts one rank's shard of the checkpoint into device
// layouts (fp16 dense matrices, repacked int4 experts, fp32 norms).
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/rank.hpp"
#include "engine/weight_convert.hpp"
#include "kernels/gemv.hpp"

namespace qw {

namespace {

// Converted-weights cache (EngineOptions::weight_cache_dir): the pieces a rank's loader uploads, in the order it
// uploads them. Every piece goes through item(): built by its lambda and uploaded (and appended to the file when
// recording), or, replaying, read from the file and uploaded without running the lambda. A file is used only when its
// key matches (the checkpoint's files, the rank, the layout version) and it was completed; it is written under a
// temporary name and renamed when the loader finishes. Bump WEIGHT_CACHE_VERSION when a conversion changes.
constexpr uint64_t WEIGHT_CACHE_VERSION = 1;
class WeightCache {
public:
    enum class Mode { Off, Record, Replay };
    WeightCache(const std::string &dir, uint64_t key, int r, ThreadPool &pool) {
        if (dir.empty()) return;
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        path_ = dir + "/rank" + std::to_string(r) + ".qww";
        const int fd = ::open(path_.c_str(), O_RDONLY);
        if (fd >= 0) {
            struct stat st {};
            Head h{};
            if (fstat(fd, &st) == 0 && size_t(st.st_size) >= sizeof h + sizeof(uint64_t) &&
                pread(fd, &h, sizeof h, 0) == ssize_t(sizeof h) && std::memcmp(h.magic, MAGIC, 8) == 0 && h.key == key) {
                size_ = size_t(st.st_size);
                void *m = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
                if (m != MAP_FAILED) {
                    map_ = static_cast<const uint8_t *>(m);
                    // Bring the file into this process's page tables with every loader thread before the uploads:
                    // faulting it in one page at a time under hipMemcpy took 30 s per card against 5.5 s of upload.
                    const auto t0 = std::chrono::steady_clock::now();
                    constexpr size_t CH = size_t(64) << 20;
                    const int64_t chunks = int64_t((size_ + CH - 1) / CH);
                    pool.parallel_for(chunks, 1, [&](int64_t b, int64_t e) {
                        for (int64_t c = b; c < e; ++c) {
                            uint8_t *p = const_cast<uint8_t *>(map_) + size_t(c) * CH;
                            const size_t n = std::min(CH, size_ - size_t(c) * CH);
                            if (madvise(p, n, 22 /* MADV_POPULATE_READ */) != 0) {  // older kernels: touch the pages
                                volatile uint8_t sink = 0;
                                for (size_t o = 0; o < n; o += 4096) sink = sink + p[o];
                            }
                        }
                    });
                    read_s = since(t0);
                    off_ = sizeof h;
                    mode_ = Mode::Replay;
                }
            }
            ::close(fd);
        }
        if (mode_ == Mode::Off) {
            out_ = std::fopen((path_ + ".tmp").c_str(), "wb");
            if (!out_) {
                log("weight cache: cannot write %s.tmp, loading without the cache", path_.c_str());
                return;
            }
            Head h{};
            std::memcpy(h.magic, MAGIC, 8);
            h.key = key;
            std::fwrite(&h, sizeof h, 1, out_);
            mode_ = Mode::Record;
        }
    }
    ~WeightCache() {
        if (map_) munmap(const_cast<uint8_t *>(map_), size_);
        if (out_) {  // not finished (an exception during the load): the partial file is dropped
            std::fclose(out_);
            std::remove((path_ + ".tmp").c_str());
        }
    }
    WeightCache(const WeightCache &) = delete;
    WeightCache &operator=(const WeightCache &) = delete;
    Mode mode() const { return mode_; }

    template <typename T, typename F>
    T *item(F &&make) {
        const auto t0 = std::chrono::steady_clock::now();
        if (mode_ == Mode::Replay) {
            Piece pc{};
            QW_CHECK(off_ + sizeof pc <= size_, "weight cache " + path_ + ": truncated (delete it)");
            std::memcpy(&pc, map_ + off_, sizeof pc);
            const size_t bytes = size_t(pc.count) * sizeof(T);
            QW_CHECK(pc.elem == sizeof(T) && off_ + sizeof pc + bytes <= size_,
                     "weight cache " + path_ + ": a piece does not match what the loader expects (delete it)");
            T *d;
            CK(hipMalloc(&d, std::max<size_t>(bytes, 1)));
            if (bytes) CK(hipMemcpy(d, map_ + off_ + sizeof pc, bytes, hipMemcpyHostToDevice));
            off_ += sizeof pc + (bytes + 63) / 64 * 64;
            ++pieces_;
            upload_s += since(t0);
            return d;
        }
        const std::vector<T> v = make();
        const auto t1 = std::chrono::steady_clock::now();
        convert_s += std::chrono::duration<double>(t1 - t0).count();
        T *d = upload(v);
        upload_s += since(t1);
        if (mode_ == Mode::Record) {
            const auto t2 = std::chrono::steady_clock::now();
            const Piece pc{uint32_t(sizeof(T)), 0, uint64_t(v.size())};
            const size_t bytes = v.size() * sizeof(T);
            static const char zeros[64] = {};
            bool ok = std::fwrite(&pc, sizeof pc, 1, out_) == 1 && (!bytes || std::fwrite(v.data(), bytes, 1, out_) == 1);
            ok = ok && ((bytes % 64) == 0 || std::fwrite(zeros, 64 - bytes % 64, 1, out_) == 1);
            if (!ok) {
                log("weight cache: writing %s.tmp failed (disk full?), continuing without it", path_.c_str());
                std::fclose(out_);
                out_ = nullptr;
                std::remove((path_ + ".tmp").c_str());
                mode_ = Mode::Off;
            }
            ++pieces_;
            write_s += since(t2);
        }
        return d;
    }
    // The loader is done: a recording becomes the cache file; a replay must have used every piece.
    void finish() {
        if (mode_ == Mode::Replay) {
            uint64_t n = 0;
            QW_CHECK(off_ + sizeof n == size_ && (std::memcpy(&n, map_ + off_, sizeof n), n == pieces_),
                     "weight cache " + path_ + ": not the pieces the loader expects (delete it)");
        } else if (mode_ == Mode::Record) {
            const bool ok = std::fwrite(&pieces_, sizeof pieces_, 1, out_) == 1 && std::fflush(out_) == 0;
            std::fclose(out_);
            out_ = nullptr;
            if (ok && std::rename((path_ + ".tmp").c_str(), path_.c_str()) == 0) return;
            log("weight cache: could not finish %s", path_.c_str());
            std::remove((path_ + ".tmp").c_str());
        }
    }
    double convert_s = 0, upload_s = 0, write_s = 0, read_s = 0;

private:
    static double since(std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    }
    static constexpr char MAGIC[9] = "QWWEIGH1";
    struct Head {
        char magic[8];
        uint64_t key;
        uint8_t pad[48];
    };
    struct Piece {
        uint32_t elem, pad;
        uint64_t count;
    };
    Mode mode_ = Mode::Off;
    std::string path_;
    uint64_t pieces_ = 0;
    std::FILE *out_ = nullptr;
    const uint8_t *map_ = nullptr;
    size_t size_ = 0, off_ = 0;
};

// The int8 copy of an fp16 [N][K] matrix for decode (QW_INT8_DENSE=1), else none.
// QW_INT8_SKIP: comma-separated groups kept fp16 (hc, qsa, gdn, shared, lm_head, ple, mtp).
// QW_INT8_DIR: GPTQ-quantized copies (tools/gptq_int8.py), r<rank>/<name>.q8, used when
// present (else round-to-nearest). QW_INT8_DUMP: writes the fp16 matrices, r<rank>/<name>.f16,
// as the GPTQ tool's input.
Q8 q8(const std::vector<uint16_t> &m, int64_t N, int64_t K, const char *group, int r, const std::string &name) {
    auto env = [](const char *v) { return std::getenv(v) ? std::string(std::getenv(v)) : std::string(); };
    static const std::string dump = env("QW_INT8_DUMP"), dir = env("QW_INT8_DIR");
    const std::string sub = "/r" + std::to_string(r) + "/";
    if (!dump.empty()) {
        std::filesystem::create_directories(dump + sub);
        std::FILE *f = std::fopen((dump + sub + name + ".f16").c_str(), "wb");
        QW_CHECK(f, "cannot write " + dump + sub + name + ".f16");
        const int64_t hdr[2] = {N, K};
        std::fwrite(hdr, sizeof hdr, 1, f);
        std::fwrite(m.data(), 2, m.size(), f);
        std::fclose(f);
    }
    static const bool on = env("QW_INT8_DENSE") != "" && env("QW_INT8_DENSE") != "0";
    static const std::string skip = "," + env("QW_INT8_SKIP") + ",";
    if (!on || skip.find(std::string(",") + group + ",") != std::string::npos) return {};
    std::vector<int8_t> q;
    std::vector<float> sc;
    if (!dir.empty())
        if (std::FILE *f = std::fopen((dir + sub + name + ".q8").c_str(), "rb")) {
            int64_t hdr[3] = {};
            bool ok = std::fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == N && hdr[1] == K && hdr[2] == gpu::I8_GROUP;
            if (ok) {
                q.resize(size_t(N * K));
                sc.resize(size_t(N * K / gpu::I8_GROUP));
                ok = std::fread(q.data(), 1, q.size(), f) == q.size() &&
                     std::fread(sc.data(), 4, sc.size(), f) == sc.size();
            }
            std::fclose(f);
            QW_CHECK(ok, "bad GPTQ file " + dir + sub + name + ".q8");
            return {upload(q), upload(sc)};
        }
    if (!dir.empty()) log("int8: no GPTQ copy of %s (rank %d), rounding to nearest", name.c_str(), r);
    quantize_rows_i8(m, N, K, q, sc, gpu::I8_GROUP);
    return {upload(q), upload(sc)};
}

// Short name of a layer's matrices: L05, mtp.
std::string layer_name(const std::string &prefix) {
    if (prefix.rfind("mtp.", 0) == 0) return "mtp";
    const auto d = prefix.find_first_of("0123456789");
    char buf[16];
    std::snprintf(buf, sizeof buf, "L%02d", std::atoi(prefix.c_str() + d));
    return buf;
}

// Hyper-connection mixer weights; inject: the block-inject rows (not in final mixers).
HcW load_hc(WeightCache &wc, const SafeTensors &st, int r, const std::string &p, bool inject, const std::string &name) {
    HcW w;
    w.w1 = wc.item<float>([&] { return shard_w1(st.get(p + "hc_norm.weight"), r); });
    std::vector<uint16_t> df;  // the prefill layout of `down`, built with it
    w.down = wc.item<uint16_t>([&] {
        const TensorView &down = st.get(p + "input_mix_weight_down.weight", DType::BF16, {HC_RANK, HC * H});
        std::vector<uint16_t> d;
        d.reserve(size_t(HC) * HC_DOWN * SH);
        for (int s = 0; s < HC; ++s) {
            add_rows(d, down, 0, HC_RANK, s * H + r * SH, SH);
            if (inject)
                add_rows(d, st.get(p + "block_inject_weight.weight", DType::BF16, {HC, HC * H}), 0, HC, s * H + r * SH, SH);
            else
                d.resize(d.size() + size_t(HC) * SH, 0);
        }
        w.down8 = q8(d, int64_t(HC) * HC_DOWN, SH, "hc", r, name + ".down");
        df.assign(size_t(gpu::D_PAD) * HC * SH, 0);
        for (int s2 = 0; s2 < HC; ++s2)
            for (int o = 0; o < HC_DOWN; ++o)
                std::copy(d.begin() + (size_t(s2) * HC_DOWN + o) * SH, d.begin() + (size_t(s2) * HC_DOWN + o + 1) * SH,
                          df.begin() + size_t(o) * HC * SH + size_t(s2) * SH);
        return d;
    });
    w.down_flat = wc.item<uint16_t>([&] { return std::move(df); });
    w.up = wc.item<uint16_t>([&] {
        const TensorView &up = st.get(p + "input_mix_weight_up.weight", DType::BF16, {HC * H, HC_RANK});
        std::vector<uint16_t> u;
        u.reserve(size_t(SH) * HC * HC_RANK);
        for (int j = 0; j < SH; ++j)
            for (int s = 0; s < HC; ++s) add_rows(u, up, s * H + r * SH + j, 1);
        return u;
    });
    return w;
}

// One decoder layer. The MTP layer's routed experts are bf16 (fused
// gate_up): they are quantized here to the same int4 g128 layout (they only
// draft tokens, which the full model verifies).
void load_layer(WeightCache &wc, LayerW &w, const SafeTensors &st, ThreadPool &pool, int r, const std::string &p,
                bool qsa, bool bf16_experts) {
    const std::string ln = layer_name(p);
    w.attn_hc = load_hc(wc, st, r, p + "attn_hyper_connection.", true, ln + ".attn_hc");
    w.mlp_hc = load_hc(wc, st, r, p + "mlp_hyper_connection.", true, ln + ".mlp_hc");
    const auto norm1 = [&](const std::string &name) {
        return wc.item<float>([&] { return plus_one(f32_of(st.get(name))); });
    };
    if (qsa) {
        const std::string a = p + "self_attn.";
        w.qsa_proj = wc.item<uint16_t>([&] {
            std::vector<uint16_t> m;
            const TensorView &q = st.get(a + "q_proj.weight", DType::BF16, {Q_HEADS * 2 * HEAD_DIM, H});
            add_rows(m, q, int64_t(r) * QH_L * 2 * HEAD_DIM, QH_L * 2 * HEAD_DIM);
            const int kvh = r / (RANKS / KV_HEADS);
            add_rows(m, st.get(a + "k_proj.weight"), kvh * HEAD_DIM, HEAD_DIM);
            add_rows(m, st.get(a + "v_proj.weight"), kvh * HEAD_DIM, HEAD_DIM);
            add_rows(m, st.get(a + "indexer.index_qk_proj.weight"), 0, (IDX_HEADS + 1) * IDX_DIM);
            QW_CHECK(m.size() == size_t(QSA_PROJ_L) * H, "qsa proj size");
            w.qsa_proj8 = q8(m, QSA_PROJ_L, H, "qsa", r, ln + ".qsa_proj");
            return m;
        });
        w.qn = norm1(a + "q_norm.weight");
        w.kn = norm1(a + "k_norm.weight");
        w.iqn = norm1(a + "indexer.q_layernorm.weight");
        w.ikn = norm1(a + "indexer.k_layernorm.weight");
        w.qsa_o = wc.item<uint16_t>([&] {
            std::vector<uint16_t> o;
            add_rows(o, st.get(a + "o_proj.weight", DType::BF16, {H, Q_HEADS * HEAD_DIM}), 0, H,
                     int64_t(r) * QH_L * HEAD_DIM, QH_L * HEAD_DIM);
            w.qsa_o8 = q8(o, H, int64_t(QH_L) * HEAD_DIM, "qsa", r, ln + ".qsa_o");
            return o;
        });
    } else {
        const std::string a = p + "linear_attn.";
        w.gdn_proj = wc.item<uint16_t>([&] {
            const TensorView &qkv = st.get(a + "in_proj_qkv.weight", DType::BF16, {GDN_QKV, H});
            std::vector<uint16_t> m;
            add_rows(m, qkv, int64_t(r) * GDN_KL * GDN_DIM, GDN_KL * GDN_DIM);
            add_rows(m, qkv, GDN_KEY + int64_t(r) * GDN_KL * GDN_DIM, GDN_KL * GDN_DIM);
            add_rows(m, qkv, 2 * GDN_KEY + int64_t(r) * GDN_VL * GDN_DIM, GDN_VL * GDN_DIM);
            add_rows(m, st.get(a + "in_proj_z.weight"), int64_t(r) * GDN_VL * GDN_DIM, GDN_VL * GDN_DIM);
            add_rows(m, st.get(a + "in_proj_b.weight"), int64_t(r) * GDN_VL, GDN_VL);
            add_rows(m, st.get(a + "in_proj_a.weight"), int64_t(r) * GDN_VL, GDN_VL);
            QW_CHECK(m.size() == size_t(GDN_PROJ_L) * H, "gdn proj size");
            w.gdn_proj8 = q8(m, GDN_PROJ_L, H, "gdn", r, ln + ".gdn_proj");
            return m;
        });
        w.gdn_conv = wc.item<float>([&] {
            auto cw = f32_of(st.get(a + "conv1d.weight", DType::BF16, {GDN_QKV, 1, GDN_CONV}));
            std::vector<float> lc;
            auto take = [&](int64_t c0, int64_t n) {
                lc.insert(lc.end(), cw.begin() + c0 * GDN_CONV, cw.begin() + (c0 + n) * GDN_CONV);
            };
            take(int64_t(r) * GDN_KL * GDN_DIM, GDN_KL * GDN_DIM);
            take(GDN_KEY + int64_t(r) * GDN_KL * GDN_DIM, GDN_KL * GDN_DIM);
            take(2 * GDN_KEY + int64_t(r) * GDN_VL * GDN_DIM, GDN_VL * GDN_DIM);
            return lc;
        });
        w.gdn_A = wc.item<float>([&] {
            auto A = f32_of(st.get(a + "A_log"));
            return std::vector<float>(A.begin() + r * GDN_VL, A.begin() + (r + 1) * GDN_VL);
        });
        w.gdn_dt = wc.item<float>([&] {
            auto dt = f32_of(st.get(a + "dt_bias"));
            return std::vector<float>(dt.begin() + r * GDN_VL, dt.begin() + (r + 1) * GDN_VL);
        });
        w.gdn_norm = wc.item<float>([&] { return f32_of(st.get(a + "norm.weight")); });
        w.gdn_out = wc.item<uint16_t>([&] {
            std::vector<uint16_t> o;
            add_rows(o, st.get(a + "out_proj.weight", DType::BF16, {H, GDN_VAL}), 0, H, int64_t(r) * GDN_VL * GDN_DIM,
                     GDN_VL * GDN_DIM);
            w.gdn_out8 = q8(o, H, int64_t(GDN_VL) * GDN_DIM, "gdn", r, ln + ".gdn_out");
            return o;
        });
    }
    // MoE
    const std::string mp = p + "mlp.";
    w.router = wc.item<uint16_t>([&] {
        std::vector<uint16_t> router;
        add_rows(router, st.get(mp + "gate.weight", DType::BF16, {N_EXPERTS, H}), 0, N_EXPERTS);
        return router;
    });
    // A projection of the rank's experts: the packed int4 words, then their fp16 scales (built together).
    auto pack_experts = [&](const char *proj, int rows, int K, uint32_t **wd, uint16_t **sd) {
        const size_t words = size_t(rows) * K / 8, groups = size_t(rows) * K / QGROUP;
        std::vector<uint16_t> scales;
        *wd = wc.item<uint32_t>([&] {
            std::vector<uint32_t> packed(size_t(EXP_L) * words);
            scales.resize(size_t(EXP_L) * groups);
            pool.parallel_for(EXP_L, 1, [&](int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) {
                    const std::string ep = mp + "experts." + std::to_string(r * EXP_L + i) + "." + proj;
                    const TensorView &pw = st.get(ep + ".weight_packed", DType::I32, {rows, K / 8});
                    const TensorView &sw = st.get(ep + ".weight_scale", DType::BF16, {rows, K / QGROUP});
                    pack_int4_words(pw.i32(), packed.data() + size_t(i) * words, words);
                    for (size_t g = 0; g < groups; ++g) scales[size_t(i) * groups + g] = bf16_to_f16(sw.u16()[g]);
                }
            });
            return packed;
        });
        *sd = wc.item<uint16_t>([&] { return std::move(scales); });
    };
    // [E][R][K] bf16 rows [row0, row0 + rows) of each local expert -> int4 g128 (symmetric, scale amax / 7)
    auto quant_experts = [&](const std::string &tensor, std::vector<int64_t> shape, int row0, int rows, int K,
                             uint32_t **wd, uint16_t **sd) {
        const size_t words = size_t(rows) * K / 8, groups = size_t(rows) * K / QGROUP;
        std::vector<uint16_t> scales;
        *wd = wc.item<uint32_t>([&] {
            const TensorView &t = st.get(tensor, DType::BF16, shape);
            std::vector<uint32_t> packed(size_t(EXP_L) * words);
            scales.resize(size_t(EXP_L) * groups);
            const size_t R = size_t(t.dim(1));
            pool.parallel_for(EXP_L, 1, [&](int64_t b0, int64_t e0) {
                std::vector<int32_t> ct(words);
                for (int64_t i = b0; i < e0; ++i) {
                    const uint16_t *src = t.u16() + (size_t(r * EXP_L + i) * R + size_t(row0)) * size_t(K);
                    quantize_int4_g128(src, groups, ct.data(), scales.data() + size_t(i) * groups);
                    pack_int4_words(ct.data(), packed.data() + size_t(i) * words, words);
                }
            });
            return packed;
        });
        *sd = wc.item<uint16_t>([&] { return std::move(scales); });
    };
    if (bf16_experts) {
        const std::string gu = mp + "experts.gate_up_proj";
        quant_experts(gu, {N_EXPERTS, 2 * FFN, H}, 0, FFN, H, &w.eg, &w.egs);
        quant_experts(gu, {N_EXPERTS, 2 * FFN, H}, FFN, FFN, H, &w.eu, &w.eus);
        quant_experts(mp + "experts.down_proj", {N_EXPERTS, H, FFN}, 0, H, FFN, &w.ed, &w.eds);
    } else {
        pack_experts("gate_proj", FFN, H, &w.eg, &w.egs);
        pack_experts("up_proj", FFN, H, &w.eu, &w.eus);
        pack_experts("down_proj", H, FFN, &w.ed, &w.eds);
    }
    w.sh_gu = wc.item<uint16_t>([&] {
        std::vector<uint16_t> gu;
        add_rows(gu, st.get(mp + "shared_expert.gate_proj.weight"), int64_t(r) * SFFN_L, SFFN_L);
        add_rows(gu, st.get(mp + "shared_expert.up_proj.weight"), int64_t(r) * SFFN_L, SFFN_L);
        add_rows(gu, st.get(mp + "shared_expert_gate.weight"), 0, 1);
        w.sh_gu8 = q8(gu, 2 * SFFN_L + 1, H, "shared", r, ln + ".sh_gu");
        return gu;
    });
    w.sh_down = wc.item<uint16_t>([&] {
        std::vector<uint16_t> sd;
        add_rows(sd, st.get(mp + "shared_expert.down_proj.weight", DType::BF16, {H, FFN}), 0, H, int64_t(r) * SFFN_L,
                 SFFN_L);
        w.sh_down8 = q8(sd, H, SFFN_L, "shared", r, ln + ".sh_down");
        return sd;
    });
}

}  // namespace

void load_rank_weights(Rank &rk, const SafeTensors &st, ThreadPool &pool, bool mtp, const std::string &cache_dir) {
    const int r = rk.r;
    const auto t0 = std::chrono::steady_clock::now();
    // The cache's key: the checkpoint's files (path, size, modification time), the rank, the MTP head, the layout.
    // An int8 run (QW_INT8_*) converts differently and reads the fp16 matrices: no cache then.
    uint64_t key = 1469598103934665603ull;
    const auto mix = [&](const void *p, size_t n) {
        for (size_t i = 0; i < n; ++i) key = (key ^ static_cast<const uint8_t *>(p)[i]) * 1099511628211ull;
    };
    const uint64_t fields[] = {WEIGHT_CACHE_VERSION, uint64_t(r), uint64_t(mtp), uint64_t(RANKS), uint64_t(H),
                               uint64_t(N_LAYERS), uint64_t(N_EXPERTS), uint64_t(VOCAB)};
    mix(fields, sizeof fields);
    for (const auto &f : st.files()) {
        struct stat sb {};
        stat(f->path().c_str(), &sb);
        const uint64_t meta[] = {uint64_t(f->size()), uint64_t(sb.st_mtime)};
        mix(f->path().data(), f->path().size());
        mix(meta, sizeof meta);
    }
    const bool int8 = std::getenv("QW_INT8_DENSE") || std::getenv("QW_INT8_DUMP") || std::getenv("QW_INT8_DIR");
    WeightCache wc(int8 ? std::string() : cache_dir, key, r, pool);
    rk.L.resize(N_LAYERS);
    for (int l = 0; l < N_LAYERS; ++l) load_layer(wc, rk.L[size_t(l)], st, pool, r, layer_prefix(l), is_qsa(l), false);
    if (mtp) {
        load_layer(wc, rk.mtp, st, pool, r, "mtp.layers.0.", true, true);
        rk.mtp_final = load_hc(wc, st, r, "mtp.hyper_connection_mixer.", false, "mtp.final_hc");
        rk.mtp_fc_h = wc.item<uint16_t>([&] {
            std::vector<uint16_t> fh;
            add_rows(fh, st.get("mtp.fc_hidden.weight", DType::BF16, {H, H}), 0, H, int64_t(r) * SH, SH);
            rk.mtp_fc_h8 = q8(fh, H, SH, "mtp", r, "mtp.fc_h");
            return fh;
        });
        rk.mtp_fc_e = wc.item<uint16_t>([&] {
            std::vector<uint16_t> fe;
            add_rows(fe, st.get("mtp.fc_embedding.weight", DType::BF16, {H, H}), 0, H, int64_t(r) * SH, SH);
            rk.mtp_fc_e8 = q8(fe, H, SH, "mtp", r, "mtp.fc_e");
            return fe;
        });
        rk.mtp_nh = wc.item<float>([&] { return shard_w1(st.get("mtp.pre_fc_norm_hidden.weight"), r); });
        rk.mtp_ne = wc.item<float>([&] {
            auto ne = plus_one(f32_of(st.get("mtp.pre_fc_norm_embedding.weight")));
            return std::vector<float>(ne.begin() + r * SH, ne.begin() + (r + 1) * SH);
        });
    }
    rk.final_hc = load_hc(wc, st, r, "model.language_model.hyper_connection_mixer.", false, "final_hc");
    rk.lm_head = wc.item<uint16_t>([&] {
        std::vector<uint16_t> head;
        add_rows(head, st.get("lm_head.weight", DType::BF16, {VOCAB, H}), int64_t(r) * VOCAB_L, VOCAB_L);
        rk.lm_head8 = q8(head, VOCAB_L, H, "lm_head", r, "lm_head");
        return head;
    });

    // PLE
    const std::string pp = layer_prefix(PLE_LAYER) + "ple.";
    rk.ple_kv = wc.item<uint16_t>([&] {
        std::vector<uint16_t> kv;
        const TensorView &kp = st.get(pp + "key_proj.weight", DType::BF16, {HC * H, H});
        for (int s = 0; s < HC; ++s) add_rows(kv, kp, s * H + r * SH, SH);
        add_rows(kv, st.get(pp + "value_proj.weight", DType::BF16, {H, H}), r * SH, SH);
        rk.ple_kv8 = q8(kv, PLE_KV, H, "ple", r, "ple_kv");
        return kv;
    });
    rk.ple_nk = wc.item<float>([&] { return shard_w1(st.get(pp + "norm_key.weight"), r); });
    rk.ple_nq = wc.item<float>([&] { return shard_w1(st.get(pp + "norm_query.weight"), r); });
    rk.ple_nc = wc.item<float>([&] { return shard_w1(st.get(pp + "norm_conv.weight"), r); });
    rk.ple_conv = wc.item<float>([&] {
        auto pc = f32_of(st.get(pp + "conv1d.weight", DType::BF16, {HC * H, 1, PLE_CONV}));
        std::vector<float> plc(size_t(HC) * SH * PLE_CONV);
        for (int s = 0; s < HC; ++s)
            for (int j = 0; j < SH; ++j)
                for (int k = 0; k < PLE_CONV; ++k)
                    plc[size_t((s * SH + j) * PLE_CONV + k)] = pc[size_t((s * H + r * SH + j) * PLE_CONV + k)];
        return plc;
    });
    const WeightCache::Mode mode = wc.mode();
    wc.finish();
    log("rank %d weights in %.1f s (%s: convert %.1f s, upload %.1f s%s)", r,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
        mode == WeightCache::Mode::Replay ? "from the weight cache" : mode == WeightCache::Mode::Record ? "written to the weight cache" : "no weight cache",
        wc.convert_s, wc.upload_s,
        mode == WeightCache::Mode::Record   ? (", cache write " + std::to_string(wc.write_s).substr(0, 5) + " s").c_str()
        : mode == WeightCache::Mode::Replay ? (", cache read " + std::to_string(wc.read_s).substr(0, 5) + " s").c_str()
                                            : "");
}

}  // namespace qw
