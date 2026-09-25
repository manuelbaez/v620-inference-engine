// PLE n-gram hashing and the int4 host-RAM embedding table (docs/MODEL.md, "PLE").
// The table stays in host memory in every engine configuration: it is ~30 GB
// and only 16 rows are touched per token.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/safetensors.hpp"

namespace qw {

using NgramIds = std::array<int64_t, cfg::NGRAM_HEADS>;

class NgramHasher {
public:
    NgramHasher();
    // Ids for the token at the end of `ctx`: ctx[n-1] is the current token,
    // ctx[n-2] and ctx[n-3] the previous ones. `n` may be smaller than 3 at the
    // start of a sequence. `prev_eos` is the index in ctx of the latest EOS
    // strictly before ctx[n-1], or -1 if there is none in ctx.
    NgramIds ids(const int32_t *ctx, int n, int prev_eos) const;

    // Ids for every position of a token sequence, with `history` the (up to
    // 2) tokens that precede tokens[0] in the same sequence.
    std::vector<NgramIds> ids_for(const std::vector<int32_t> &history,
                                  const std::vector<int32_t> &tokens) const;

    const std::array<int64_t, cfg::NGRAM_HEADS> &sizes() const { return sizes_; }
    const std::array<uint64_t, cfg::NGRAM> &multipliers() const { return mult_; }

private:
    std::array<uint64_t, cfg::NGRAM> mult_{};
    std::array<int64_t, cfg::NGRAM_HEADS> sizes_{}, offsets_{};
};

class PleTable {
public:
    // Opens <dir>/META.json and the shard_*.safetensors files.
    explicit PleTable(const std::string &dir);

    int64_t rows() const { return rows_; }
    // Dequantizes one 160-value row into out.
    void row(int64_t id, float *out) const;
    // Gathers all 16 heads for one token into out[2560].
    void gather(const NgramIds &ids, float *out) const;
    // Asks the kernel to page the table in.
    void prefetch() const;

private:
    int64_t rows_ = 0, rows_per_shard_ = 0;
    std::vector<std::unique_ptr<SafeTensors>> shards_;
    std::vector<const uint8_t *> q_;       // [rows_per_shard, 80]
    std::vector<const uint16_t *> scale_;  // [rows_per_shard, 10] fp16
};

}  // namespace qw
