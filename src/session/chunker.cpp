#include "session/chunker.hpp"

namespace qw {

namespace {

uint64_t mix(uint64_t h, uint32_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h * 0xff51afd7ed558ccdull;
}

}  // namespace

std::vector<Chunk> chunk_prompt(const std::vector<int32_t> &tokens, int32_t boundary) {
    constexpr int WINDOW = 16;
    std::vector<Chunk> out;
    const int64_t n = int64_t(tokens.size());
    int64_t start = 0;
    uint64_t h = 0;
    auto cut = [&](int64_t end) {
        if (end > start) out.push_back({start, end - start, h});
        start = end;
        h = 0;
    };
    for (int64_t i = 0; i < n; ++i) {
        if (tokens[size_t(i)] == boundary && i > start) cut(i);  // a message starts here
        h = mix(h, uint32_t(tokens[size_t(i)]));
        const int64_t len = i + 1 - start;
        if (len >= CHUNK_MAX) {
            cut(i + 1);
            continue;
        }
        if (len >= CHUNK_MIN && i + 1 >= WINDOW) {  // content-defined cut: hash of the last WINDOW tokens
            uint64_t w = 0;
            for (int64_t j = i + 1 - WINDOW; j <= i; ++j) w = mix(w, uint32_t(tokens[size_t(j)]));
            if (w % CHUNK_AVG == 0) cut(i + 1);
        }
    }
    cut(n);
    return out;
}

}  // namespace qw
