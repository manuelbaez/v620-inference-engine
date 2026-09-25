// Position-independent chunks of a prompt, for non-prefix reuse: cut at chat
// message boundaries, and inside long messages at content-defined points (a
// rolling hash of the last tokens), so an edit only changes the chunks it
// touches and a document keeps its chunks wherever it appears.
#pragma once

#include <cstdint>
#include <vector>

namespace qw {

struct Chunk {
    int64_t start, len;
    uint64_t hash;  // of the tokens only (not the position or what precedes)
};

// boundary < 0: no message boundaries. Chunks are MIN..MAX tokens (shorter at
// message ends), ~AVG on average inside long messages.
std::vector<Chunk> chunk_prompt(const std::vector<int32_t> &tokens, int32_t boundary);

constexpr int CHUNK_MIN = 128, CHUNK_AVG = 512, CHUNK_MAX = 2048;

}  // namespace qw
