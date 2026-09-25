// Chunker: chunks tile the prompt, respect the size bounds and message
// boundaries, and a document keeps its chunks at another position and after
// an edit elsewhere.
#include <cstdio>
#include <set>
#include <vector>

#include "session/chunker.hpp"

using namespace qw;

static std::vector<int32_t> text(int n, int seed) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    uint32_t x = uint32_t(seed) * 2654435761u + 1;
    for (auto &t : v) {
        x = x * 1664525u + 1013904223u;
        t = int32_t(100 + (x >> 8) % 30000);
    }
    return v;
}

int main() {
    constexpr int32_t B = 7;
    int fails = 0;
    const auto doc = text(6000, 1), a = text(900, 2), b = text(1700, 3);
    auto join = [&](std::vector<std::vector<int32_t>> parts) {
        std::vector<int32_t> p;
        for (auto &x : parts) {
            p.push_back(B);
            p.insert(p.end(), x.begin(), x.end());
        }
        return p;
    };
    const auto p1 = join({a, doc}), p2 = join({b, doc});
    const auto c1 = chunk_prompt(p1, B), c2 = chunk_prompt(p2, B);
    int64_t pos = 0;
    for (const Chunk &c : c1) {
        fails += c.start != pos || c.len > CHUNK_MAX;
        pos += c.len;
    }
    fails += pos != int64_t(p1.size());
    std::set<uint64_t> h1;
    int64_t doc_tokens = 0, shared = 0;
    for (const Chunk &c : c1)
        if (c.start > int64_t(a.size()) + 1) {
            h1.insert(c.hash);
            doc_tokens += c.len;
        }
    for (const Chunk &c : c2)
        if (h1.count(c.hash)) shared += c.len;
    std::printf("%zu chunks; the document's chunks found at another position: %lld of %lld tokens\n", c1.size(),
                (long long)shared, (long long)doc_tokens);
    fails += shared != doc_tokens;
    // an edit inside the document changes only nearby chunks
    auto edited = doc;
    edited[3000] = 12345;
    const auto c3 = chunk_prompt(join({a, edited}), B);
    int64_t kept = 0;
    std::set<uint64_t> all1;
    for (const Chunk &c : c1) all1.insert(c.hash);
    for (const Chunk &c : c3)
        if (all1.count(c.hash)) kept += c.len;
    std::printf("after a one-token edit: %lld of %zu tokens still in known chunks\n", (long long)kept, p1.size());
    fails += kept < int64_t(p1.size()) - CHUNK_MAX * 2;
    std::printf("%s\n", fails ? "FAILED" : "all ok");
    return fails ? 1 : 0;
}
