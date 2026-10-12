// Session::suspend: a prompt part-way through its prefill gives up its slot, another request uses the slot, and the
// prompt continues from the block store when it has a slot again. Its logits at the end must be bit-identical to
// the same prompt prefilled in the same pieces without the interruption (a restore from the store is exact).
// Twice: suspended right after a piece (the pool still has that piece's snapshot of the slot), and after other
// slots' prefills have taken every snapshot of the pool (suspend then snapshots the slot itself).
// Usage: test_suspend [--n 9000] [--piece 1024]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "engine/engine.hpp"
#include "session/session.hpp"

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

int main(int argc, char **argv) {
    int n = 9000, piece = 1024;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--n" && i + 1 < argc) n = std::atoi(argv[++i]);
        if (std::string(argv[i]) == "--piece" && i + 1 < argc) piece = std::atoi(argv[++i]);
    }
    try {
        EngineOptions opt;
        opt.slot_tokens = {16384, 16384, 16384, 16384};
        opt.prefill_chunk = 4096;
        opt.warmup = false;
        Engine e(opt);
        setenv("QW_HOST_CACHE_GB", "8", 1);
        unsetenv("QW_DISK_CACHE_DIR");
        int fails = 0;
        const auto check = [&](bool ok, const char *what) {
            std::printf("%-86s %s\n", what, ok ? "ok" : "FAIL");
            fails += !ok;
        };
        const int at = 3 * piece;  // where the prompt is suspended
        for (int round = 0; round < 2; ++round) {
            const auto A = text(n, 11 + round), small = text(60, 21 + round), other = text(piece, 31 + round);
            // the prompt in the same pieces, without interruption (engine calls, no store)
            e.slot_reset(3);
            std::vector<float> ref;
            for (int p0 = 0; p0 < n; p0 += piece)
                ref = e.prefill(3, std::vector<int32_t>(A.begin() + p0, A.begin() + std::min(n, p0 + piece)));
            e.slot_reset(3);

            Session s(e);
            // three slots are busy, so the prompt and the short request have to use the same one
            for (int i = 0; i < 3; ++i) {
                const auto d = text(200, 41 + 10 * round + i);
                s.set_prompt(s.acquire(d, 16), d);
            }
            const int slot = s.acquire(A, 16);
            s.begin_prompt(slot, A);
            for (int i = 0; i < 3; ++i) s.prefill_some(slot, piece);
            if (round == 1)  // other slots' prefills use up the snapshot pool
                for (int i = 0; i < 2 * Engine::SNAPSHOTS; ++i) {
                    const int o = (slot + 1 + i % 3) % 4;
                    const auto d = text(300 + i, 61 + i);
                    s.set_prompt(o, d);
                }
            const int64_t saved = s.suspend(slot);
            s.release(slot);
            const int slot_s = s.acquire(small, 16);
            s.set_prompt(slot_s, small);
            s.release(slot_s);
            const int slot2 = s.acquire(A, 16);
            const int64_t cached = s.begin_prompt(slot2, A);
            bool done = false;
            while (!done) done = s.prefill_some(slot2, piece);
            const std::vector<float> got = e.logits();
            std::printf("round %d: suspended in slot %d at %d tokens (%lld saved), the short request took slot %d, resumed in "
                        "slot %d with %lld from the store\n",
                        round, slot, at, (long long)saved, slot_s, slot2, (long long)cached);
            check(saved == at && cached == at, "the whole prefilled part was saved and restored");
            check(slot_s == slot, "the short request used the prompt's slot");
            check(got.size() == ref.size() && std::memcmp(got.data(), ref.data(), ref.size() * sizeof(float)) == 0,
                  round ? "after the pool's snapshots were taken: logits identical to the uninterrupted prefill"
                        : "logits identical to the uninterrupted prefill");
            (void)other;
            for (int sl = 0; sl < 4; ++sl) e.slot_reset(sl);
        }
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
