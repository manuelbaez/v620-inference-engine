// The sharded indexer selection (qsa_select_local_T per shard, qsa_select_merge_T over the shards' candidates) against
// the selection over all groups (qsa_select_T): the same token lists, for queries that see fewer groups than a shard
// holds, shards with fewer than 512 visible groups, the dense case, and scores full of ties (ties go to the lowest
// group). One device plays the four ranks in turn.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "kernels/qsa.hpp"

using namespace qw;

#define CK(x)                                                                           \
    do {                                                                                \
        hipError_t err_ = (x);                                                          \
        if (err_ != hipSuccess) fail(std::string(#x) + ": " + hipGetErrorString(err_)); \
    } while (0)

int main() {
    try {
        CK(hipSetDevice(0));
        std::mt19937 rng(11);
        const int Q = 64, K = cfg::IDX_TOPK_BLOCKS;
        int fails = 0;
        // first query position; levels: distinct score values (few: many ties; 0: continuous)
        const struct {
            int64_t q0;
            int levels;
        } cases[] = {{2000, 0}, {2040, 3}, {2100, 0}, {4100, 7}, {8190, 0}, {8200, 1}, {20000, 50}, {100003, 0}, {262144, 5}};
        for (const auto &c : cases) {
            const int nb_max = int((c.q0 + Q) / cfg::IDX_RATIO), ld = (nb_max + 7) / 8 * 8;
            std::vector<float> sc(size_t(Q) * ld);
            for (auto &v : sc)
                v = c.levels ? float(rng() % unsigned(c.levels)) * 0.25f : std::uniform_real_distribution<float>(0, 8)(rng);
            float *d_sc, *d_loc;
            int32_t *d_ref, *d_refn, *d_got, *d_gotn;
            gpu::IdxCand *d_cand, *d_all;
            const int per = ((nb_max + cfg::RANKS - 1) / cfg::RANKS + 7) / 8 * 8;
            CK(hipMalloc(&d_sc, sc.size() * 4));
            CK(hipMalloc(&d_loc, size_t(Q) * per * 4));
            CK(hipMalloc(&d_ref, size_t(Q) * gpu::LIST_W * 4));
            CK(hipMalloc(&d_got, size_t(Q) * gpu::LIST_W * 4));
            CK(hipMalloc(&d_refn, size_t(Q) * 4));
            CK(hipMalloc(&d_gotn, size_t(Q) * 4));
            CK(hipMalloc(&d_cand, size_t(Q) * K * sizeof(gpu::IdxCand)));
            CK(hipMalloc(&d_all, size_t(Q) * cfg::RANKS * K * sizeof(gpu::IdxCand)));
            CK(hipMemcpy(d_sc, sc.data(), sc.size() * 4, hipMemcpyHostToDevice));
            CK(hipMemset(d_ref, 0xff, size_t(Q) * gpu::LIST_W * 4));
            CK(hipMemset(d_got, 0xff, size_t(Q) * gpu::LIST_W * 4));
            gpu::qsa_select_T(d_sc, nb_max > K ? ld : 0, c.q0, Q, d_ref, d_refn, nullptr);
            std::vector<gpu::IdxCand> all(size_t(Q) * cfg::RANKS * K), cand(size_t(Q) * K);
            for (int r = 0; r < cfg::RANKS; ++r) {
                const int c0 = std::min(nb_max, r * per), n = std::min(per, nb_max - c0);
                std::vector<float> loc(size_t(Q) * per, -1.f);
                for (int q = 0; q < Q; ++q)
                    for (int i = 0; i < n; ++i) loc[size_t(q) * per + i] = sc[size_t(q) * ld + c0 + i];
                CK(hipMemcpy(d_loc, loc.data(), loc.size() * 4, hipMemcpyHostToDevice));
                gpu::qsa_select_local_T(d_loc, per, c0, n, c.q0, Q, d_cand, nullptr);
                CK(hipDeviceSynchronize());
                CK(hipMemcpy(cand.data(), d_cand, cand.size() * sizeof(gpu::IdxCand), hipMemcpyDeviceToHost));
                for (int q = 0; q < Q; ++q)
                    std::copy_n(cand.begin() + ptrdiff_t(q) * K, K, all.begin() + (ptrdiff_t(q) * cfg::RANKS + r) * K);
            }
            CK(hipMemcpy(d_all, all.data(), all.size() * sizeof(gpu::IdxCand), hipMemcpyHostToDevice));
            gpu::qsa_select_merge_T(d_all, c.q0, Q, d_got, d_gotn, nullptr);
            CK(hipDeviceSynchronize());
            std::vector<int32_t> ref(size_t(Q) * gpu::LIST_W), got(ref.size()), refn(ref.size() / gpu::LIST_W), gotn(refn.size());
            CK(hipMemcpy(ref.data(), d_ref, ref.size() * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(got.data(), d_got, got.size() * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(refn.data(), d_refn, refn.size() * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(gotn.data(), d_gotn, gotn.size() * 4, hipMemcpyDeviceToHost));
            int bad = 0;
            for (int q = 0; q < Q; ++q) {
                bool same = refn[size_t(q)] == gotn[size_t(q)];
                for (int i = 0; same && i < refn[size_t(q)]; ++i)
                    same = ref[size_t(q) * gpu::LIST_W + i] == got[size_t(q) * gpu::LIST_W + i];
                bad += !same;
            }
            std::printf("q0 %7lld, %2d score levels, %6d groups: %s (%d of %d queries differ)\n", (long long)c.q0,
                        c.levels, nb_max, bad ? "FAIL" : "ok", bad, Q);
            fails += bad;
            for (void *p : {(void *)d_sc, (void *)d_loc, (void *)d_ref, (void *)d_got, (void *)d_refn, (void *)d_gotn,
                            (void *)d_cand, (void *)d_all})
                CK(hipFree(p));
        }
        std::printf(fails ? "FAILED\n" : "PASS\n");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
