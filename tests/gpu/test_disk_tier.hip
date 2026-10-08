// Disk tier alone (no GPU needed at run time): written files read back
// identical, the queue's byte count returns to zero once the writes are on
// disk, scan() indexes the files, and reads from several threads at once (the
// block store's load workers) are safe and correct, and a read into buffers that
// ranks share (a store keeping one KV copy per replica group) skips the
// replicas' bytes in the file; a block written from such buffers is a v2 file holding one buffer per group (and a
// scan and read of it work, also into one buffer per rank).
#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "session/disk_tier.hpp"

using namespace qw;

constexpr int FILES = 24;
constexpr size_t RANK_BYTES = 3u << 20;

static uint8_t pattern(int file, size_t i) {
    return uint8_t((size_t(file) * 131 + i * 2654435761u) >> 7);
}

int main() {
    try {
        const std::string dir = "/tmp/qw_disk_tier_unit";
        std::system(("rm -rf " + dir).c_str());
        int fails = 0;
        auto check = [&](bool ok, const char *what) {
            if (!ok) std::printf("FAIL: %s\n", what);
            fails += !ok;
        };
        DiskTier d(dir, 42);

        size_t total = 0;
        for (int i = 0; i < FILES; ++i) {
            auto buf = std::make_shared<std::vector<uint8_t>>(size_t(RANKS) * RANK_BYTES);
            for (size_t j = 0; j < buf->size(); ++j) (*buf)[j] = pattern(i, j);
            Engine::RankBufs bufs;
            for (int r = 0; r < RANKS; ++r) bufs[size_t(r)] = buf->data() + size_t(r) * RANK_BYTES;
            DiskTier::Meta m;
            m.hash = 1000 + uint64_t(i);
            m.parent = 7;
            m.tokens.assign(256, i);
            const size_t b = d.write(m, {}, bufs, RANK_BYTES, buf);
            check(b > size_t(RANKS) * RANK_BYTES + 256 * 4, "write returns the file size (payload, tokens and header)");
            total += b;
        }
        const size_t queued = d.pending_bytes();
        check(queued <= total, "pending bytes never exceed what was queued");
        d.flush();
        check(d.pending_bytes() == 0, "pending bytes are zero after flush");
        std::printf("queued %zu MB of %zu MB when flush was called\n", queued >> 20, total >> 20);

        const auto metas = d.scan();
        check(int(metas.size()) == FILES, "scan finds every file");
        for (const auto &m : metas) check(m.tokens.size() == 256 && !m.snap && m.parent == 7, "scan metadata");

        // 4 threads read interleaved files at once
        std::vector<std::thread> th;
        std::vector<int> bad(4, 0);
        const auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < 4; ++t)
            th.emplace_back([&, t] {
                std::vector<uint8_t> out(size_t(RANKS) * RANK_BYTES);
                for (int i = t; i < FILES; i += 4) {
                    Engine::RankBufs bufs;
                    for (int r = 0; r < RANKS; ++r) bufs[size_t(r)] = out.data() + size_t(r) * RANK_BYTES;
                    if (!d.read(1000 + uint64_t(i), false, bufs, RANK_BYTES, nullptr)) {
                        ++bad[size_t(t)];
                        continue;
                    }
                    for (size_t j = 0; j < out.size(); ++j)
                        if (out[j] != pattern(i, j)) {
                            ++bad[size_t(t)];
                            break;
                        }
                }
            });
        for (auto &t : th) t.join();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        for (int t = 0; t < 4; ++t) check(bad[size_t(t)] == 0, "parallel reads return the written bytes");
        std::printf("4 threads read %zu MB in %.3f s\n", total >> 20, s);

        // a read waits for its own queued write
        {
            auto buf = std::make_shared<std::vector<uint8_t>>(size_t(RANKS) * RANK_BYTES, 0x5a);
            Engine::RankBufs bufs;
            for (int r = 0; r < RANKS; ++r) bufs[size_t(r)] = buf->data() + size_t(r) * RANK_BYTES;
            DiskTier::Meta m;
            m.hash = 5000;
            m.parent = 7;
            m.tokens.assign(256, 1);
            d.write(m, {}, bufs, RANK_BYTES, buf);
            std::vector<uint8_t> out(size_t(RANKS) * RANK_BYTES);
            Engine::RankBufs ob;
            for (int r = 0; r < RANKS; ++r) ob[size_t(r)] = out.data() + size_t(r) * RANK_BYTES;
            check(d.read(5000, false, ob, RANK_BYTES, nullptr) && out[17] == 0x5a && out.back() == 0x5a,
                  "a read right after its write sees the whole file");
        }
        // buffers shared by the ranks of a replica group: the file holds four buffers, the read fills the two
        // distinct ones and skips the replicas' bytes (here made different on purpose, to show they are skipped)
        {
            auto buf = std::make_shared<std::vector<uint8_t>>(size_t(RANKS) * RANK_BYTES);
            for (int r = 0; r < RANKS; ++r)
                for (size_t j = 0; j < RANK_BYTES; ++j) (*buf)[size_t(r) * RANK_BYTES + j] = uint8_t(pattern(60 + r, j));
            Engine::RankBufs bufs;
            for (int r = 0; r < RANKS; ++r) bufs[size_t(r)] = buf->data() + size_t(r) * RANK_BYTES;
            DiskTier::Meta m;
            m.hash = 6000;
            m.parent = 7;
            m.tokens.assign(256, 2);
            d.write(m, {}, bufs, RANK_BYTES, buf);
            std::vector<uint8_t> a(RANK_BYTES, 0), b(RANK_BYTES, 0);
            Engine::RankBufs shared;
            shared[0] = shared[1] = a.data();
            shared[2] = shared[3] = b.data();
            check(d.read(6000, false, shared, RANK_BYTES, nullptr), "a read into shared buffers succeeds");
            bool first = true, second = true;
            for (size_t j = 0; j < RANK_BYTES; ++j) {
                first = first && a[j] == pattern(60, j);   // rank 0's bytes, not rank 1's
                second = second && b[j] == pattern(62, j);  // rank 2's bytes, found past the skipped rank 1
            }
            check(first, "the shared buffer holds the first rank's bytes (the replica's were skipped)");
            check(second, "the next group's buffer holds its own bytes (the skip lands on the right offset)");
        }
        // v2 blocks: buffers shared by the ranks of a group are written once per group
        {
            const size_t RB = RANK_BYTES;
            std::vector<uint8_t> a(RB), b(RB);
            for (size_t j = 0; j < RB; ++j) a[j] = pattern(70, j), b[j] = pattern(72, j);
            auto keep = std::make_shared<int>(0);
            Engine::RankBufs grouped_bufs;
            grouped_bufs[0] = grouped_bufs[1] = a.data();
            grouped_bufs[2] = grouped_bufs[3] = b.data();
            DiskTier::Meta m;
            m.hash = 7000;
            m.parent = 7;
            m.tokens.assign(256, 3);
            const size_t v2_bytes = d.write(m, {}, grouped_bufs, RB, keep);
            // the same block from four distinct buffers is a v1 file with a buffer per rank
            std::vector<uint8_t> four(size_t(RANKS) * RB, 1);
            Engine::RankBufs v1_bufs;
            for (int r = 0; r < RANKS; ++r) v1_bufs[size_t(r)] = four.data() + size_t(r) * RB;
            m.hash = 7001;
            const size_t v1_bytes = d.write(m, {}, v1_bufs, RB, keep);
            d.flush();
            check(v1_bytes - v2_bytes == size_t(RANKS - RANKS / cfg::KV_REPLICAS) * RB, "a grouped block is smaller by the replicas' buffers");
            struct stat st{};
            char name[64];
            std::snprintf(name, sizeof name, "%s/%016llx.qwb", dir.c_str(), 7000ull);
            check(stat(name, &st) == 0 && size_t(st.st_size) == v2_bytes, "the v2 file's size is the size write() returned");
            std::snprintf(name, sizeof name, "%s/%016llx.qwb", dir.c_str(), 7001ull);
            check(stat(name, &st) == 0 && size_t(st.st_size) == v1_bytes, "distinct buffers still write a v1 file");
            int found = 0;
            for (const auto &mm : d.scan()) found += (mm.hash == 7000 || mm.hash == 7001) && mm.tokens.size() == 256;
            check(found == 2, "scan indexes v2 and v1 blocks");

            // read into grouped buffers, and into a buffer per rank (the replicas get their group's bytes)
            std::vector<uint8_t> ra(RB, 0), rb(RB, 0);
            Engine::RankBufs into_grouped;
            into_grouped[0] = into_grouped[1] = ra.data();
            into_grouped[2] = into_grouped[3] = rb.data();
            check(d.read(7000, false, into_grouped, RB, nullptr), "a v2 block reads into grouped buffers");
            check(ra == a && rb == b, "grouped buffers hold their group's bytes");
            std::vector<uint8_t> out(size_t(RANKS) * RB, 0);
            Engine::RankBufs into_four;
            for (int r = 0; r < RANKS; ++r) into_four[size_t(r)] = out.data() + size_t(r) * RB;
            check(d.read(7000, false, into_four, RB, nullptr), "a v2 block reads into a buffer per rank");
            bool same = true;
            for (size_t j = 0; j < RB; ++j)
                same = same && out[j] == a[j] && out[RB + j] == a[j] && out[2 * RB + j] == b[j] && out[3 * RB + j] == b[j];
            check(same, "every rank gets its group's bytes, replicas included");
            check(d.read(7001, false, into_four, RB, nullptr) && out[0] == 1 && out[3 * RB + RB - 1] == 1, "a v1 block still reads");
        }
        {  // remove() never waits for the disk: queued writes are dropped, files go once the writer gets to them
            const auto exists = [&](uint64_t h) {
                struct stat st{};
                char name[64];
                std::snprintf(name, sizeof name, "/%016llx.qwb", (unsigned long long)h);
                return stat((dir + name).c_str(), &st) == 0;
            };
            const auto put = [&](uint64_t h, int fill) {
                auto buf = std::make_shared<std::vector<uint8_t>>(size_t(RANKS) * RANK_BYTES, uint8_t(fill));
                Engine::RankBufs bufs;
                for (int r = 0; r < RANKS; ++r) bufs[size_t(r)] = buf->data() + size_t(r) * RANK_BYTES;
                DiskTier::Meta m;
                m.hash = h;
                m.parent = 7;
                m.tokens.assign(256, 1);
                d.write(m, {}, bufs, RANK_BYTES, buf);
            };
            d.flush();
            check(exists(1000) && exists(1001), "the first files are on disk");
            for (int i = 0; i < 40; ++i) put(9000 + uint64_t(i), i);  // ~480 MB queued
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 20; i < 40; ++i) d.remove(9000 + uint64_t(i), false);  // most of them still queued
            d.remove(1000, false);                                                // long on disk
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("21 removals behind %zu MB of queued writes took %.2f ms\n", d.pending_bytes() >> 20, ms);
            check(ms < 100, "removals do not wait for the queued writes");
            put(9030, 77);  // written again after its removal
            d.flush();
            check(d.pending_bytes() == 0, "pending bytes are zero after the removals");
            bool gone = !exists(1000);
            for (int i = 20; i < 40; ++i) gone = gone && (i == 30 || !exists(9000 + uint64_t(i)));
            check(gone, "removed files are not on disk after flush");
            bool kept = exists(1001) && exists(9030);
            for (int i = 0; i < 20; ++i) kept = kept && exists(9000 + uint64_t(i));
            check(kept, "the other files, and the one written again, are");
            std::vector<uint8_t> back(size_t(RANKS) * RANK_BYTES);
            Engine::RankBufs rb;
            for (int r = 0; r < RANKS; ++r) rb[size_t(r)] = back.data() + size_t(r) * RANK_BYTES;
            check(d.read(9030, false, rb, RANK_BYTES, nullptr) && back[0] == 77 && back.back() == 77,
                  "the file written again holds the new bytes");
        }
        d.flush();
        check(d.pending_bytes() == 0, "pending bytes are zero at the end");
        std::system(("rm -rf " + dir).c_str());
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
