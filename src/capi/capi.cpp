#include "qw/capi.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>

#include "core/common.hpp"
#include "core/json.hpp"
#include "engine/engine.hpp"
#include "session/session.hpp"

struct qw_handle {
    std::unique_ptr<qw::Engine> engine;
    std::unique_ptr<qw::Session> session;
    std::string error;
};

namespace {

template <typename F>
auto guarded(qw_handle *h, F &&f, decltype(f()) on_error) -> decltype(f()) {
    try {
        h->error.clear();
        return f();
    } catch (const std::exception &e) {
        h->error = e.what();
        return on_error;
    }
}

qw::SamplingParams params(const qw_sampling *p) {
    qw::SamplingParams s;
    s.temperature = p->temperature;
    s.top_p = p->top_p;
    s.top_k = p->top_k;
    s.min_p = p->min_p;
    s.presence_penalty = p->presence_penalty;
    s.frequency_penalty = p->frequency_penalty;
    s.repetition_penalty = p->repetition_penalty;
    s.seed = p->seed;
    return s;
}

}  // namespace

extern "C" {

qw_handle *qw_open(const char *options_json, char *err, int errlen) {
    auto h = std::make_unique<qw_handle>();
    try {
        qw::EngineOptions opt;
        if (options_json && *options_json) {
            qw::Json j = qw::Json::parse(options_json);
            if (auto *v = j.find("model_dir")) opt.model_dir = v->as_str();
            if (auto *v = j.find("ple_dir")) opt.ple_dir = v->as_str();
            if (auto *v = j.find("slots")) {
                opt.slot_tokens.clear();
                for (size_t i = 0; i < v->size(); ++i) opt.slot_tokens.push_back(int((*v)[i].as_int()));
            }
            if (auto *v = j.find("slot_spill")) {  // tokens beyond each slot's VRAM, in pinned host memory
                opt.slot_spill.clear();
                for (size_t i = 0; i < v->size(); ++i) opt.slot_spill.push_back(int((*v)[i].as_int()));
            }
            if (auto *v = j.find("slot_max_tokens")) opt.slot_max_tokens = int(v->as_int());  // spill up to this capacity
            if (auto *v = j.find("weight_cache_dir")) opt.weight_cache_dir = v->as_str();
            if (auto *v = j.find("prefill_chunk")) opt.prefill_chunk = int(v->as_int());
            if (auto *v = j.find("warmup")) opt.warmup = v->as_bool();
            if (auto *v = j.find("devices")) {
                QW_CHECK(v->size() == qw::RANKS, "devices must list 4 GPUs");
                for (size_t i = 0; i < qw::RANKS; ++i) opt.devices[i] = int((*v)[i].as_int());
            }
        }
        h->engine = std::make_unique<qw::Engine>(opt);
        h->session = std::make_unique<qw::Session>(*h->engine);
        return h.release();
    } catch (const std::exception &e) {
        if (err && errlen > 0) {
            std::strncpy(err, e.what(), size_t(errlen - 1));
            err[errlen - 1] = 0;
        }
        return nullptr;
    }
}

void qw_close(qw_handle *h) {
    delete h;
}
const char *qw_error(qw_handle *h) {
    return h->error.c_str();
}
int qw_engine_failure(qw_handle *h, char *buf, int buflen) {
    const std::string why = h->engine->failure();
    if (!why.empty() && buf && buflen > 0) {
        std::strncpy(buf, why.c_str(), size_t(buflen - 1));
        buf[buflen - 1] = 0;
    }
    return int(why.size());
}
int qw_num_slots(qw_handle *h) {
    return h->engine->num_slots();
}
int64_t qw_slot_capacity(qw_handle *h, int slot) {
    return guarded(h, [&] { return h->engine->slot_capacity(slot); }, int64_t(-1));
}

int qw_acquire(qw_handle *h, const int32_t *tokens, int64_t n, int64_t max_new) {
    return guarded(h, [&] { return h->session->acquire(std::vector<int32_t>(tokens, tokens + n), max_new); }, -1);
}

int qw_release(qw_handle *h, int slot) {
    return guarded(
        h,
        [&] {
            h->session->release(slot);
            return 0;
        },
        -1);
}

int64_t qw_set_prompt(qw_handle *h, int slot, const int32_t *tokens, int64_t n) {
    return guarded(
        h, [&] { return h->session->set_prompt(slot, std::vector<int32_t>(tokens, tokens + n)); }, int64_t(-1));
}

int qw_has_vision(qw_handle *h) {
    return h->engine->has_vision() ? 1 : 0;
}

static std::vector<qw::Session::Media> media_of(const qw_media *media, int n_media) {
    std::vector<qw::Session::Media> ms;
    for (int i = 0; i < n_media; ++i) {
        const qw_media &m = media[i];
        ms.push_back({m.hash, m.video != 0, m.t, m.h, m.w, m.patches, std::vector<int64_t>(m.starts, m.starts + m.t)});
    }
    return ms;
}

int64_t qw_set_prompt_media(qw_handle *h, int slot, const int32_t *tokens, int64_t n, const qw_media *media,
                            int n_media) {
    return guarded(
        h,
        [&] {
            return h->session->set_prompt(slot, std::vector<int32_t>(tokens, tokens + n), media_of(media, n_media));
        },
        int64_t(-1));
}

int64_t qw_begin_prompt(qw_handle *h, int slot, const int32_t *tokens, int64_t n, const qw_media *media, int n_media) {
    return guarded(
        h,
        [&] {
            return h->session->begin_prompt(slot, std::vector<int32_t>(tokens, tokens + n), media_of(media, n_media));
        },
        int64_t(-1));
}

int qw_prefetch(qw_handle *h, int slot, const int32_t *tokens, int64_t n, const qw_media *media, int n_media) {
    return guarded(
        h,
        [&] {
            return h->session->prefetch(slot, std::vector<int32_t>(tokens, tokens + n), media_of(media, n_media)) ? 1
                                                                                                                 : 0;
        },
        -1);
}

int qw_prefill_some(qw_handle *h, int slot, int64_t max_tokens) {
    return guarded(h, [&] { return h->session->prefill_some(slot, max_tokens) ? 1 : 0; }, -1);
}

int qw_prefill_batch(qw_handle *h, int n, const int32_t *slots, const int64_t *max_tokens, int32_t *done) {
    return guarded(
        h,
        [&] {
            std::vector<std::pair<int, int64_t>> reqs;
            for (int i = 0; i < n; ++i) reqs.push_back({slots[i], max_tokens[i]});
            const auto d = h->session->prefill_batch(reqs);
            for (int i = 0; i < n; ++i) done[i] = d[size_t(i)] ? 1 : 0;
            return 0;
        },
        -1);
}

int qw_acquire_media(qw_handle *h, const int32_t *tokens, int64_t n, int64_t max_new, const qw_media *media,
                     int n_media) {
    return guarded(
        h,
        [&] {
            return h->session->acquire(std::vector<int32_t>(tokens, tokens + n), max_new, media_of(media, n_media));
        },
        -1);
}

int32_t qw_sample_prompt(qw_handle *h, int slot, const qw_sampling *p, float *lp) {
    return guarded(h, [&] { return h->session->sample_prompt(slot, params(p), lp); }, int32_t(-1));
}

int qw_decode(qw_handle *h, int n, const int32_t *slots, const int32_t *tokens) {
    return guarded(
        h,
        [&] {
            std::vector<qw::Engine::Row> rows;
            for (int i = 0; i < n; ++i) rows.push_back({slots[i], tokens[i]});
            h->session->decode(rows);
            return 0;
        },
        -1);
}

int32_t qw_sample_row(qw_handle *h, int row, const qw_sampling *p, float *lp) {
    return guarded(h, [&] { return h->session->sample_row(row, params(p), lp); }, int32_t(-1));
}

int qw_persist(qw_handle *h) {
    return guarded(
        h,
        [&] {
            h->session->persist();
            return 0;
        },
        -1);
}

int qw_set_boundary_token(qw_handle *h, int32_t id) {
    h->session->set_boundary_token(id);
    return 0;
}

int qw_get_cache_stats(qw_handle *h, qw_cache_stats *out) {
    const qw::BlockStore::Stats s = h->session->cache_stats();
    const qw::Session::ReuseCounters c = h->session->reuse_counters();
    *out = {s.hits,      s.tokens_restored, s.snapshots_saved, s.ram_bytes,        s.disk_bytes,
            s.blocks,    s.snapshots,       c.prompt_tokens,   c.reused_tokens,    c.blend_candidate_tokens};
    return 0;
}

int qw_get_slot_timing(qw_handle *h, int slot, qw_slot_timing *out) {
    return guarded(
        h,
        [&] {
            if (slot < 0 || slot >= h->session->num_slots()) return -1;
            const qw::Session::Timing t = h->session->slot_timing(slot);
            *out = {t.restore_s, t.prefill_s, t.save_s, t.pin_wait_s};
            return 0;
        },
        -1);
}

int qw_has_mtp(qw_handle *h) {
    return h->engine->has_mtp() ? 1 : 0;
}

int qw_set_stop_tokens(qw_handle *h, int slot, const int32_t *ids, int n) {
    return guarded(
        h,
        [&] {
            h->session->set_stop_tokens(slot, std::vector<int32_t>(ids, ids + n));
            return 0;
        },
        -1);
}

int qw_generate(qw_handle *h, int n, const qw_step_req *reqs, int k, int32_t *tokens, float *logprobs, int32_t *counts,
                int32_t *first_rows, int32_t *stopped) {
    return guarded(
        h,
        [&] {
            std::vector<qw::Session::StepReq> rq;
            for (int i = 0; i < n; ++i)
                rq.push_back({reqs[i].slot, reqs[i].pending, reqs[i].budget, params(&reqs[i].sampling),
                              (reqs[i].flags & QW_STEP_LOGITS) != 0});
            const auto &out = h->session->generate(rq, k);
            const size_t w = size_t(std::max(k, 0)) + 1;
            for (int i = 0; i < n; ++i) {
                const auto &o = out[size_t(i)];
                QW_CHECK(o.tokens.size() <= w, "generate: more tokens than k + 1");
                std::copy(o.tokens.begin(), o.tokens.end(), tokens + size_t(i) * w);
                std::copy(o.logprobs.begin(), o.logprobs.end(), logprobs + size_t(i) * w);
                counts[i] = int32_t(o.tokens.size());
                first_rows[i] = o.first_row;
                stopped[i] = o.stopped ? 1 : 0;
            }
            return 0;
        },
        -1);
}

int qw_top_logprobs(qw_handle *h, int row, int k, int32_t *ids, float *lps) {
    return guarded(
        h,
        [&] {
            std::vector<int32_t> i;
            std::vector<float> l;
            if (row < 0)
                h->session->top_logprobs_prompt(k, i, l);
            else
                h->session->top_logprobs_row(row, k, i, l);
            std::memcpy(ids, i.data(), i.size() * 4);
            std::memcpy(lps, l.data(), l.size() * 4);
            return int(i.size());
        },
        -1);
}

int qw_top_logprobs_prompt(qw_handle *h, int slot, int k, int32_t *ids, float *lps) {
    return guarded(
        h,
        [&] {
            std::vector<int32_t> i;
            std::vector<float> l;
            h->session->top_logprobs_prompt(slot, k, i, l);
            std::memcpy(ids, i.data(), i.size() * 4);
            std::memcpy(lps, l.data(), l.size() * 4);
            return int(i.size());
        },
        -1);
}

}  // extern "C"
