#include "engine/capi.hpp"

#include <cstring>
#include <memory>
#include <string>

#include "core/common.hpp"
#include "core/json.hpp"
#include "engine/engine.hpp"
#include "engine/session.hpp"

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

void qw_close(qw_handle *h) { delete h; }
const char *qw_error(qw_handle *h) { return h->error.c_str(); }
int qw_num_slots(qw_handle *h) { return h->engine->num_slots(); }
int64_t qw_slot_capacity(qw_handle *h, int slot) {
    return guarded(h, [&] { return h->engine->slot_capacity(slot); }, int64_t(-1));
}

int qw_acquire(qw_handle *h, const int32_t *tokens, int64_t n, int64_t max_new) {
    return guarded(h, [&] { return h->session->acquire(std::vector<int32_t>(tokens, tokens + n), max_new); }, -1);
}

int qw_release(qw_handle *h, int slot) {
    return guarded(h, [&] {
        h->session->release(slot);
        return 0;
    }, -1);
}

int64_t qw_set_prompt(qw_handle *h, int slot, const int32_t *tokens, int64_t n) {
    return guarded(h, [&] { return h->session->set_prompt(slot, std::vector<int32_t>(tokens, tokens + n)); },
                   int64_t(-1));
}

int32_t qw_sample_prompt(qw_handle *h, int slot, const qw_sampling *p, float *lp) {
    return guarded(h, [&] { return h->session->sample_prompt(slot, params(p), lp); }, int32_t(-1));
}

int qw_decode(qw_handle *h, int n, const int32_t *slots, const int32_t *tokens) {
    return guarded(h, [&] {
        std::vector<qw::Engine::Row> rows;
        for (int i = 0; i < n; ++i) rows.push_back({slots[i], tokens[i]});
        h->session->decode(rows);
        return 0;
    }, -1);
}

int32_t qw_sample_row(qw_handle *h, int row, const qw_sampling *p, float *lp) {
    return guarded(h, [&] { return h->session->sample_row(row, params(p), lp); }, int32_t(-1));
}

int qw_top_logprobs(qw_handle *h, int row, int k, int32_t *ids, float *lps) {
    return guarded(h, [&] {
        std::vector<int32_t> i;
        std::vector<float> l;
        if (row < 0) h->session->top_logprobs_prompt(k, i, l);
        else h->session->top_logprobs_row(row, k, i, l);
        std::memcpy(ids, i.data(), i.size() * 4);
        std::memcpy(lps, l.data(), l.size() * 4);
        return int(i.size());
    }, -1);
}

}  // extern "C"
