#include "core/config.hpp"

#include "core/common.hpp"
#include "core/json.hpp"
#include "core/safetensors.hpp"

namespace qw::cfg {

void check_config(const std::string &model_dir) {
    Json root = Json::parse(read_file(model_dir + "/config.json"));
    const Json &t = root["text_config"];
    auto want = [&](const char *key, int64_t v) {
        int64_t got = t[key].as_int();
        if (got != v)
            fail(std::string("config.json: ") + key + " = " + std::to_string(got) +
                 ", engine is built for " + std::to_string(v));
    };
    want("hidden_size", H);
    want("hc_count", HC);
    want("hc_lowrank", HC_RANK);
    want("num_hidden_layers", N_LAYERS);
    want("vocab_size", VOCAB);
    want("eos_token_id", EOS);
    want("num_attention_heads", Q_HEADS);
    want("num_key_value_heads", KV_HEADS);
    want("head_dim", HEAD_DIM);
    want("indexer_n_heads", IDX_HEADS);
    want("indexer_head_dim", IDX_DIM);
    want("indexer_kv_heads", 1);
    want("indexer_budget", IDX_BUDGET);
    want("indexer_compress_ratio", IDX_RATIO);
    want("linear_num_key_heads", GDN_QK_HEADS);
    want("linear_num_value_heads", GDN_V_HEADS);
    want("linear_key_head_dim", GDN_DIM);
    want("linear_value_head_dim", GDN_DIM);
    want("linear_conv_kernel_dim", GDN_CONV);
    want("num_experts", N_EXPERTS);
    want("num_experts_per_tok", TOP_K);
    want("moe_intermediate_size", FFN);
    want("shared_expert_intermediate_size", FFN);
    want("ngram_size", NGRAM);
    want("heads_per_ngram", NGRAM_HEADS / (NGRAM - 1));
    want("ple_embed_dim", H);
    want("ple_conv_kernel_size", PLE_CONV);
    want("ngram_vocab_size_base", NGRAM_VOCAB_BASE);
    want("seed", int64_t(PLE_SEED));
    QW_CHECK(t["ple_layer_ids"].size() == 1 && t["ple_layer_ids"][0].as_int() == PLE_LAYER + 1,
             "config.json: unexpected ple_layer_ids");
    QW_CHECK(t["output_gate_type"].as_str() == "sigmoid", "config.json: GDN gate is not sigmoid");
    QW_CHECK(t["norm_topk_prob"].as_bool(), "config.json: norm_topk_prob is false");
    QW_CHECK(t["full_attention_interval"].as_int() == 4, "config.json: attention interval");
    double theta = t["rope_parameters"]["rope_theta"].as_double();
    QW_CHECK(theta == ROPE_THETA, "config.json: rope_theta");
    double prf = t["partial_rotary_factor"].as_double();
    QW_CHECK(int(prf * HEAD_DIM + 0.5) == ROPE_DIM, "config.json: partial_rotary_factor");
    const Json &layer_types = t["layer_types"];
    for (int i = 0; i < N_LAYERS; ++i) {
        bool full = layer_types[size_t(i)].as_str() == "full_attention";
        QW_CHECK(full == is_qsa(i), "config.json: layer_types pattern differs");
    }
    const Json &q = root["quantization_config"]["config_groups"]["group_0"]["weights"];
    QW_CHECK(q["num_bits"].as_int() == 4 && q["group_size"].as_int() == QGROUP &&
                 q["symmetric"].as_bool(),
             "config.json: experts are not symmetric int4 g128");
}

}  // namespace qw::cfg
