// Weights of the vision tower (model.visual.*, a Qwen3-VL ViT), converted on
// the host once and uploaded to every card (each card encodes whole images).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/safetensors.hpp"

namespace qw {

struct VisionConfig {
    int depth = 27, dim = 1152, heads = 16, mlp = 4304, out_dim = 2560;
    int patch_in = 3 * 2 * 16 * 16;  // channels x temporal x patch x patch
    int merge = 2, grid_side = 48;   // learned position table: grid_side^2 entries
};

// Host copies: matrices fp16 [out][in], vectors fp32.
struct VisionHostWeights {
    VisionConfig cfg;
    std::vector<uint16_t> patch_w;
    std::vector<float> patch_b, pos_table;
    struct Block {
        std::vector<float> n1w, n1b, n2w, n2b, qkv_b, proj_b, fc1_b, fc2_b;
        std::vector<uint16_t> qkv, proj, fc1, fc2;
    };
    std::vector<Block> blocks;
    std::vector<float> m_nw, m_nb, m_fc1_b, m_fc2_b;
    std::vector<uint16_t> m_fc1, m_fc2;
};

// Null when the checkpoint has no vision tower.
std::unique_ptr<VisionHostWeights> load_vision_weights(const SafeTensors &st);

// Device copies on one card (allocated on the current device).
struct VisionDeviceWeights {
    uint16_t *patch_w = nullptr;
    float *patch_b = nullptr, *pos_table = nullptr;
    struct Block {
        float *n1w, *n1b, *n2w, *n2b, *qkv_b, *proj_b, *fc1_b, *fc2_b;
        uint16_t *qkv, *proj, *fc1, *fc2;
    };
    std::vector<Block> blocks;
    float *m_nw = nullptr, *m_nb = nullptr, *m_fc1_b = nullptr, *m_fc2_b = nullptr;
    uint16_t *m_fc1 = nullptr, *m_fc2 = nullptr;
    size_t bytes = 0;
};
VisionDeviceWeights upload_vision_weights(const VisionHostWeights &hw);

}  // namespace qw
