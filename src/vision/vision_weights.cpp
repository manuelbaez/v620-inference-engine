#include "vision/vision_weights.hpp"

#include <string>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/weight_convert.hpp"

namespace qw {

namespace {

std::vector<uint16_t> f16(const SafeTensors &st, const std::string &name, std::vector<int64_t> shape) {
    const TensorView &t = st.get(name, DType::BF16, shape);
    std::vector<uint16_t> out(size_t(t.numel()));
    for (size_t i = 0; i < out.size(); ++i) out[i] = bf16_to_f16(t.u16()[i]);
    return out;
}

std::vector<float> f32(const SafeTensors &st, const std::string &name, std::vector<int64_t> shape) {
    return f32_of(st.get(name, DType::BF16, shape));
}

template <typename T>
T *dev(const std::vector<T> &v, size_t &bytes) {
    bytes += v.size() * sizeof(T);
    return upload(v);
}

}  // namespace

std::unique_ptr<VisionHostWeights> load_vision_weights(const SafeTensors &st) {
    const std::string p = "model.visual.";
    if (!st.has(p + "patch_embed.proj.weight")) return nullptr;
    auto hw = std::make_unique<VisionHostWeights>();
    const VisionConfig &c = hw->cfg;
    const int64_t D = c.dim, M = c.mlp, DM = int64_t(D) * c.merge * c.merge;
    hw->patch_w = f16(st, p + "patch_embed.proj.weight", {D, 3, 2, 16, 16});
    hw->patch_b = f32(st, p + "patch_embed.proj.bias", {D});
    hw->pos_table = f32(st, p + "pos_embed.weight", {int64_t(c.grid_side) * c.grid_side, D});
    for (int b = 0; b < c.depth; ++b) {
        const std::string q = p + "blocks." + std::to_string(b) + ".";
        VisionHostWeights::Block blk;
        blk.n1w = f32(st, q + "norm1.weight", {D});
        blk.n1b = f32(st, q + "norm1.bias", {D});
        blk.n2w = f32(st, q + "norm2.weight", {D});
        blk.n2b = f32(st, q + "norm2.bias", {D});
        blk.qkv = f16(st, q + "attn.qkv.weight", {3 * D, D});
        blk.qkv_b = f32(st, q + "attn.qkv.bias", {3 * D});
        blk.proj = f16(st, q + "attn.proj.weight", {D, D});
        blk.proj_b = f32(st, q + "attn.proj.bias", {D});
        blk.fc1 = f16(st, q + "mlp.linear_fc1.weight", {M, D});
        blk.fc1_b = f32(st, q + "mlp.linear_fc1.bias", {M});
        blk.fc2 = f16(st, q + "mlp.linear_fc2.weight", {D, M});
        blk.fc2_b = f32(st, q + "mlp.linear_fc2.bias", {D});
        hw->blocks.push_back(std::move(blk));
    }
    hw->m_nw = f32(st, p + "merger.norm.weight", {D});
    hw->m_nb = f32(st, p + "merger.norm.bias", {D});
    hw->m_fc1 = f16(st, p + "merger.linear_fc1.weight", {DM, DM});
    hw->m_fc1_b = f32(st, p + "merger.linear_fc1.bias", {DM});
    hw->m_fc2 = f16(st, p + "merger.linear_fc2.weight", {int64_t(c.out_dim), DM});
    hw->m_fc2_b = f32(st, p + "merger.linear_fc2.bias", {int64_t(c.out_dim)});
    return hw;
}

VisionDeviceWeights upload_vision_weights(const VisionHostWeights &hw) {
    VisionDeviceWeights d;
    size_t &n = d.bytes;
    d.patch_w = dev(hw.patch_w, n);
    d.patch_b = dev(hw.patch_b, n);
    d.pos_table = dev(hw.pos_table, n);
    for (const auto &b : hw.blocks)
        d.blocks.push_back({dev(b.n1w, n), dev(b.n1b, n), dev(b.n2w, n), dev(b.n2b, n), dev(b.qkv_b, n),
                            dev(b.proj_b, n), dev(b.fc1_b, n), dev(b.fc2_b, n), dev(b.qkv, n), dev(b.proj, n),
                            dev(b.fc1, n), dev(b.fc2, n)});
    d.m_nw = dev(hw.m_nw, n);
    d.m_nb = dev(hw.m_nb, n);
    d.m_fc1 = dev(hw.m_fc1, n);
    d.m_fc1_b = dev(hw.m_fc1_b, n);
    d.m_fc2 = dev(hw.m_fc2, n);
    d.m_fc2_b = dev(hw.m_fc2_b, n);
    return d;
}

}  // namespace qw
