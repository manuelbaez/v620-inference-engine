// Read-only mmap access to a set of safetensors files, looked up by tensor name.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace qw {

enum class DType { F32, F16, BF16, I32, I64, U8, I8, F8E4M3, U32 };

size_t dtype_size(DType t);
const char *dtype_name(DType t);

struct TensorView {
    std::string name;
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    const void *data = nullptr;
    size_t nbytes = 0;

    int64_t numel() const;
    int64_t dim(int i) const;  // negative counts from the end
    const uint16_t *u16() const { return static_cast<const uint16_t *>(data); }
    const int32_t *i32() const { return static_cast<const int32_t *>(data); }
    const uint8_t *u8() const { return static_cast<const uint8_t *>(data); }
    const float *f32() const { return static_cast<const float *>(data); }
};

class MappedFile {
public:
    explicit MappedFile(const std::string &path);
    ~MappedFile();
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;

    // Pins the mapping in RAM (reads it in); false if the kernel refuses.
    bool lock() const;
    // Reads every page once (warms the page cache without pinning).
    void touch() const;
    const uint8_t *data() const { return data_; }
    size_t size() const { return size_; }
    const std::string &path() const { return path_; }

    // Hint the kernel to read the whole file ahead (MADV_WILLNEED).
    void prefetch() const;

private:
    std::string path_;
    uint8_t *data_ = nullptr;
    size_t size_ = 0;
};

class SafeTensors {
public:
    // Opens one file and indexes its tensors. With `replace`, its tensors take the place of those already indexed
    // under the same name (an overlay for experiments); otherwise a repeated name is an error.
    // With `only`, just the tensors whose names start with it.
    void add_file(const std::string &path, bool replace = false, const std::string &only = "");
    // Opens every *.safetensors file named in <dir>/model.safetensors.index.json,
    // plus any extra files given.
    void add_index(const std::string &dir, const std::vector<std::string> &extra = {});
    // Every tensor whose name starts with `prefix` from the checkpoint in `dir` instead of this one's (whose tensors of
    // that prefix are dropped first): e.g. another checkpoint's MTP head (QW_MTP_FROM).
    void overlay(const std::string &dir, const std::string &prefix);

    const std::vector<std::unique_ptr<MappedFile>> &files() const { return files_; }
    bool has(const std::string &name) const { return tensors_.count(name) != 0 || synthesized(name); }
    // Unmaps every file (views into them become invalid) and frees the decoded tensors.
    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        decoded_.clear();
        tensors_.clear();
        files_.clear();
    }
    // A ModelOpt checkpoint's tensors the loaders read as bf16 come back decoded (core/fp_decode.hpp), on first use,
    // kept until clear(): MXFP8 and NVFP4 matrices other than the routed experts (whose loaders read NVFP4 as it is),
    // F32 weights (norms; rounded to bf16), and the MTP head's fused experts (mtp.layers.0.mlp.experts.gate_up_proj and
    // down_proj, from its NVFP4 experts). Other checkpoints have none of these: their tensors are returned as stored.
    const TensorView &get(const std::string &name) const;
    // get() plus a dtype and shape check.
    const TensorView &get(const std::string &name, DType dt, std::vector<int64_t> shape) const;

    const std::unordered_map<std::string, TensorView> &all() const { return tensors_; }

private:
    struct Decoded {
        std::vector<uint16_t> data;
        TensorView view;
    };
    bool synthesized(const std::string &name) const;  // the MTP fused experts of a ModelOpt checkpoint
    bool needs_decoding(const TensorView &t) const;
    void decode(const std::string &name, Decoded &d) const;

    std::vector<std::unique_ptr<MappedFile>> files_;
    std::unordered_map<std::string, TensorView> tensors_;
    mutable std::mutex mu_;
    mutable std::unordered_map<std::string, std::unique_ptr<Decoded>> decoded_;
};

std::string read_file(const std::string &path);

}  // namespace qw
