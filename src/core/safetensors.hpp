// Read-only mmap access to a set of safetensors files, looked up by tensor name.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace qw {

enum class DType { F32, F16, BF16, I32, I64, U8, I8 };

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
    // Opens one file and indexes its tensors.
    void add_file(const std::string &path);
    // Opens every *.safetensors file named in <dir>/model.safetensors.index.json,
    // plus any extra files given.
    void add_index(const std::string &dir, const std::vector<std::string> &extra = {});

    const std::vector<std::unique_ptr<MappedFile>> &files() const { return files_; }
    bool has(const std::string &name) const { return tensors_.count(name) != 0; }
    // Unmaps every file (views into them become invalid).
    void clear() {
        tensors_.clear();
        files_.clear();
    }
    const TensorView &get(const std::string &name) const;
    // get() plus a dtype and shape check.
    const TensorView &get(const std::string &name, DType dt, std::vector<int64_t> shape) const;

    const std::unordered_map<std::string, TensorView> &all() const { return tensors_; }

private:
    std::vector<std::unique_ptr<MappedFile>> files_;
    std::unordered_map<std::string, TensorView> tensors_;
};

std::string read_file(const std::string &path);

}  // namespace qw
