#include "core/safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <set>
#include <sstream>

#include "core/common.hpp"
#include "core/json.hpp"

namespace qw {

size_t dtype_size(DType t) {
    switch (t) {
        case DType::F32:
        case DType::I32:
            return 4;
        case DType::F16:
        case DType::BF16:
            return 2;
        case DType::I64:
            return 8;
        case DType::U8:
        case DType::I8:
            return 1;
    }
    return 0;
}

const char *dtype_name(DType t) {
    switch (t) {
        case DType::F32:
            return "F32";
        case DType::F16:
            return "F16";
        case DType::BF16:
            return "BF16";
        case DType::I32:
            return "I32";
        case DType::I64:
            return "I64";
        case DType::U8:
            return "U8";
        case DType::I8:
            return "I8";
    }
    return "?";
}

static DType parse_dtype(const std::string &s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "I32") return DType::I32;
    if (s == "I64") return DType::I64;
    if (s == "U8") return DType::U8;
    if (s == "I8") return DType::I8;
    fail("safetensors: unsupported dtype " + s);
}

int64_t TensorView::numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

int64_t TensorView::dim(int i) const {
    int n = int(shape.size());
    if (i < 0) i += n;
    QW_CHECK(i >= 0 && i < n, "bad dim index for " + name);
    return shape[size_t(i)];
}

MappedFile::MappedFile(const std::string &path) : path_(path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open " + path);
    struct stat st;
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        fail("cannot stat " + path);
    }
    size_ = size_t(st.st_size);
    void *p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) fail("cannot mmap " + path);
    data_ = static_cast<uint8_t *>(p);
}

MappedFile::~MappedFile() {
    if (data_) ::munmap(data_, size_);
}

void MappedFile::prefetch() const {
    ::madvise(data_, size_, MADV_WILLNEED);
}

void SafeTensors::add_file(const std::string &path) {
    files_.push_back(std::make_unique<MappedFile>(path));
    const MappedFile &f = *files_.back();
    QW_CHECK(f.size() >= 8, "safetensors too small: " + path);
    uint64_t hlen;
    std::memcpy(&hlen, f.data(), 8);
    QW_CHECK(8 + hlen <= f.size(), "safetensors header overruns file: " + path);
    Json h = Json::parse(std::string_view(reinterpret_cast<const char *>(f.data() + 8), hlen));
    const uint8_t *base = f.data() + 8 + hlen;
    size_t body = f.size() - 8 - hlen;
    for (auto &kv : h.obj) {
        if (kv.first == "__metadata__") continue;
        TensorView t;
        t.name = kv.first;
        t.dtype = parse_dtype(kv.second["dtype"].as_str());
        for (auto &d : kv.second["shape"].arr) t.shape.push_back(d.as_int());
        int64_t b = kv.second["data_offsets"][0].as_int();
        int64_t e = kv.second["data_offsets"][1].as_int();
        QW_CHECK(b >= 0 && e >= b && size_t(e) <= body, "bad offsets for " + t.name);
        t.data = base + b;
        t.nbytes = size_t(e - b);
        QW_CHECK(t.nbytes == size_t(t.numel()) * dtype_size(t.dtype), "size mismatch for " + t.name);
        QW_CHECK(tensors_.count(t.name) == 0, "duplicate tensor " + t.name);
        tensors_.emplace(t.name, std::move(t));
    }
}

void SafeTensors::add_index(const std::string &dir, const std::vector<std::string> &extra) {
    Json idx = Json::parse(read_file(dir + "/model.safetensors.index.json"));
    std::set<std::string> names;
    for (auto &kv : idx["weight_map"].obj) names.insert(kv.second.as_str());
    for (auto &e : extra) names.insert(e);
    for (auto &n : names) add_file(dir + "/" + n);
}

const TensorView &SafeTensors::get(const std::string &name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) fail("missing tensor " + name);
    return it->second;
}

const TensorView &SafeTensors::get(const std::string &name, DType dt, std::vector<int64_t> shape) const {
    const TensorView &t = get(name);
    if (t.dtype != dt || t.shape != shape) {
        std::ostringstream os;
        os << "tensor " << name << ": got " << dtype_name(t.dtype) << " [";
        for (auto d : t.shape) os << d << ",";
        os << "], want " << dtype_name(dt) << " [";
        for (auto d : shape) os << d << ",";
        os << "]";
        fail(os.str());
    }
    return t;
}

std::string read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot read " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace qw
