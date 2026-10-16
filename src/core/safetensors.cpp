#include "core/safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <set>
#include <sstream>

#include "core/common.hpp"
#include "core/fp_decode.hpp"
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
        case DType::F8E4M3:
            return 1;
        case DType::U32:
            return 4;
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
        case DType::F8E4M3:
            return "F8_E4M3";
        case DType::U32:
            return "U32";
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
    if (s == "F8_E4M3") return DType::F8E4M3;
    if (s == "U32") return DType::U32;
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

bool MappedFile::lock() const {
    return ::mlock(data_, size_) == 0;
}

void MappedFile::touch() const {
    volatile uint8_t sink = 0;
    for (size_t i = 0; i < size_; i += 4096) sink = sink + data_[i];
    (void)sink;
}

void MappedFile::prefetch() const {
    ::madvise(data_, size_, MADV_WILLNEED);
}

void SafeTensors::add_file(const std::string &path, bool replace) {
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
        QW_CHECK(replace || tensors_.count(t.name) == 0, "duplicate tensor " + t.name);
        tensors_[t.name] = std::move(t);
    }
}

void SafeTensors::add_index(const std::string &dir, const std::vector<std::string> &extra) {
    Json idx = Json::parse(read_file(dir + "/model.safetensors.index.json"));
    std::set<std::string> names;
    for (auto &kv : idx["weight_map"].obj) names.insert(kv.second.as_str());
    for (auto &e : extra) names.insert(e);
    for (auto &n : names) add_file(dir + "/" + n);
}

namespace {
bool ends_with(const std::string &s, const std::string &e) {
    return s.size() >= e.size() && s.compare(s.size() - e.size(), e.size(), e) == 0;
}
const char *MTP_GU = "mtp.layers.0.mlp.experts.gate_up_proj", *MTP_DOWN = "mtp.layers.0.mlp.experts.down_proj";
}  // namespace

bool SafeTensors::synthesized(const std::string &name) const {
    return (name == MTP_GU || name == MTP_DOWN) && !tensors_.count(name) &&
           tensors_.count("mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_2");
}

bool SafeTensors::needs_decoding(const TensorView &t) const {
    if (!ends_with(t.name, ".weight")) return false;
    const std::string base = t.name.substr(0, t.name.size() - 7);
    if (t.dtype == DType::F8E4M3) return tensors_.count(base + ".weight_scale") != 0;  // MXFP8
    if (t.dtype == DType::U8)  // NVFP4, but not the routed experts
        return tensors_.count(base + ".weight_scale_2") &&
               !(t.name.rfind("model.language_model.", 0) == 0 && t.name.find(".mlp.experts.") != std::string::npos);
    return t.dtype == DType::F32 && ends_with(t.name, "norm.weight");
}

void SafeTensors::decode(const std::string &name, Decoded &d) const {
    auto nvfp4_rows = [&](const std::string &base, float *out) {  // [rows][cols] fp32
        const TensorView &w = tensors_.at(base + ".weight"), &s = tensors_.at(base + ".weight_scale");
        const TensorView &g = tensors_.at(base + ".weight_scale_2");
        QW_CHECK(w.dtype == DType::U8 && s.dtype == DType::F8E4M3 && g.dtype == DType::F32 && g.numel() == 1 &&
                     s.dim(0) == w.dim(0) && s.dim(1) * 8 == w.dim(1),
                 base + ": unexpected NVFP4 layout");
        nvfp4_to_f32(w.u8(), s.u8(), *static_cast<const float *>(g.data), size_t(w.dim(0)), size_t(2 * w.dim(1)), out);
    };
    if (name == MTP_GU || name == MTP_DOWN) {
        const bool gu = name == MTP_GU;
        int64_t E = 0;
        while (tensors_.count("mtp.layers.0.mlp.experts." + std::to_string(E) + ".down_proj.weight")) ++E;
        const TensorView &w0 = tensors_.at(std::string("mtp.layers.0.mlp.experts.0.") + (gu ? "gate" : "down") + "_proj.weight");
        const int64_t R = w0.dim(0), K = 2 * w0.dim(1), rows = gu ? 2 * R : R;
        d.data.resize(size_t(E * rows * K));
        std::vector<float> m(size_t(R * K));
        for (int64_t e = 0; e < E; ++e)
            for (int part = 0; part < (gu ? 2 : 1); ++part) {
                nvfp4_rows("mtp.layers.0.mlp.experts." + std::to_string(e) + "." +
                               (gu ? (part ? "up_proj" : "gate_proj") : "down_proj"),
                           m.data());
                uint16_t *dst = d.data.data() + size_t((e * rows + part * R) * K);
                for (size_t i = 0; i < m.size(); ++i) dst[i] = f32_to_bf16(m[i]);
            }
        d.view = {name, DType::BF16, {E, rows, K}, d.data.data(), d.data.size() * 2};
        return;
    }
    const TensorView &t = tensors_.at(name);
    const std::string base = name.substr(0, name.size() - 7);
    if (t.dtype == DType::F32) {
        d.data.resize(size_t(t.numel()));
        const float *f = static_cast<const float *>(t.data);
        for (size_t i = 0; i < d.data.size(); ++i) d.data[i] = f32_to_bf16(f[i]);
        d.view = {name, DType::BF16, t.shape, d.data.data(), d.data.size() * 2};
        return;
    }
    QW_CHECK(t.shape.size() == 2, name + ": a quantized tensor that is not a matrix");
    const int64_t rows = t.dim(0), cols = t.dtype == DType::U8 ? 2 * t.dim(1) : t.dim(1);
    d.data.resize(size_t(rows * cols));
    if (t.dtype == DType::F8E4M3) {
        const TensorView &s = tensors_.at(base + ".weight_scale");
        QW_CHECK(s.dtype == DType::U8 && s.dim(0) == rows && s.dim(1) * 32 == cols, name + ": unexpected MXFP8 scale");
        mxfp8_to_bf16(t.u8(), s.u8(), size_t(rows), size_t(cols), d.data.data());
    } else {
        std::vector<float> m(size_t(rows * cols));
        nvfp4_rows(base, m.data());
        for (size_t i = 0; i < m.size(); ++i) d.data[i] = f32_to_bf16(m[i]);
    }
    d.view = {name, DType::BF16, {rows, cols}, d.data.data(), d.data.size() * 2};
}

const TensorView &SafeTensors::get(const std::string &name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end() && !needs_decoding(it->second)) return it->second;
    if (it == tensors_.end() && !synthesized(name)) fail("missing tensor " + name);
    std::lock_guard<std::mutex> lk(mu_);
    auto &d = decoded_[name];
    if (!d) {
        d = std::make_unique<Decoded>();
        decode(name, *d);
    }
    return d->view;
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
