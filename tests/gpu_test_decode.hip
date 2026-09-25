// Checks the decode kernels against plain CPU computations on random data.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "kernels/decode.hpp"

using namespace qw;
using namespace qw::cfg;

#define CK(x)                                                                                \
    do {                                                                                     \
        hipError_t e = (x);                                                                  \
        if (e != hipSuccess) fail(std::string(#x) + ": " + hipGetErrorString(e));            \
    } while (0)

static std::mt19937 rng(1234);
static float urand(float a = -1.f, float b = 1.f) { return std::uniform_real_distribution<float>(a, b)(rng); }

template <typename T>
static T *to_dev(const std::vector<T> &v) {
    T *d;
    CK(hipMalloc(&d, v.size() * sizeof(T)));
    CK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}
template <typename T>
static std::vector<T> from_dev(const T *d, size_t n) {
    std::vector<T> v(n);
    CK(hipMemcpy(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}

// fp16-roundtrip a value so the CPU side sees what the GPU sees.
static float h(float x) { return f16_to_f32(f32_to_f16(x)); }

static int failures = 0;
static void report(const char *name, const std::vector<float> &got, const std::vector<float> &want, float tol) {
    double max_err = 0, max_ref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_err = std::max(max_err, double(std::fabs(got[i] - want[i])));
        max_ref = std::max(max_ref, double(std::fabs(want[i])));
    }
    double rel = max_err / std::max(max_ref, 1e-12);
    bool ok = rel < tol;
    failures += !ok;
    std::printf("%-28s max_abs %.3e  rel %.3e  %s\n", name, max_err, rel, ok ? "ok" : "FAIL");
}

static void test_gemv() {
    for (auto [N, K] : {std::pair{2560, 6144}, {10240, 2560}, {324, 10240}, {512, 2560}}) {
        for (int M : {1, 4, 8}) {
            std::vector<uint16_t> W(size_t(N) * K), x(size_t(M) * K);
            std::vector<float> Wf(W.size()), xf(x.size());
            for (size_t i = 0; i < W.size(); ++i) { Wf[i] = h(urand() * 0.05f); W[i] = f32_to_f16(Wf[i]); }
            for (size_t i = 0; i < x.size(); ++i) { xf[i] = h(urand()); x[i] = f32_to_f16(xf[i]); }
            std::vector<float> want(size_t(M) * N);
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N; ++n) {
                    double s = 0;
                    for (int k = 0; k < K; ++k) s += double(Wf[size_t(n) * K + k]) * xf[size_t(m) * K + k];
                    want[size_t(m) * N + n] = float(s);
                }
            auto dW = to_dev(W); auto dx = to_dev(x);
            float *dy; CK(hipMalloc(&dy, want.size() * 4));
            gpu::gemv_f16(dW, dx, dy, N, K, M, nullptr);
            CK(hipDeviceSynchronize());
            char name[64];
            std::snprintf(name, sizeof name, "gemv_f16 %dx%d M=%d", N, K, M);
            report(name, from_dev(dy, want.size()), want, 1e-3f);
            CK(hipFree(dW)); CK(hipFree(dx)); CK(hipFree(dy));
        }
    }
}

// A small fake expert bank: only the experts we route to are filled.
static void test_moe() {
    const int E = 16, top_k = TOP_K;
    auto make = [&](int rows, int K, std::vector<int32_t> &ct, std::vector<uint16_t> &sc, std::vector<float> &deq) {
        ct.assign(size_t(E) * rows * K / 8, 0);
        sc.resize(size_t(E) * rows * K / QGROUP);
        deq.resize(size_t(E) * rows * K);
        for (auto &s : sc) s = f32_to_f16(urand(0.001f, 0.02f));
        for (int e = 0; e < E; ++e)
            for (int r = 0; r < rows; ++r)
                for (int k = 0; k < K; ++k) {
                    int q = int(rng() % 16);
                    size_t w = (size_t(e) * rows + r) * (K / 8) + k / 8;
                    ct[w] |= int32_t(uint32_t(q) << (4 * (k % 8)));
                    float s = f16_to_f32(sc[(size_t(e) * rows + r) * (K / QGROUP) + k / QGROUP]);
                    deq[(size_t(e) * rows + r) * K + k] = float(q - 8) * s;
                }
    };
    std::vector<int32_t> gct, uct, dct;
    std::vector<uint16_t> gs, us, ds;
    std::vector<float> gdq, udq, ddq;
    make(FFN, H, gct, gs, gdq);
    make(FFN, H, uct, us, udq);
    make(H, FFN, dct, ds, ddq);
    auto pack = [](const std::vector<int32_t> &ct) {
        std::vector<uint32_t> p(ct.size());
        gpu::pack_int4_words(ct.data(), p.data(), ct.size());
        return p;
    };
    std::vector<int32_t> ids = {3, 0, 15, 7, 8, 1, 12, 5, 9, 2};
    std::vector<float> wts(top_k);
    for (auto &w : wts) w = urand(0.f, 0.3f);
    std::vector<uint16_t> x(H);
    std::vector<float> xf(H);
    for (int i = 0; i < H; ++i) { xf[size_t(i)] = h(urand()); x[size_t(i)] = f32_to_f16(xf[size_t(i)]); }

    std::vector<float> hwant(size_t(top_k) * FFN), ywant(H, 0.f);
    for (int j = 0; j < top_k; ++j) {
        int e = ids[size_t(j)];
        for (int r = 0; r < FFN; ++r) {
            double g = 0, u = 0;
            for (int k = 0; k < H; ++k) {
                g += double(gdq[(size_t(e) * FFN + r) * H + k]) * xf[size_t(k)];
                u += double(udq[(size_t(e) * FFN + r) * H + k]) * xf[size_t(k)];
            }
            float gs_ = float(g);
            hwant[size_t(j) * FFN + r] = gs_ / (1.f + std::exp(-gs_)) * float(u);
        }
    }
    std::vector<float> hq(hwant.size());
    for (size_t i = 0; i < hq.size(); ++i) hq[i] = h(hwant[i]);
    for (int j = 0; j < top_k; ++j) {
        int e = ids[size_t(j)];
        for (int n = 0; n < H; ++n) {
            double s = 0;
            for (int k = 0; k < FFN; ++k) s += double(ddq[(size_t(e) * H + n) * FFN + k]) * hq[size_t(j) * FFN + k];
            ywant[size_t(n)] += wts[size_t(j)] * float(s);
        }
    }

    auto dg = to_dev(pack(gct)), du = to_dev(pack(uct)), dd = to_dev(pack(dct));
    auto dgs = to_dev(gs), dus = to_dev(us), dds = to_dev(ds);
    auto dids = to_dev(ids);
    auto dw = to_dev(wts);
    auto dx = to_dev(x);
    uint16_t *dh; CK(hipMalloc(&dh, size_t(top_k) * FFN * 2));
    float *dy; CK(hipMalloc(&dy, H * 4));
    gpu::moe_gate_up(dg, dgs, du, dus, dids, nullptr, dx, dh, top_k, nullptr);
    CK(hipDeviceSynchronize());
    auto hgot16 = from_dev(dh, size_t(top_k) * FFN);
    std::vector<float> hgot(hgot16.size());
    for (size_t i = 0; i < hgot.size(); ++i) hgot[i] = f16_to_f32(hgot16[i]);
    report("moe_gate_up", hgot, hwant, 2e-3f);
    // feed the down projection the CPU's h so the check isolates the kernel
    std::vector<uint16_t> hq16(hq.size());
    for (size_t i = 0; i < hq.size(); ++i) hq16[i] = f32_to_f16(hq[i]);
    CK(hipMemcpy(dh, hq16.data(), hq16.size() * 2, hipMemcpyHostToDevice));
    gpu::moe_down(dd, dds, dids, nullptr, dw, dh, dy, top_k, false, nullptr);
    CK(hipDeviceSynchronize());
    report("moe_down (fused combine)", from_dev(dy, H), ywant, 1e-3f);
}

static void test_gdn() {
    const int steps = 6;
    std::vector<float> conv_w(size_t(GDN_QKV) * 4), A_log(GDN_V_HEADS), dt_bias(GDN_V_HEADS), nw(GDN_DIM);
    for (auto &v : conv_w) v = urand(-0.5f, 0.5f);
    for (auto &v : A_log) v = urand(-1.f, 1.f);
    for (auto &v : dt_bias) v = urand(-1.f, 1.f);
    for (auto &v : nw) v = urand(0.5f, 1.5f);
    std::vector<float> S_ref(size_t(GDN_V_HEADS) * GDN_DIM * GDN_DIM, 0.f), hist(size_t(3) * GDN_QKV, 0.f);

    auto dconv_w = to_dev(conv_w), dA = to_dev(A_log), ddt = to_dev(dt_bias), dnw = to_dev(nw);
    float *dS, *dring, *dqkv, *dz, *da, *db;
    uint16_t *dout;
    CK(hipMalloc(&dS, S_ref.size() * 4)); CK(hipMemset(dS, 0, S_ref.size() * 4));
    CK(hipMalloc(&dring, size_t(4) * GDN_QKV * 4)); CK(hipMemset(dring, 0, size_t(4) * GDN_QKV * 4));
    CK(hipMalloc(&dqkv, GDN_QKV * 4)); CK(hipMalloc(&dz, GDN_VAL * 4));
    CK(hipMalloc(&da, GDN_V_HEADS * 4)); CK(hipMalloc(&db, GDN_V_HEADS * 4));
    CK(hipMalloc(&dout, GDN_VAL * 2));

    for (int step = 0; step < steps; ++step) {
        std::vector<float> qkv(GDN_QKV), z(GDN_VAL), a(GDN_V_HEADS), b(GDN_V_HEADS);
        for (auto &v : qkv) v = urand(-2.f, 2.f);
        for (auto &v : z) v = urand(-2.f, 2.f);
        for (auto &v : a) v = urand(-2.f, 2.f);
        for (auto &v : b) v = urand(-2.f, 2.f);
        // CPU: same math as src/ref gdn() for one token
        std::vector<float> c(GDN_QKV);
        for (int ch = 0; ch < GDN_QKV; ++ch) {
            float acc = conv_w[size_t(ch) * 4 + 3] * qkv[size_t(ch)];
            for (int d = 1; d < 4; ++d) acc += conv_w[size_t(ch) * 4 + 3 - d] * hist[size_t(3 - d) * GDN_QKV + ch];
            c[size_t(ch)] = acc / (1.f + std::exp(-acc));
        }
        for (int d = 0; d < 2; ++d) std::copy(hist.begin() + (d + 1) * GDN_QKV, hist.begin() + (d + 2) * GDN_QKV, hist.begin() + d * GDN_QKV);
        std::copy(qkv.begin(), qkv.end(), hist.begin() + 2 * GDN_QKV);
        float *q = c.data(), *k = c.data() + GDN_KEY, *v = c.data() + 2 * GDN_KEY;
        for (int hh = 0; hh < GDN_QK_HEADS; ++hh) {
            double nq = 0, nk = 0;
            for (int i = 0; i < GDN_DIM; ++i) { nq += q[hh * GDN_DIM + i] * q[hh * GDN_DIM + i]; nk += k[hh * GDN_DIM + i] * k[hh * GDN_DIM + i]; }
            for (int i = 0; i < GDN_DIM; ++i) {
                q[hh * GDN_DIM + i] /= float(std::sqrt(nq + EPS)) * std::sqrt(float(GDN_DIM));
                k[hh * GDN_DIM + i] /= float(std::sqrt(nk + EPS));
            }
        }
        std::vector<float> want(GDN_VAL);
        for (int hh = 0; hh < GDN_V_HEADS; ++hh) {
            int kh = hh / 3;
            float av = a[size_t(hh)] + dt_bias[size_t(hh)];
            float sp = av > 20 ? av : std::log1p(std::exp(av));
            float decay = std::exp(-std::exp(A_log[size_t(hh)]) * sp), beta = 1.f / (1.f + std::exp(-b[size_t(hh)]));
            float *Sh = S_ref.data() + size_t(hh) * GDN_DIM * GDN_DIM;
            for (int i = 0; i < GDN_DIM * GDN_DIM; ++i) Sh[i] *= decay;
            float u[GDN_DIM], o[GDN_DIM] = {};
            for (int j = 0; j < GDN_DIM; ++j) {
                float s = 0;
                for (int i = 0; i < GDN_DIM; ++i) s += Sh[i * GDN_DIM + j] * k[kh * GDN_DIM + i];
                u[j] = (v[hh * GDN_DIM + j] - s) * beta;
            }
            for (int i = 0; i < GDN_DIM; ++i)
                for (int j = 0; j < GDN_DIM; ++j) Sh[i * GDN_DIM + j] += k[kh * GDN_DIM + i] * u[j];
            for (int j = 0; j < GDN_DIM; ++j)
                for (int i = 0; i < GDN_DIM; ++i) o[j] += Sh[i * GDN_DIM + j] * q[kh * GDN_DIM + i];
            double ss = 0;
            for (int j = 0; j < GDN_DIM; ++j) ss += o[j] * o[j];
            float r = float(1.0 / std::sqrt(ss / GDN_DIM + EPS));
            for (int j = 0; j < GDN_DIM; ++j)
                want[size_t(hh) * GDN_DIM + j] = o[j] * r * nw[size_t(j)] / (1.f + std::exp(-z[size_t(hh) * GDN_DIM + j]));
        }

        CK(hipMemcpy(dqkv, qkv.data(), qkv.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dz, z.data(), z.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(da, a.data(), a.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(db, b.data(), b.size() * 4, hipMemcpyHostToDevice));
        int64_t *dpos; CK(hipMalloc(&dpos, 8));
        int64_t hpos = step; CK(hipMemcpy(dpos, &hpos, 8, hipMemcpyHostToDevice));
        gpu::GdnStep p{dqkv, dring, dconv_w, dz, da, db, dA, ddt, dnw, dS, dout, dpos, GDN_V_HEADS};
        gpu::gdn_decode(p, nullptr);
        CK(hipDeviceSynchronize());
        auto g16 = from_dev(dout, GDN_VAL);
        std::vector<float> got(GDN_VAL);
        for (size_t i = 0; i < got.size(); ++i) got[i] = f16_to_f32(g16[i]);
        char name[64];
        std::snprintf(name, sizeof name, "gdn_decode step %d", step);
        report(name, got, want, 3e-3f);
    }
    report("gdn state after steps", from_dev(dS, S_ref.size()), S_ref, 1e-4f);
}

int main() {
    try {
        int n = 0;
        if (hipGetDeviceCount(&n) != hipSuccess || n == 0) {
            std::printf("no GPU visible; skipping\n");
            return 0;
        }
        test_gemv();
        test_moe();
        test_gdn();
    } catch (const std::exception &e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
