/* tests/test_gdn.c — no-framework asserts for the GDN delta-rule reference.
 *
 * Build: cc -std=c17 -Wall -Wextra -Wshadow -Wstrict-prototypes -O2 -Iinclude \
 *        src/core/types.c src/core/config.c src/core/gdn.c tests/test_gdn.c \
 *        -o /tmp/opencode/tg && /tmp/opencode/tg
 *
 * main() returns the count of failures (0 = all pass).
 */
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qw/gdn.h"
#include "qw/config.h"
#include "qw/types.h"

static int g_fail = 0;
#define CHECK(cond) do { \
    if (cond) { printf("  ok   %s\n", #cond); } \
    else { printf("  FAIL %s (line %d)\n", #cond, __LINE__); g_fail++; } \
} while (0)

/* fp32 exact-equality is fine for the hand-computed oracle (tiny values,
 * exact operations); tolerance for the chunked-vs-sequential identity. */
static float tdiff(float a, float b) { float d = a - b; return d < 0 ? -d : d; }

/* ---------------------------------------- test 1: zero state, single step */
/* Hand-computed 2x2 oracle.
 * S=0, g=1 (identity decay), k=[k0,k1], v=[v0,v1], beta=1, q=[q0,q1].
 *   S <- S*g = 0
 *   u <- v - S^T k = v - 0 = [v0,v1]
 *   S <- 0 + 1*(k outer v) = [[k0v0,k0v1],[k1v0,k1v1]]
 *   o <- q^T S = [q0k0v0+q1k1v0, q0k0v1+q1k1v1]
 */
static void test_single_step_zero(void)
{
    printf("test 1: zero state + one step == hand-computed outer product\n");
    const size_t dk = 2, dv = 2;
    float S[4] = { 0,0,0,0 };
    const float k[2] = { 2.0f, 3.0f };
    const float v[2] = { 4.0f, 5.0f };
    const float g = 1.0f, beta = 1.0f;
    const float q[2] = { 1.0f, 2.0f };
    float out[2];

    qw_err e = qw_gdn_update_block(S, k, v, &g, &beta, q, out, dk, dv);
    CHECK(e == QW_OK);

    /* S = [[k0v0,k0v1],[k1v0,k1v1]] = [[8,10],[12,15]] */
    const float S_exp[4] = { 8.0f, 10.0f, 12.0f, 15.0f };
    int s_ok = 1;
    for (size_t i = 0; i < 4; i++)
        if (S[i] != S_exp[i]) s_ok = 0;
    CHECK(s_ok);

    /* o = [q0k0v0+q1k1v0, q0k0v1+q1k1v1] = [1*8+2*12, 1*10+2*15] = [32,40] */
    CHECK(out[0] == 32.0f && out[1] == 40.0f);
}

/* ---------------------------------------------- test 2: decay g=0 wipes */
static void test_decay_zero(void)
{
    printf("test 2: g=0 wipes state before the update\n");
    const size_t dk = 2, dv = 2;
    float S[4] = { 9.0f, -7.0f, 5.0f, 1.0f }; /* nonzero pre-state */
    const float k[2] = { 1.0f, 1.0f };
    const float v[2] = { 2.0f, 3.0f };
    const float g = 0.0f, beta = 0.5f;
    const float q[2] = { 1.0f, 1.0f };
    float out[2];

    qw_err e = qw_gdn_update_block(S, k, v, &g, &beta, q, out, dk, dv);
    CHECK(e == QW_OK);

    /* g=0 -> S=0, u=v=[2,3], S += beta*(k outer u) = 0.5*[[2,3],[2,3]] */
    const float S_exp[4] = { 1.0f, 1.5f, 1.0f, 1.5f };
    int s_ok = 1;
    for (size_t i = 0; i < 4; i++)
        if (S[i] != S_exp[i]) s_ok = 0;
    CHECK(s_ok);
    /* o = [1*1+1*1, 1*1.5+1*1.5] = [2,3] */
    CHECK(out[0] == 2.0f && out[1] == 3.0f);
}

/* ----------------------------- test 3: beta=0 pure read of previous state */
static void test_beta_zero(void)
{
    printf("test 3: beta=0 -> no rank-1 update; out = q^T(S*g)\n");
    const size_t dk = 2, dv = 2;
    float S[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float k[2] = { 7.0f, 8.0f };   /* irrelevant: beta=0 */
    const float v[2] = { 9.0f, 9.0f };   /* irrelevant: beta=0 */
    const float g = 0.5f, beta = 0.0f;
    const float q[2] = { 1.0f, 1.0f };
    float out[2];

    qw_err e = qw_gdn_update_block(S, k, v, &g, &beta, q, out, dk, dv);
    CHECK(e == QW_OK);

    /* S decays to 0.5*S = [[0.5,1],[1.5,2]] and is NOT further updated. */
    const float S_exp[4] = { 0.5f, 1.0f, 1.5f, 2.0f };
    int s_ok = 1;
    for (size_t i = 0; i < 4; i++)
        if (S[i] != S_exp[i]) s_ok = 0;
    CHECK(s_ok);
    /* o = q^T S = [0.5+1.5, 1+2] = [2,3] */
    CHECK(out[0] == 2.0f && out[1] == 3.0f);
}

/* --------------------------------- test 4: S^T k == v -> delta term zero */
static void test_delta_zero(void)
{
    printf("test 4: S^T k == v -> u == 0 -> no rank-1 change\n");
    const size_t dk = 2, dv = 2;
    float S[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float k[2] = { 1.0f, 2.0f };
    const float v[2] = { 7.0f, 10.0f };  /* S^T k = [1*1+3*2, 2*1+4*2]=[7,10] */
    const float g = 1.0f, beta = 5.0f;   /* nonzero lr, but u == 0 */
    const float q[2] = { 1.0f, 1.0f };
    float out[2];

    /* S^T k (pre-decay, g=1 so S unchanged) must equal v for the test. */
    float stk0 = S[0] * k[0] + S[2] * k[1];
    float stk1 = S[1] * k[0] + S[3] * k[1];
    CHECK(stk0 == v[0] && stk1 == v[1]);

    qw_err e = qw_gdn_update_block(S, k, v, &g, &beta, q, out, dk, dv);
    CHECK(e == QW_OK);

    /* u = v - S^T k = 0, so S is unchanged (g=1 keeps it, beta*0 adds none). */
    const float S_exp[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    int s_ok = 1;
    for (size_t i = 0; i < 4; i++)
        if (S[i] != S_exp[i]) s_ok = 0;
    CHECK(s_ok);
    /* o = q^T S = [1+3, 2+4] = [4,6] */
    CHECK(out[0] == 4.0f && out[1] == 6.0f);
}

/* -------------------- test 5: chunked T-step == T sequential single steps */
static void test_chunked_equals_sequential(void)
{
    printf("test 5: T-step window == T sequential single-step calls\n");
    const size_t dk = 3, dv = 2, T = 5;

    float S[6] = { 0.5f, -1.0f, 2.0f, 0.25f, -0.75f, 3.0f };

    /* Deterministic per-token inputs (token-major, t*len offsets). */
    float k[T * dk], v[T * dv], q[T * dk];
    float g[T], beta[T];
    for (size_t t = 0; t < T; t++) {
        for (size_t i = 0; i < dk; i++) {
            k[t * dk + i] = (float)((t * 3 + i) % 7) - 3.0f;
            q[t * dk + i] = (float)((t * 5 + i) % 5) - 2.0f;
        }
        for (size_t d = 0; d < dv; d++)
            v[t * dv + d] = (float)((t * 2 + d) % 4) - 1.5f;
        g[t]    = 0.5f + 0.1f * (float)t;   /* in (0.5, 0.9) */
        beta[t] = 0.2f + 0.05f * (float)t;  /* in (0.2, 0.4) */
    }

    /* Reference: T sequential single-step calls. */
    float S_seq[6];
    memcpy(S_seq, S, sizeof(S_seq));
    float out_seq[T * dv];
    for (size_t t = 0; t < T; t++) {
        qw_err e = qw_gdn_update_block(S_seq, k + t * dk, v + t * dv,
                                       &g[t], &beta[t], q + t * dk,
                                       out_seq + t * dv, dk, dv);
        if (e != QW_OK) { CHECK(0); return; }
    }

    /* Windowed: one T-step call on an identical starting state. */
    float S_win[6];
    memcpy(S_win, S, sizeof(S_win));
    float out_win[T * dv];
    qw_err e = qw_gdn_update_sliding_window(S_win, k, v, g, beta, q, out_win,
                                            T, dk, dv);
    CHECK(e == QW_OK);

    int state_ok = 1, out_ok = 1;
    for (size_t i = 0; i < dk * dv; i++)
        if (tdiff(S_seq[i], S_win[i]) > 1e-4f) state_ok = 0;
    for (size_t i = 0; i < T * dv; i++)
        if (tdiff(out_seq[i], out_win[i]) > 1e-4f) out_ok = 0;
    CHECK(state_ok);
    CHECK(out_ok);
}

/* ------------------------------- test 6: per-layer-per-sequence state size */
static void test_state_size(void)
{
    printf("test 6: state_bytes == n_gdn*n_v_heads*dk*dv*4*n_seq*n_gdn_layers\n");
    qw_model_cfg cfg = qw_model_qwen38_flash_next_default();
    CHECK(qw_model_validate(&cfg) == QW_OK);

    const int n_gdn_layers = 36;   /* 36 GDN of the 48 layers */
    const size_t n_seq = 2;

    /* Per layer, per sequence: n_v_heads * dk * dv * 4 bytes.
     * 48 * 128 * 128 * 4 = 3145728 B = 3.0 MiB (one layer, one seq). */
    uint64_t per_layer_seq;
    if (qw_gdn_state_bytes(&cfg, 1, 1, &per_layer_seq) != QW_OK) {
        CHECK(0); return;
    }
    const uint64_t expected_per_layer_seq =
        (uint64_t)cfg.gdn_n_v_heads * (uint64_t)cfg.gdn_state_dk *
        (uint64_t)cfg.gdn_state_dv * 4ULL;
    CHECK(per_layer_seq == expected_per_layer_seq);
    printf("  per-layer-per-sequence state = %llu B\n",
           (unsigned long long)per_layer_seq);

    /* Whole buffer: n_gdn_layers * n_seq * per_layer_seq.
     * 36 * 2 * 48 * 128 * 128 * 4. */
    uint64_t total;
    if (qw_gdn_state_bytes(&cfg, n_seq, (size_t)n_gdn_layers, &total) != QW_OK) {
        CHECK(0); return;
    }
    uint64_t expected_total = (uint64_t)n_gdn_layers * (uint64_t)n_seq *
                              expected_per_layer_seq;
    CHECK(total == expected_total);
    printf("  full buffer (n_seq=%zu, n_gdn_layers=%d) = %llu B\n",
           n_seq, n_gdn_layers, (unsigned long long)total);

    /* Per-head block is 128x128 fp32 = 256 KiB; n_v_heads blocks per
     * layer/seq. Sanity the per-head and per-head counts line up. */
    CHECK(per_layer_seq / ((uint64_t)cfg.gdn_state_dk *
                           (uint64_t)cfg.gdn_state_dv * 4ULL)
          == (uint64_t)cfg.gdn_n_v_heads);
}

int main(void)
{
    test_single_step_zero();
    test_decay_zero();
    test_beta_zero();
    test_delta_zero();
    test_chunked_equals_sequential();
    test_state_size();
    if (g_fail == 0) {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d FAILURES\n", g_fail);
    return g_fail;
}
