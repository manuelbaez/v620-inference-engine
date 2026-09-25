#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "vec_dot.cuh"

#define N 1024  // 1024 independent length-4 dot products

__global__ void vdot_kernel(const int8_t *a, const int8_t *b, int *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    out[i] = qw_vdot4_q8_row(a + 4 * i, b + 4 * i, 1);
}

static int64_t ref_sum(const int8_t *a, const int8_t *b) {
  int64_t s = 0;
  for (int i = 0; i < 4 * N; i++)
    s += (int)a[i] * (int)b[i];
  return s;
}

int main(void) {
  static int8_t ha[4 * N], hb[4 * N];
  for (int i = 0; i < 4 * N; i++) {
    ha[i] = (int8_t)((i * 7) % 11 - 5);
    hb[i] = (int8_t)(((i * 13) % 7) - 3);
  }
  int64_t ref = ref_sum(ha, hb);

  int8_t *da = NULL, *db = NULL;
  int *dout = NULL;
  if (hipMalloc((void **)&da, 4 * N) != hipSuccess ||
      hipMalloc((void **)&db, 4 * N) != hipSuccess ||
      hipMalloc((void **)&dout, N * sizeof(int)) != hipSuccess) {
    if (hipGetLastError() == hipErrorOutOfMemory ||
        hipGetLastError() == hipErrorNoDevice) {
      printf("SKIP: no GPU\n");
      return 0;
    }
    printf("FAIL: hipMalloc: %s\n", hipGetErrorString(hipGetLastError()));
    return 1;
  }
  if (hipMemcpy(da, ha, 4 * N, hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(db, hb, 4 * N, hipMemcpyHostToDevice) != hipSuccess) {
    printf("FAIL: hipMemcpy H2D: %s\n", hipGetErrorString(hipGetLastError()));
    return 1;
  }
  vdot_kernel<<<1, N>>>(da, db, dout);
  if (hipDeviceSynchronize() != hipSuccess) {
    printf("FAIL: hipDeviceSynchronize: %s\n", hipGetErrorString(hipGetLastError()));
    return 1;
  }
  static int ho[N];
  if (hipMemcpy(ho, dout, N * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
    printf("FAIL: hipMemcpy D2H: %s\n", hipGetErrorString(hipGetLastError()));
    return 1;
  }
  int64_t gpu = 0;
  for (int i = 0; i < N; i++)
    gpu += ho[i];
  if (gpu == ref)
    printf("PASS sum=%lld\n", (long long)ref);
  else {
    for (int i = 0; i < N; i++) {
      int64_t expect = 0;
      for (int j = 0; j < 4; j++)
        expect += (int)ha[4 * i + j] * (int)hb[4 * i + j];
      if (ho[i] != expect) {
        printf("FAIL at index %d: gpu=%d ref=%lld\n", i, ho[i], (long long)expect);
        return 1;
      }
    }
    printf("FAIL: sums differ gpu=%lld ref=%lld\n", (long long)gpu, (long long)ref);
    return 1;
  }
  if (hipFree(da) != hipSuccess || hipFree(db) != hipSuccess ||
      hipFree(dout) != hipSuccess)
    printf("WARN: hipFree: %s\n", hipGetErrorString(hipGetLastError()));
  return 0;
}
