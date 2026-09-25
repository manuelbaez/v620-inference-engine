#!/usr/bin/env bash
# Prove int8/int4 dot intrinsics lower to RDNA2 (gfx1030) hardware dot instructions.
set -euo pipefail

if ! command -v hipcc >/dev/null 2>&1; then
  echo "SKIP: hipcc not installed; cannot verify ISA lowering"
  exit 0
fi

tmp="$(mktemp -d /tmp/opencode/verify_isa.XXXXXX)"
trap 'rm -rf "$tmp"' EXIT

src="$tmp/dots.hip"
cat > "$src" <<'EOF'
#include <hip/hip_runtime.h>
typedef short v2s __attribute__((ext_vector_type(2)));
__global__ void k_sdot4(int *o, const int *a, const int *b) {
  *o = __builtin_amdgcn_sdot4(a[0], b[0], *o, false);
}
__global__ void k_udot8(int *o, const int *a, const int *b) {
  *o = __builtin_amdgcn_udot8(a[0], b[0], *o, false);
}
__global__ void k_sdot2(int *o, const short *a, const short *b) {
  *o = __builtin_amdgcn_sdot2(*(const v2s *)a, *(const v2s *)b, *o, false);
}
EOF

asm="$tmp/out.s"
hipcc --offload-arch=gfx1030 -S -o "$asm" "$src"

# gfx1030 emits: v_dot4c_i32_i8, v_dot8_u32_u4, v_dot2_i32_i16
rc=0
for m in "v_dot4.*i32_i8" "v_dot8_(i32_i4|u32_u4)" "v_dot2.*i32_i16"; do
  if grep -Eq "$m" "$asm"; then
    echo "PASS $m"
  else
    echo "FAIL $m"
    rc=1
  fi
done
exit "$rc"
