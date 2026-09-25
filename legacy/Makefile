# qw: single-model inference engine for Qwen3.8-Flash-Next on gfx1030 (AMD Radeon Pro V620)
# C17 host code + HIP GPU kernels behind extern "C" shims. No external deps: libc + pthread only.

CC      ?= cc
AR      ?= ar
CFLAGS  = -std=c17 -O3 -march=native -g -Wall -Wextra -Wshadow -Wstrict-prototypes -Iinclude -MMD -MP -pthread
LDFLAGS = -pthread -lm

# --- source discovery (auto-picks up files added by other agents) -----------
SRCS := $(wildcard src/*/*.c)
OBJS := $(patsubst src/%.c,obj/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

TOOLS     := $(wildcard tools/*.c)
TOOL_BINS := $(patsubst tools/%.c,bin/%,$(TOOLS))

TESTS     := $(wildcard tests/test_*.c)
TEST_BINS := $(patsubst tests/test_%.c,bin/test_%,$(TESTS))

# --- HIP (optional): `make` must succeed without hipcc ----------------------
HIP     ?= hipcc
HAVE_HIP := $(shell command -v $(HIP) >/dev/null 2>&1 && echo yes)
HIPCFLAGS = -std=c++17 --offload-arch=gfx1030 -O3

HIP_SRCS := $(wildcard kernel/*.hip)
HIP_OBJS := $(patsubst kernel/%.hip,build/%.o,$(HIP_SRCS))

# --- targets ----------------------------------------------------------------
.PHONY: all lib tools test clean hipsyntax disasm

all: lib tools

lib: build/libqw.a

tools: $(TOOL_BINS)

test: $(TEST_BINS)
	@set -e; for t in $(TEST_BINS); do printf '== %s ==\n' "$$t"; "$$t"; done

# verify the integer dot intrinsics lower to real gfx1030 v_dot instructions
# (runs tools/verify_isa.sh; exits 0 with a message when hipcc is absent).
disasm:
	@sh tools/verify_isa.sh

build/libqw.a: $(OBJS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

obj/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# tools: each tools/foo.c -> bin/foo
# Link order: object first, THEN the library, THEN LDFLAGS. The previous rule
# put $(LDFLAGS) before build/libqw.a, so -pthread was scanned too late and
# tools that pull in the quant layer (pthread) failed to link.
bin/%: tools/%.c build/libqw.a
	@mkdir -p bin
	$(CC) $(CFLAGS) $< build/libqw.a -o $@ $(LDFLAGS)

# tests: each tests/test_foo.c -> bin/test_foo (linked against the static lib)
bin/test_%: tests/test_%.c build/libqw.a
	@mkdir -p bin
	$(CC) $(CFLAGS) $< build/libqw.a -o $@ $(LDFLAGS)

# HIP object build (only when hipcc present; `all` never depends on this)
build/%.o: kernel/%.hip
	@mkdir -p $(dir $@)
	$(HIP) $(HIPCFLAGS) -c $< -o $@

# syntax-check HIP sources; no-op if kernel/ empty or no hipcc
hipsyntax:
ifeq ($(HAVE_HIP),yes)
	@if [ -n "$(HIP_SRCS)" ]; then \
		printf 'HIP syntax check: %s\n' $(HIP_SRCS); \
		$(HIP) $(HIPCFLAGS) -fsyntax-only $(HIP_SRCS); \
	else \
		printf 'kernel/ empty; nothing to check\n'; \
	fi
else
	@printf 'hipcc not found; skipping HIP syntax check\n'
endif

clean:
	rm -rf obj bin build

-include $(DEPS)
