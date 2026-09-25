# qw: inference engine for Qwen3.8-Flash-Next on AMD Radeon Pro V620 (gfx1030).
# Host code is C++17. GPU code (*.hip) is built with hipcc when present.

CXX      ?= g++
HIPCXX   ?= hipcc
CXXFLAGS ?= -O3 -march=native -g
CXXFLAGS += -std=c++17 -Wall -Wextra -fopenmp-simd -fPIC -Isrc -MMD -MP -pthread
LDFLAGS  += -pthread
GPU_LIBS := -lrocblas

BUILD := build
CORE_SRCS := $(wildcard src/core/*.cpp) $(wildcard src/ref/*.cpp)
CORE_OBJS := $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(CORE_SRCS))

TOOLS := $(patsubst tools/%.cpp,$(BUILD)/%,$(wildcard tools/*.cpp))
TESTS := $(patsubst tests/%.cpp,$(BUILD)/%,$(wildcard tests/test_*.cpp))

.PHONY: all gpu test clean
all: $(TOOLS) $(TESTS)

# --- GPU side (hipcc, gfx1030). Not part of `all` so hosts without ROCm build.
HIPFLAGS ?= -O3 -g
HIPFLAGS += -std=c++17 --offload-arch=gfx1030 -fPIC -Isrc -MMD -MP -Wall
GPU_OBJS := $(patsubst src/%.hip,$(BUILD)/obj/%.o,$(wildcard src/kernels/*.hip src/engine/*.hip)) \
            $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(wildcard src/engine/*.cpp))
GPU_BINS := $(patsubst tools/%.hip,$(BUILD)/%,$(wildcard tools/*.hip)) \
            $(patsubst tests/%.hip,$(BUILD)/%,$(wildcard tests/gpu_test_*.hip))
gpu: $(GPU_BINS) $(BUILD)/libqw_engine.so

# The engine as a shared library with a C API (src/engine/capi.hpp), for server/.
$(BUILD)/libqw_engine.so: $(GPU_OBJS) $(BUILD)/libqw.a
	$(HIPCXX) --offload-arch=gfx1030 -shared -o $@ $(GPU_OBJS) $(BUILD)/libqw.a $(LDFLAGS) $(GPU_LIBS)

$(BUILD)/obj/%.o: src/%.hip
	@mkdir -p $(dir $@)
	$(HIPCXX) $(HIPFLAGS) -c $< -o $@

$(BUILD)/obj/engine/%.o: src/engine/%.cpp
	@mkdir -p $(dir $@)
	$(HIPCXX) $(HIPFLAGS) -x hip -c $< -o $@

# hipcc applies -x hip to every input after a .hip file, so compile each
# program separately and link objects only.
$(BUILD)/obj/tools/%.o: tools/%.hip
	@mkdir -p $(dir $@)
	$(HIPCXX) $(HIPFLAGS) -c $< -o $@

$(BUILD)/obj/tests/%.o: tests/%.hip
	@mkdir -p $(dir $@)
	$(HIPCXX) $(HIPFLAGS) -c $< -o $@

$(BUILD)/%: $(BUILD)/obj/tools/%.o $(GPU_OBJS) $(BUILD)/libqw.a
	$(HIPCXX) --offload-arch=gfx1030 $(filter %.o %.a,$^) -o $@ $(LDFLAGS) $(GPU_LIBS)

$(BUILD)/gpu_test_%: $(BUILD)/obj/tests/gpu_test_%.o $(GPU_OBJS) $(BUILD)/libqw.a
	$(HIPCXX) --offload-arch=gfx1030 $(filter %.o %.a,$^) -o $@ $(LDFLAGS) $(GPU_LIBS)

$(BUILD)/libqw.a: $(CORE_OBJS)
	@mkdir -p $(dir $@)
	ar rcs $@ $^

$(BUILD)/obj/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%: tools/%.cpp $(BUILD)/libqw.a
	$(CXX) $(CXXFLAGS) $< $(BUILD)/libqw.a -o $@ $(LDFLAGS)

$(BUILD)/test_%: tests/test_%.cpp $(BUILD)/libqw.a
	$(CXX) $(CXXFLAGS) $< $(BUILD)/libqw.a -o $@ $(LDFLAGS)

test: $(TESTS)
	@set -e; for t in $(TESTS); do echo "== $$t"; $$t; done

clean:
	rm -rf $(BUILD)

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
.SECONDARY:
