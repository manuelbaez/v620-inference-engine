# Convenience wrapper around CMake (the build is defined in CMakeLists.txt).
BUILD ?= build
JOBS  ?= $(shell nproc)

.PHONY: all configure test clean
all: configure
	cmake --build $(BUILD) -j $(JOBS)

configure:
	@test -f $(BUILD)/CMakeCache.txt || cmake -S . -B $(BUILD) $(CMAKE_ARGS)

test: all
	ctest --test-dir $(BUILD) --output-on-failure -LE gpu

clean:
	rm -rf $(BUILD)
