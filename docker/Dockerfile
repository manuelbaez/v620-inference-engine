# The qw engine and its OpenAI server, for the llm-backend-amd container
# (llama-swap spawns it like the other backends; see scripts/build-image.sh).
# Build on the serving box: host code is compiled with -march=native.
FROM rocm/dev-ubuntu-24.04:7.2.3

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake build-essential rocblas-dev hip-dev python3 python3-venv ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV ROCM_PATH=/opt/rocm
ENV PATH=$ROCM_PATH/bin:$PATH

WORKDIR /opt/qw
COPY CMakeLists.txt ./
COPY include include
COPY src src
COPY tools tools
COPY bench bench
COPY tests tests
# Only the engine library is needed to serve; objects are dropped in the same layer.
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && cmake --build build -j "$(nproc)" --target qw_engine qw_gpu_cli \
    && find build -name '*.o' -delete \
    && find build -type d -name CMakeFiles -prune -exec rm -rf {} +

RUN python3 -m venv /opt/qw/venv \
    && /opt/qw/venv/bin/pip install --no-cache-dir tokenizers jinja2 aiohttp numpy pillow
COPY server server

# Copies as compute-shader blits (the engine also defaults to this): SDMA
# completions can go missing on this box (docs/DESIGN.md, platform notes).
ENV HSA_ENABLE_SDMA=0
ENV PYTHONPATH=/opt/qw/server
# prefix-cache disk tier: mount a persistent volume here
VOLUME /cache
ENTRYPOINT ["/opt/qw/venv/bin/python", "-m", "qwserve", "--lib", "/opt/qw/build/libqw_engine.so", "--disk-cache-dir", "/cache/prefix"]
