# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Multi-stage OMIX build of llm-scaler-vllm (Intel XPU).
#
#   builder : intel/omix:0.1.0-devel-ubuntu24.04    -- oneAPI 2025.3 DPC++ compiler
#                                                       + oneCCL/oneDNN/oneMKL.  Compiles
#                                                       all native wheels.  NOT shipped.
#   runtime : intel/pytorch:xpu-2.11.0-ubuntu24.04  -- torch 2.11.0+xpu + GPU UMD driver
#                                                       in /opt/venv.  No compiler.  Shipped.
#
# Why: the legacy single-base build (llm-scaler-platform) carried the whole oneAPI
# toolchain into the final image (~25-34GB).  Here the compiler lives only in the
# builder stage; the runtime stage receives pre-built wheels, so the shipped image
# is much smaller.
#
# Scope ("lean core"): vLLM + custom ESIMD kernels + vllm-xpu-kernels + core serving
# deps + bigdl-core (ships vllm_int4_for_multi_arc.so used by the sym_int4 path).
# Dropped vs legacy: MinerU, UCX/NIXL (PD disaggregation), triton reinstall.
#
# Build context root MUST be the `vllm/` dir (so ./patches, ./custom-esimd-kernels-vllm
# and ./examples resolve).

# =========================================================================
# Stage 1: builder -- compile every native wheel with the oneAPI toolchain
# =========================================================================
FROM intel/omix:0.1.0-devel-ubuntu24.04 AS builder

ARG http_proxy
ARG https_proxy
ARG no_proxy
# vllm-xpu-kernels pinned base commit (identical to legacy build).
ARG VLLM_XPU_KERNELS_COMMIT=3cab97a

ENV VIRTUAL_ENV=/opt/buildenv \
    PATH="/root/.local/bin:/opt/buildenv/bin:${PATH}" \
    UV_HTTP_TIMEOUT=500 \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu" \
    UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu" \
    UV_INDEX_STRATEGY="unsafe-best-match" \
    UV_LINK_MODE="copy" \
    VLLM_TARGET_DEVICE=xpu \
    WHEELS=/wheels

# Required: build RUN steps use `source .../setvars.sh` and `set -eo pipefail`,
# both bash-only (unavailable under the default /bin/sh).
SHELL ["bash", "-c"]

COPY ./patches/vllm_for_multi_arc.patch ./patches/vllm_xpu_kernels.patch /tmp/
COPY ./custom-esimd-kernels-vllm /src/custom-esimd-kernels-vllm

# --- build tooling (the oneAPI DPC++ compiler is already in the OMIX devel base) ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
        git curl ca-certificates \
        python3-dev python3-venv python3-pip \
        ninja-build cmake pkg-config numactl

# --- isolated build venv (avoid OMIX system-python PEP 668 block; clean resolution) ---
RUN python3 -m venv ${VIRTUAL_ENV}

# --- uv for fast, deterministic installs (matches legacy build) ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# --- torch MUST match the runtime base (2.11.0+xpu) so SyclExtension .so files link
#     against the same libtorch ABI the runtime ships (rpath $ORIGIN/../../torch/lib) ---
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --upgrade pip wheel setuptools && \
    uv pip install torch==2.11.0+xpu torchaudio torchvision \
        --index-url https://download.pytorch.org/whl/xpu

RUN mkdir -p ${WHEELS}

# ---------------------------------------------------------------------------
# 1a. vLLM v0.21.0 + multi-arc patch  ->  wheel
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    git clone --depth 1 -b v0.21.0 https://github.com/vllm-project/vllm.git /src/vllm && \
    cd /src/vllm && \
    git apply /tmp/vllm_for_multi_arc.patch && \
    uv pip install -r requirements/xpu.txt && \
    uv pip install grpcio-tools protobuf nanobind && \
    export CMAKE_PREFIX_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])'):${CMAKE_PREFIX_PATH:-}" && \
    pip wheel --no-build-isolation --no-deps . -w ${WHEELS} && \
    rm -rf build

# ---------------------------------------------------------------------------
# 1b. custom ESIMD kernels  ->  wheel  (TORCH_XPU_ARCH_LIST=bmg-g21)
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /src/custom-esimd-kernels-vllm && \
    TORCH_XPU_ARCH_LIST=bmg-g21 MAX_JOBS=1 python3 setup.py bdist_wheel && \
    cp dist/*.whl ${WHEELS}/ && \
    rm -rf build dist

---------------------------------------------------------------------------
1c. vllm-xpu-kernels (patch + pinned commit)  ->  wheel
---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    git clone https://github.com/vllm-project/vllm-xpu-kernels.git /src/vllm-xpu-kernels && \
    cd /src/vllm-xpu-kernels && \
    git checkout ${VLLM_XPU_KERNELS_COMMIT} && \
    git apply /tmp/vllm_xpu_kernels.patch && \
    sed -i 's|^--extra-index-url=https://download.pytorch.org/whl/xpu|# --extra-index-url=https://download.pytorch.org/whl/xpu|' requirements.txt && \
    sed -i '/^torch==/s/^/# /' requirements.txt && \
    sed -i 's|^triton-xpu|# triton-xpu|' requirements.txt && \
    sed -i 's|^transformers|# transformers|' requirements.txt && \
    uv pip install -r requirements.txt && \
    export CMAKE_PREFIX_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])'):${CMAKE_PREFIX_PATH:-}" && \
    pip wheel --no-build-isolation --no-deps . -w ${WHEELS} && \
    rm -rf build .deps

# 1d. kvweave
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    git clone --filter=blob:none --no-checkout --depth 1 \
    https://github.com/open-edge-platform/edge-ai-libraries.git /src/kvq && \
    cd /src/kvq && \
    git sparse-checkout set --cone libraries/kv-quant-offload && \
    git checkout && \
    cd libraries/kv-quant-offload && \
    pip wheel --no-build-isolation --no-deps . -w ${WHEELS} && \
    rm -rf build

# 1e. lmcache
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    git clone --branch v0.4.7 https://github.com/LMCache/LMCache.git  /src/LMCache && \
    cd /src/LMCache && \
    uv  pip install -r requirements/common.txt && \
    git apply /src/kvq/libraries/kv-quant-offload/integration/lmcache/patches/lmcache-mp-hybrid.patch && \
    git apply /src/kvq/libraries/kv-quant-offload/integration/lmcache/patches/lmcache-v0.4.7-mp-hybrid-to-kvweave.patch && \
    BUILD_WITH_SYCL=1 pip wheel --no-build-isolation --no-deps . -w ${WHEELS} && \
    rm -rf build


# Wheel inventory (visible in the build log for sanity).
RUN ls -la ${WHEELS}/

# =========================================================================
# Stage 2: runtime -- lightweight serving image (no compiler, no /opt/intel/oneapi)
# =========================================================================
FROM intel/pytorch:xpu-2.11.0-ubuntu24.04

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG ONECCL_INSTALLER="intel-oneccl-2021.15.9.14_offline.sh"
ARG ONECCL_INSTALLER_SHA256="f7ab81b6ed1b10dd35fadec366a78046d8af214888dfd625047ce8953d5aa4ef"

# The base already exports PATH=/opt/venv/bin:... and DEVICE=xpu.
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VLLM_QUANTIZE_Q40_LIB="/opt/venv/lib/python3.12/site-packages/vllm_int4_for_multi_arc.so" \
    VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    LMCACHE_MP_HOST=tcp://localhost \
    LMCACHE_MP_PORT=6555 \
    LMCACHE_MP_HTTP_PORT=8090 \
    LMCACHE_MP_L1_SIZE_GB=5 \
    LMCACHE_MP_EVICTION_POLICY=LRU \
    LMCACHE_MP_EVICTION_TRIGGER_WATERMARK=0.7 \
    LMCACHE_MP_EVICTION_RATIO=0.3 \
    LMCACHE_MP_L1_KVWEAVE_QUANT=1 \
    LMCACHE_MP_KVWEAVE_NUM_KV_HEADS=4 \
    LMCACHE_MP_KVWEAVE_HEAD_DIM=256 \
    LMCACHE_MP_KVWEAVE_NUM_THREADS=8 \
    LMCACHE_MP_L2_PATH=/lmcache_disk

WORKDIR /llm-scaler/vllm

# --- patched source trees (post-patch vLLM, vllm-xpu-kernels, and the ESIMD
#     kernels) kept in the shipped image for reference / editable-install dev
#     workflows.  These are plain COPY (not bind-mounted): unlike the wheels
#     below, they're meant to persist in the final image. ---
COPY --from=builder /src/vllm ./vllm
COPY --from=builder /src/vllm-xpu-kernels ./vllm-xpu-kernels
COPY --from=builder /src/custom-esimd-kernels-vllm ./custom-esimd-kernels-vllm
COPY --from=builder /src/kvq ./kvq
COPY --from=builder /src/LMCache ./LMCache

# --- thin runtime OS deps, vLLM + kernel wheels, lean-core serving deps, and the
#     triton-xpu fix/shrink all in ONE layer: cleanup must happen in the same RUN
#     that creates the bloat, otherwise a later layer can't reclaim the space.
#     Wheels are bind-mounted (not COPY'd) from the builder stage: a COPY would
#     bake them into their own image layer that a later `rm` could never reclaim
#     (union FS layers are immutable); the bind mount makes them visible only for
#     this RUN's lifetime, so nothing from /wheels ever lands in the final image. ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
        ffmpeg libsndfile1 libsm6 libxext6 libgl1 numactl curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fL --retry 3 -o "${ONECCL_INSTALLER}" "https://github.com/uxlfoundation/oneCCL/releases/download/2021.15.9/${ONECCL_INSTALLER}" && \
    printf "%s  %s\n" "${ONECCL_INSTALLER_SHA256}" "${ONECCL_INSTALLER}" > /tmp/oneccl.sha256 && \
    sha256sum -c /tmp/oneccl.sha256 && \
    rm -f /tmp/oneccl.sha256 && \
    bash "${ONECCL_INSTALLER}" -s -x -f /tmp && \
    ONECCL_CUP="/tmp/${ONECCL_INSTALLER%.sh}/packages/intel.oneapi.lin.ccl.runtime,v=2021.15.9+13/cupPayload.cup" && \
    ONECCL_DST="/tmp/oneccl-runtime" && \
    python3 -m zipfile -e "${ONECCL_CUP}" "${ONECCL_DST}" && \
    ONECCL_LIB="${ONECCL_DST}/_installdir/ccl/2021.15/lib" && \
    rm -f "${ONECCL_LIB}/libccl.so" "${ONECCL_LIB}/libccl.so.1" "${ONECCL_LIB}/libccl_openmp.so" "${ONECCL_LIB}/libccl_openmp.so.0" && \
    ln -sf libccl.so.1 "${ONECCL_LIB}/libccl.so" && \
    ln -sf libccl.so.1.0 "${ONECCL_LIB}/libccl.so.1" && \
    ln -sf libccl_openmp.so.0 "${ONECCL_LIB}/libccl_openmp.so" && \
    ln -sf libccl_openmp.so.0.1 "${ONECCL_LIB}/libccl_openmp.so.0" && \
    ONECCL_CPU_LIB="${ONECCL_DST}/_installdir/ccl/2021.15/lib/ccl/cpu/lib" && \
    rm -f "${ONECCL_CPU_LIB}/libccl.so" "${ONECCL_CPU_LIB}/libccl.so.1" "${ONECCL_CPU_LIB}/libccl_openmp.so" "${ONECCL_CPU_LIB}/libccl_openmp.so.0" && \
    ln -sf libccl.so.1 "${ONECCL_CPU_LIB}/libccl.so" && \
    ln -sf libccl.so.1.0 "${ONECCL_CPU_LIB}/libccl.so.1" && \
    ln -sf libccl_openmp.so.0 "${ONECCL_CPU_LIB}/libccl_openmp.so" && \
    ln -sf libccl_openmp.so.0.1 "${ONECCL_CPU_LIB}/libccl_openmp.so.0" && \
    mkdir -p /opt/intel/oneapi/ccl && \
    rm -rf /opt/intel/oneapi/ccl/2021.15 && \
    cp -a /tmp/oneccl-runtime/_installdir/ccl/2021.15 /opt/intel/oneapi/ccl/ && \
    rm "${ONECCL_INSTALLER}" && \
    echo "source /opt/intel/oneapi/setvars.sh --force" >> /root/.bashrc && \
    echo "source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force" >> /root/.bashrc && \
    rm -f /opt/intel/oneapi/ccl/latest && \
    ln -s /opt/intel/oneapi/ccl/2021.15 /opt/intel/oneapi/ccl/latest && \
    find /opt/intel/oneapi/ccl -mindepth 1 -maxdepth 1 -type d -not -name 2021.15 -exec rm -rf {} + && \
    rm -rf "/tmp/${ONECCL_INSTALLER%.sh}" /tmp/oneccl-runtime && \
    pip install -r /llm-scaler/vllm/LMCache/requirements/common.txt && \
    pip install /wheels/vllm-*.whl --extra-index-url https://download.pytorch.org/whl/xpu && \
    pip install --no-deps --force-reinstall \
        /wheels/custom_esimd_kernels_vllm-*.whl \
        /wheels/vllm_xpu_kernels-*.whl \
        /wheels/kvweave*.whl \
        /wheels/lmcache*.whl && \
    pip install transformers==5.8.0 accelerate hf_transfer 'modelscope!=1.15.0' && \
    pip install librosa soundfile decord ijson && \
    pip install bigdl-core==2.4.0b2 && \
    pip uninstall -y triton triton-xpu || true && \
    pip install --no-deps triton-xpu==3.7.0 \
        --extra-index-url https://download.pytorch.org/whl/xpu && \
    pip uninstall -y oneccl oneccl-devel || true && \
    rm -f /opt/venv/lib/libccl.so /opt/venv/lib/libccl.so.1 /opt/venv/lib/libccl.so.1.0 \
        /opt/venv/lib/libccl_openmp.so /opt/venv/lib/libccl_openmp.so.0 /opt/venv/lib/libccl_openmp.so.0.1 && \
    ln -sf /opt/intel/oneapi/ccl/2021.15/lib/libccl.so.1.0 /opt/venv/lib/libccl.so.1 && \
    ln -sf /opt/venv/lib/libccl.so.1 /opt/venv/lib/libccl.so && \
    ln -sf /opt/intel/oneapi/ccl/2021.15/lib/libccl_openmp.so.0.1 /opt/venv/lib/libccl_openmp.so.0 && \
    ln -sf /opt/venv/lib/libccl_openmp.so.0 /opt/venv/lib/libccl_openmp.so && \
    TRITON=/opt/venv/lib/python3.12/site-packages/triton && \
    strip --strip-debug ${TRITON}/_C/libtriton.so && \
    rm -f ${TRITON}/plugins/libMLIRDialectPlugin.so.22.0git && \
    ln -s libMLIRDialectPlugin.so ${TRITON}/plugins/libMLIRDialectPlugin.so.22.0git

# --- entrypoint: start the LMCache MP server, wait for it to report healthy,
#     then exec `vllm serve` with whatever args `docker run` supplied.
#     LMCacheMPConnector connects during vLLM's KV-transfer init, so the server
#     has to be listening first -- hence the healthcheck poll rather than a
#     fixed sleep. The server is only started when the `vllm serve` args
#     actually ask for it, i.e. when `--kv-transfer-config` (or
#     `--kv-transfer-config=...`) is present -- without that flag vLLM never
#     connects to it, so running plain vLLM needs no extra env var. Knobs come
#     from the LMCACHE_MP_* env vars set above and can be overridden with
#     `docker run -e`; ENABLE_LMCACHE=0 force-skips it even when the flag is
#     present.
#     The L2 tier is a filesystem adapter under LMCACHE_MP_L2_PATH -- bind-mount
#     a host dir there (`-v /some/dir:/lmcache_disk`) to keep it across runs,
#     otherwise it lives in the container's writable layer.
#     The kvweave KV geometry is probed from the served model's own config.json:
#     this image takes the model from the `vllm serve` args rather than an env
#     var, so the helper scans "$@" for `--model <path>` / `--model=<path>` / a
#     bare leading positional, then emits `export`s that get eval'd. Any failure
#     (no model arg, HF hub id with no local files, unparseable config) just
#     warns and leaves the LMCACHE_MP_KVWEAVE_* defaults above in place. ---
ENTRYPOINT ["bash", "-c", "\
lmcache_wanted=0; \
for _arg in \"$@\"; do \
  case \"$_arg\" in --kv-transfer-config|--kv-transfer-config=*) lmcache_wanted=1; break;; esac; \
done; \
if [ \"$lmcache_wanted\" = 1 ] && [ \"${ENABLE_LMCACHE:-1}\" != \"0\" ]; then \
  eval \"$(python3 -c '\
import json,os,sys\n\
a=sys.argv[1:];m=None\n\
for i,x in enumerate(a):\n\
    if x==\"--model\" and i+1<len(a): m=a[i+1];break\n\
    if x.startswith(\"--model=\"): m=x[8:];break\n\
else:\n\
    if a and not a[0].startswith(\"-\"): m=a[0]\n\
p=m and (m+\"/config.json\" if os.path.isdir(m) else m)\n\
if not p or not os.path.isfile(p): sys.exit(\"WARN: no local config.json for model %r; keeping kvweave defaults\"%m)\n\
try: c=json.load(open(p))\n\
except Exception as e: sys.exit(\"WARN: cannot parse %s: %s\"%(p,e))\n\
for k in (\"text_config\",\"llm_config\",\"language_config\"):\n\
    if isinstance(c.get(k),dict): c={**c,**c[k]};break\n\
n=c.get(\"num_attention_heads\");kv=c.get(\"num_key_value_heads\",n);hd=c.get(\"head_dim\")\n\
if hd is None and c.get(\"hidden_size\") and n: hd=c[\"hidden_size\"]//n\n\
if kv: print(\"export LMCACHE_MP_KVWEAVE_NUM_KV_HEADS=%d\"%kv)\n\
if hd: print(\"export LMCACHE_MP_KVWEAVE_HEAD_DIM=%d\"%hd)\n\
sys.stderr.write(\"INFO: kvweave geometry from %s: num_kv_heads=%s head_dim=%s\\n\"%(p,kv,hd))\
' \"$@\")\"; \
  echo '=== starting LMCache MP server ==='; \
  mkdir -p \"${LMCACHE_MP_L2_PATH:-/lmcache_disk}\"; \
  LMCACHE_LOG_LEVEL=\"${LMCACHE_LOG_LEVEL:-INFO}\" \
  lmcache server \
    --port \"${LMCACHE_MP_PORT:-6555}\" \
    --http-port \"${LMCACHE_MP_HTTP_PORT:-8090}\" \
    --l1-size-gb \"${LMCACHE_MP_L1_SIZE_GB:-1}\" \
    --eviction-policy \"${LMCACHE_MP_EVICTION_POLICY:-LRU}\" \
    --eviction-trigger-watermark \"${LMCACHE_MP_EVICTION_TRIGGER_WATERMARK:-0.7}\" \
    --eviction-ratio \"${LMCACHE_MP_EVICTION_RATIO:-0.3}\" \
    --chunk-size 1024 \
    --l2-adapter \"{\\\"type\\\": \\\"fs\\\", \\\"base_path\\\": \\\"${LMCACHE_MP_L2_PATH:-/lmcache_disk}\\\"}\" & \
  lmcache_pid=$!; \
  trap 'kill \"$lmcache_pid\" 2>/dev/null || true' EXIT INT TERM; \
  ready=0; \
  for _ in $(seq 1 \"${LMCACHE_STARTUP_TIMEOUT:-60}\"); do \
    kill -0 \"$lmcache_pid\" 2>/dev/null || { echo 'ERROR: LMCache server exited during startup' >&2; exit 1; }; \
    if python3 -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:${LMCACHE_MP_HTTP_PORT:-8090}/healthcheck', timeout=1).read()\" >/dev/null 2>&1; then ready=1; break; fi; \
    sleep 1; \
  done; \
  [ \"$ready\" = 1 ] || { echo 'ERROR: LMCache server did not become healthy' >&2; exit 1; }; \
  echo \"=== LMCache MP server ready (pid=$lmcache_pid, metrics: http://localhost:${LMCACHE_MP_HTTP_PORT:-8090}/metrics) ===\"; \
elif [ \"$lmcache_wanted\" != 1 ]; then \
  echo '=== no --kv-transfer-config in args: skipping LMCache server ==='; \
else \
  echo '=== ENABLE_LMCACHE=0: skipping LMCache server ==='; \
fi; \
echo '=== starting vllm serve ==='; \
exec vllm serve \"$@\"", "--"]