## Build image
```bash
cd /home/edgeai/jlwang/dev_ptl/llm-scaler/vllm
docker buildx build --load \
  -f docker/Dockerfile.lmcache \
  -t llm-scaler-vllm:lmcache \
  --build-arg http_proxy="${http_proxy:-}" \
  --build-arg https_proxy="${https_proxy:-}" \
  --build-arg no_proxy="${no_proxy:-}" \
  .
```

> This image builds vLLM, vllm-xpu-kernels, and KVWeave in the `docker/Dockerfile.lmcache` builder stage and does not depend on a prebuilt vllm-xpu-kernels wheel. The build context must be the current `vllm/` directory, with `patches/vllm_for_multi_arc.patch`, `patches/vllm_xpu_kernels.patch`, and `custom-esimd-kernels-vllm/` present.
>
> To obtain the block size:
> ```bash
> python3 /llm-scaler/vllm/kvq/libraries/kv-quant-offload/tests/calc_vllm_block_size.py \
>   "${MODEL:-Qwen3.5-9B}" \
>   --dtype "${DTYPE:-float16}" \
>   --tp-size "${TP:-1}" \
>   --block-size "${VLLM_INITIAL_BLOCK_SIZE:-64}" \
>   --ma-mode align \
>   --enable-prefix-caching
> ```
> The output includes `vLLM block size: ...`. The startup script applies that value to LMCache `--chunk-size` and vLLM `--block-size` / `--max_num_batched_tokens`.

## Run Container
```bash
docker run -itd --net=host --privileged --ipc=host --device=/dev/dri -v /models:/llm/models -e no_proxy=localhost,127.0.0.1 --memory="256G" --name=lmcache_Test --shm-size="16g" --entrypoint /bin/bash llm-scaler-vllm:lmcache
```

## Test LMCache + vLLM

First set `MODEL_PATH` (the model root directory) and `MODEL` (the model name). All three terminals below derive the model path and KV geometry from `$MODEL_PATH/$MODEL`:

```bash
docker exec -it lmcache_Test bash
export MODEL_PATH=/models
export MODEL=Qwen3.5-9B
export SERVE=Qwen3.5-9B
export LMCACHE_MP_KVWEAVE_NUM_KV_HEADS=$(python3 -c "import json;c=json.load(open('$MODEL_PATH/$MODEL/config.json'));c=c.get('text_config',c);print(c['num_key_value_heads'])")
export LMCACHE_MP_KVWEAVE_HEAD_DIM=$(python3 -c "import json;c=json.load(open('$MODEL_PATH/$MODEL/config.json'));c=c.get('text_config',c);print(c['head_dim'])")
export LMCACHE_LOG_LEVEL=INFO
export LMCACHE_MP_L1_KVWEAVE_QUANT=1
# NUM_KV_HEADS / HEAD_DIM are read from $MODEL_PATH/$MODEL/config.json by the two python3 commands above.
export LMCACHE_MP_KVWEAVE_NUM_THREADS=8   # OpenMP thread count; tune for the CPU, independent of the model.
export LMCACHE_MP_KVWEAVE_PRECOND=1
export LMCACHE_MP_KVWEAVE_SCALING_METHOD=per_channel

# Independent Mamba conv/ssm quantization settings. Unset values fall back to their LINEAR_* defaults.
export LMCACHE_MP_KVWEAVE_LINEAR_QUANT_ENABLED=1
export LMCACHE_MP_KVWEAVE_CONV_SCALING_METHOD=per_token
export LMCACHE_MP_KVWEAVE_CONV_RH=1
export LMCACHE_MP_KVWEAVE_SSM_SCALING_METHOD=per_token
export LMCACHE_MP_KVWEAVE_SSM_RH=1
export LMCACHE_MP_KVWEAVE_LINEAR_ASYM=1

export BLOCK_SIZE=$(python3 /llm-scaler/vllm/kvq/libraries/kv-quant-offload/tests/calc_vllm_block_size.py \
  "$MODEL_PATH/$MODEL" \
  --dtype float16 \
  --tp-size 1 \
  --block-size "${VLLM_INITIAL_BLOCK_SIZE:-64}" \
  --ma-mode align \
  --enable-prefix-caching \
  | awk -F': ' '/^vLLM block size[[:space:]]*:/ {print $2}' \
  | awk '{print $1}')

# With LMCACHE_MP_L2_ENABLE=false, omit --l2-adapter and use L1 only.
export LMCACHE_MP_L2_ENABLE=true
L2_ARGS=()
if [[ "${LMCACHE_MP_L2_ENABLE,,}" == "true" ]]; then
  mkdir -p /lmcache_disk
  L2_ARGS+=(--l2-adapter '{"type": "fs", "base_path": "/lmcache_disk"}')
fi

lmcache server \
  --port 6555 \
  --http-port 8090 \
  --l1-size-gb 5 \
  --eviction-policy LRU \
  --eviction-trigger-watermark 0.7 \
  --eviction-ratio 0.3 \
  --chunk-size "${BLOCK_SIZE}" \
  "${L2_ARGS[@]}"
```

Terminal 2: Start vLLM server
```bash
docker exec -it lmcache_Test bash
export MODEL_PATH=/models
export MODEL=Qwen3.5-9B
export SERVE=Qwen3.5-9B
export LMCACHE_MP_KVWEAVE_NUM_KV_HEADS=$(python3 -c "import json;c=json.load(open('$MODEL_PATH/$MODEL/config.json'));c=c.get('text_config',c);print(c['num_key_value_heads'])")
export LMCACHE_MP_KVWEAVE_HEAD_DIM=$(python3 -c "import json;c=json.load(open('$MODEL_PATH/$MODEL/config.json'));c=c.get('text_config',c);print(c['head_dim'])")
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
export TORCH_LLM_ALLREDUCE=1
export VLLM_USE_V1=1
export W_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LMCACHE_LOG_LEVEL=INFO
export LMCACHE_MP_L1_KVWEAVE_QUANT=1
export LMCACHE_MP_KVWEAVE_NUM_THREADS=8   # OpenMP thread count; tune for the CPU, independent of the model.
export LMCACHE_MP_KVWEAVE_PRECOND=1
export LMCACHE_MP_KVWEAVE_SCALING_METHOD=per_channel

export LMCACHE_MP_KVWEAVE_LINEAR_QUANT_ENABLED=1
export LMCACHE_MP_KVWEAVE_CONV_SCALING_METHOD=per_token
export LMCACHE_MP_KVWEAVE_CONV_RH=1
export LMCACHE_MP_KVWEAVE_SSM_SCALING_METHOD=per_token
export LMCACHE_MP_KVWEAVE_SSM_RH=1
export LMCACHE_MP_KVWEAVE_LINEAR_ASYM=1

export BLOCK_SIZE=$(python3 /llm-scaler/vllm/kvq/libraries/kv-quant-offload/tests/calc_vllm_block_size.py \
  "$MODEL_PATH/$MODEL" \
  --dtype float16 \
  --tp-size 1 \
  --block-size "${VLLM_INITIAL_BLOCK_SIZE:-64}" \
  --ma-mode align \
  --enable-prefix-caching \
  | awk -F': ' '/^vLLM block size[[:space:]]*:/ {print $2}' \
  | awk '{print $1}')

# The vLLM block size is derived dynamically from the model, TP size, and shape.
# This manual example applies it to LMCache --chunk-size and vLLM --block-size / --max_num_batched_tokens.
# VLLM_SERVER_DEV_MODE=1 and --enable-prompt-tokens-details are only used for function tests
VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH/$MODEL" \
  --dtype=float16 \
  --enforce-eager \
  --port 8000 \
  --block-size "${BLOCK_SIZE}" \
  --gpu-memory-util 0.3 \
  --trust-remote-code \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --max_num_batched_tokens="${BLOCK_SIZE}" \
  --max_model_len 100000 \
  -tp=1 \
  --quantization fp8 \
  --served-model-name "$SERVE" \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":6555}}'   --enable-prompt-tokens-details
```

Terminal 3: Test vLLM server
Follow libraries.ai.kvweave/tests/vllm-bench-two-waves.sh
```bash
docker exec -it lmcache_Test bash
# This script uses TOKENIZER_PATH rather than MODEL_PATH. It combines TOKENIZER_PATH and MODEL
# into $TOKENIZER_PATH/$MODEL for the tokenizer and --model path.
export MODEL_PATH=/models
export MODEL=Qwen3.5-9B
export SERVE=Qwen3.5-9B
cd /llm-scaler/vllm/kvq/libraries/kv-quant-offload/tests
SEQ=1 WARMUP=1 NUM_REQUESTS=2 TOKENIZER_PATH="$MODEL_PATH" MODEL="$MODEL" SERVE="$SERVE" INPUT_LEN=40960 MAX_TOKENS=128 bash ./vllm-bench-two-waves.sh
```
## LMCache MP KVWeave Environment Variables

Source: `lmcache/v1/multiprocess/transfer_context/worker_transfer.py`

### Common Startup Variables (Block Size and Model Path)

These are the common inputs to the startup workflow. The first two `docker exec` examples combine `MODEL_PATH` and `MODEL` into `$MODEL_PATH/$MODEL`, calculate a dynamic `BLOCK_SIZE` with `calc_vllm_block_size.py`, and pass the result to both LMCache and vLLM:

| Environment Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/models` | Model root directory, normally mounted into the container as `/models`; combined with `MODEL` to form `$MODEL_PATH/$MODEL` |
| `MODEL` | `Qwen3.5-9B` | Model directory or model name, combined with `MODEL_PATH` to form the final model location, for example `$MODEL_PATH/$MODEL` |
| `SERVE` | `Qwen3.5-9B` | Value exposed by the OpenAI API server as `--served-model-name`; it does not have to match the model directory name |
| `DTYPE` | `float16` | Dtype used to calculate block size; it should normally match the vLLM server `--dtype` |
| `TP` | `1` | Tensor parallel size; block-size calculation passes it as `--tp-size` |
| `VLLM_INITIAL_BLOCK_SIZE` | `64` | Initial block-size hint passed to `calc_vllm_block_size.py`; used as the fallback when the script cannot infer a better value from the model config |
| `BLOCK_SIZE` | Calculated dynamically | Calculated with `python3 /llm-scaler/vllm/kvq/libraries/kv-quant-offload/tests/calc_vllm_block_size.py ...`, then passed to LMCache `--chunk-size` and vLLM `--block-size` / `--max_num_batched_tokens` |
| `LMCACHE_MP_L2_ENABLE` | `true` | Enables filesystem L2. When `true`, pass `--l2-adapter` and use `LMCACHE_MP_L2_PATH`; when `false`, use L1 only |
| `LMCACHE_MP_L2_PATH` | `/lmcache_disk` | In-container data directory for filesystem L2; mount a host directory here for persistence |

Recommended command, matching the examples above:

```bash
export BLOCK_SIZE=$(python3 /llm-scaler/vllm/kvq/libraries/kv-quant-offload/tests/calc_vllm_block_size.py \
  "$MODEL_PATH/$MODEL" \
  --dtype float16 \
  --tp-size 1 \
  --block-size "${VLLM_INITIAL_BLOCK_SIZE:-64}" \
  --ma-mode align \
  --enable-prefix-caching \
  | awk -F': ' '/^vLLM block size[[:space:]]*:/ {print $2}' \
  | awk '{print $1}')
```

This does not hardcode a constant. It calculates an appropriate block size from the model configuration and `tp-size`, then uses it consistently for LMCache and vLLM.

### General Transfer Mode

| Environment Variable | Default | Description |
|---|---|---|
| `LMCACHE_MP_TRANSFER_MODE` | `auto` | Transfer routing mode: `auto` selects by device type (CUDA to `handle`, others to `data`); `handle` forces the IPC/SHM zero-copy path; `data` forces the worker-side gather/scatter copy path |

### L1 KVWeave Quantization and Core Parameters (Attention Path)

| Environment Variable | Default | Description |
|---|---|---|
| `LMCACHE_MP_L1_KVWEAVE_QUANT` | `true` | Enables KVWeave quantization for L1 |
| `LMCACHE_MP_KVWEAVE_NUM_KV_HEADS` | `4` | Number of attention KV heads. **Must match the model**. Read `num_key_value_heads` from `$MODEL_PATH/$MODEL/config.json`; for MHA models without that field, use `num_attention_heads` |
| `LMCACHE_MP_KVWEAVE_HEAD_DIM` | `256` | Attention KV head dimension. **Must match the model**. Read `head_dim` from `$MODEL_PATH/$MODEL/config.json`; if absent, calculate `hidden_size // num_attention_heads` |
| `LMCACHE_MP_KVWEAVE_PRECOND` | `true` | Enables Hadamard preconditioning for the attention path |
| `LMCACHE_MP_KVWEAVE_NUM_THREADS` | `8` | OpenMP thread count used by the native quantization kernel |
| `LMCACHE_MP_KVWEAVE_SCALING_METHOD` | `per_channel` | Default scaling method for the attention path: `per_tensor`, `per_channel`, or `per_token` |

### Shared Linear/Mamba Settings

These settings are shared by conv and ssm and provide fallback defaults for their independent settings.

| Environment Variable | Default | Description |
|---|---|---|
| `LMCACHE_MP_KVWEAVE_LINEAR_MAX_SIZE_RATIO` | `1.20` | Maximum allowed serialized-size ratio for the Linear/Mamba group. This experimental allowance does not apply to the attention group |
| `LMCACHE_MP_KVWEAVE_LINEAR_QUANT_ENABLED` | `true` | Enables quantization for the Linear/Mamba group |
| `LMCACHE_MP_KVWEAVE_CONV_QUANT_ENABLED` | `true` | Independently enables quantization for Mamba `conv_state`; can be disabled independently of `LINEAR_QUANT_ENABLED` |
| `LMCACHE_MP_KVWEAVE_SSM_QUANT_ENABLED` | `true` | Independently enables quantization for Mamba `ssm_state`; can be disabled independently of `LINEAR_QUANT_ENABLED` |
| `LMCACHE_MP_KVWEAVE_LINEAR_RH` | `false` | Default random Hadamard transform (RH) setting for the Linear/Mamba group. Overridden by `CONV_RH` and `SSM_RH`; `SSM_RH` has its own `true` default |
| `LMCACHE_MP_KVWEAVE_LINEAR_ASYM` | `true` | Enables asymmetric (zero-point) quantization for the shared Linear/Mamba conv and ssm group |
| `LMCACHE_MP_KVWEAVE_LINEAR_PRECOND` | `false` | Default Hadamard preconditioning setting for the Linear/Mamba group |
| `LMCACHE_MP_KVWEAVE_LINEAR_SCALING_METHOD` | `per_tensor` | Default scaling method for the Linear/Mamba group. Overridden by `CONV_SCALING_METHOD` and `SSM_SCALING_METHOD` |

### Mamba conv_state Independent Settings

Unset values fall back to the corresponding `LINEAR_*` setting.

| Environment Variable | Default | Description |
|---|---|---|
| `LMCACHE_MP_KVWEAVE_CONV_SCALING_METHOD` | `LMCACHE_MP_KVWEAVE_LINEAR_SCALING_METHOD` (`per_tensor`) | Scaling method for conv_state: `per_tensor`, `per_channel`, or `per_token` |
| `LMCACHE_MP_KVWEAVE_CONV_RH` | `LMCACHE_MP_KVWEAVE_LINEAR_RH` (`false`) | Enables RH for conv_state. **Constraint**: conv has no real head axis, so `rh=true` supports only `per_token`. With `per_tensor` or `per_channel`, it automatically falls back to `scaling_method='per_tensor'` and `rh=False`, logging the reason with `logger.info` |

### Mamba ssm_state Independent Settings

These settings use their independent defaults when unset and do not fall back to `LINEAR_*`.

| Environment Variable | Default | Description |
|---|---|---|
| `LMCACHE_MP_KVWEAVE_SSM_SCALING_METHOD` | `per_channel` | Scaling method for ssm_state: `per_tensor`, `per_channel`, or `per_token` |
| `LMCACHE_MP_KVWEAVE_SSM_RH` | `true` | Enables RH for ssm_state. SSM has a real `(head_num, head_dim)` structure, so all six combinations of scaling method and RH are supported; the default `per_channel` with `rh=true` is valid |

### Notes

- `asym` is currently controlled for both conv and ssm by the shared `LMCACHE_MP_KVWEAVE_LINEAR_ASYM` switch (default `true`); it is not split into separate conv and ssm settings.
- `LMCACHE_MP_KVWEAVE_CONV_SCALING_METHOD` and `LMCACHE_MP_KVWEAVE_SSM_SCALING_METHOD` accept only `{per_tensor, per_channel, per_token}`. An invalid value raises `ValueError` during context creation.
- When `LMCACHE_MP_KVWEAVE_CONV_RH=true` with `LMCACHE_MP_KVWEAVE_CONV_SCALING_METHOD=per_tensor` or `per_channel`, LMCache does not raise an error. It automatically falls back to `per_tensor + rh=False` and logs the reason with `logger.info`. This is a caller/LMCache configuration-layer limitation, not a native-kernel limitation; the lower-level `mamba_quant.quantize_mamba_substate_4bit` function still raises `ValueError` for that combination.