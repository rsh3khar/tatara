# Tatara

An inference engine for Qwen3.6-35B-A3B (affine Q4, group 64) on Apple
silicon, written from scratch in C++ and Metal. One model, compiled into
the engine as static data. OpenAI-compatible HTTP serving, deterministic
output, and speculative decoding with exact verification.

## Requirements

- Apple silicon Mac (measured on an M4 Pro, 48 GiB)
- Xcode command line tools (Apple Clang, Metal)
- CMake 3.25+, Python 3.11+

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests:

```sh
ctest --test-dir build
python3 -m unittest discover -s tests
```

## Model

Download the Qwen3.6-35B-A3B 4-bit Safetensors weights. The exact
expected artifacts and their SHA-256 hashes are listed in
`catalog/model_packages/qwen36-35b-a3b/artifact.toml`; verify a download
byte-for-byte with:

```sh
./bin/tatara inspect /path/to/model \
  --verify-manifest catalog/model_packages/qwen36-35b-a3b/artifact.toml
```

Two entry points appear below: `./bin/tatara` is the Python host
tooling (inspection, preparation); `./build/tatara` is the native
engine (serving, benchmarking, validation).

Then bind the weights to a prepared record (metadata only; weights are
never copied):

```sh
./bin/tatara prepare /path/to/model \
  --package catalog/model_packages/qwen36-35b-a3b/model.toml \
  --output checkpoint.tatara
```

## Serve

```sh
./build/tatara serve serve.toml graph
```

`graph` selects block prefill, the measured configuration; without it
the serve boots in single-token prefill mode and prompt ingestion runs
at decode speed.

`serve.toml`:

```toml
schema_version = 1

[model]
record = "/absolute/path/to/checkpoint.tatara"
artifact_root = "/absolute/path/to/model"

[service]
bind = "127.0.0.1"
port = 11434                      # Ollama's default, so local clients work unchanged
max_context_tokens = 0            # 0 = the machine maximum
default_max_output_tokens = 512
max_concurrent_requests = 1
queue_depth = 0
request_deadline_ms = 0
drain_timeout_ms = 0

[cache]
prompt_reuse = false
budget_bytes = 0

[memory]
os_runtime_reserve_bytes = 4294967296
unified_external_occupancy_bytes = 0
metal_external_occupancy_bytes = 0
graph_object_budget_bytes = 1073741824
graph_scratch_lanes = 3

[observability]
log_format = "json"
metrics = true

[speculative]
enabled = false
```

The surface is OpenAI-compatible: `/v1/chat/completions`,
`/v1/completions` (text or token-id prompts), `/v1/models`,
`/health/live`, `/health/ready`, `/metrics`. Configuration is
fail-closed: unknown keys and unsupported combinations refuse at boot
with named diagnostics. Output length is bounded only by context
capacity.

Speculative decoding verifies drafted tokens against the model before
committing them and falls back to serial decoding on any refusal. It
requires a trained draft checkpoint (`draft_checkpoint` under
`[speculative]`); without one, set `enabled = false` as above.

## Performance

Measured on one Apple M4 Pro (48 GiB), single stream, greedy decoding,
affine-Q4 g64 weights, medians of interleaved runs. One machine class;
other configurations are unmeasured.

The stock `mlx_lm` rows are the same machine, same prompts, greedy,
interleaved runs of MLX 0.32.0, stated as the reference point. The
100k MLX prefill point is from the same instrument in a separate
session.

Prefill (input tok/s, wall, at context length):

| Context      | 4k  | 16k | 32k | 64k | 100k |
|--------------|-----|-----|-----|-----|------|
| tatara       | 788 | 730 | 628 | 472 | 375  |
| mlx_lm 0.32  | 743 | 775 | 625 | 516 | 389  |

Decode (output tok/s at context length; ms per output token):

| Context      | ≤1k  | 4k   | 16k  | 32k  | 64k  | 100k | 160k  | 262k  |
|--------------|------|------|------|------|------|------|-------|-------|
| tatara       | 90   | 84   | 78   | 66   | 51   | 43   | ~30\* | ~20\* |
| TPOT ms      | 11.1 | 11.9 | 12.8 | 15.2 | 19.6 | 23.3 | n/a   | n/a   |
| mlx_lm 0.32  | 85   | 81   | 71   | 61   | 47   | 39   | 30    | 22    |

\* zero-cache marginal instrument at depth, one calibration point.

Speculative decoding (draft block 16, every committed token verified
against the model): 126–142 output tok/s on repetitive and structured
workloads at short context; slower than serial decoding on fresh
single-prompt generation, which is why it is opt-in per configuration.

Memory, per the compiled model plan:

- Weights: 19.0 GiB resident.
- KV cache: 20 KiB per token (10 global-attention layers ×
  2 KV heads × 256 dims × K and V, bf16), which is 5.0 GiB at the
  full 262,144-token window.
- Linear-attention state: 61 MiB per sequence, constant at any context
  (30 gated-delta layers; conv tail + fp32 recurrent state).
- Capacity: the serve computes the admitted context window from the
  machine envelope and configured reserves at boot and prints the
  arithmetic; the full 262,144-token window executes on 48 GiB. One
  sequence per engine (single-stream).

## Other commands

```text
./build/tatara doctor        host capability and capacity
./build/tatara validate      prepared record vs this build and machine
./build/tatara config        validate a configuration file
./build/tatara benchmark     single-stream decode timing
```

## Uninstall

Tatara writes nothing outside the paths you name. Delete this directory
and your model directory; if you installed the Python tooling,
`pip uninstall tatara`.

## License

Apache-2.0 (see `LICENSE` and `NOTICE`). Bundled third-party components
retain their own licenses under `THIRD_PARTY_LICENSES/`.
