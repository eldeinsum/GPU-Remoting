# API Coverage Workflow

GPU-Remoting tracks CUDA-family support from two directions:

1. Static coverage: APIs available in generated bindings versus APIs declared in `cudasys/src/hooks/*.rs`.
2. Workload coverage: APIs observed while running native CUDA workloads under Nsight Systems.

Generated reports belong under `target/coverage/`, which is ignored by git.

## Static Coverage

```bash
python3 tools/coverage/api_coverage.py \
    --format markdown \
    --output target/coverage/static-api-coverage.md
```

For machine-readable output:

```bash
python3 tools/coverage/api_coverage.py \
    --format json \
    --include-api-list \
    --output target/coverage/static-api-coverage.json
```

Statuses:

- `implemented`: present in generated bindings and declared in hooks.
- `placeholder`: present in generated bindings and still generated as unimplemented.
- `unknown`: present in generated bindings but not found in hooks or placeholders.
- `extra-hook`: declared as a hook but not present in generated bindings. Internal helper hooks can appear here.

## Workload Tracing

Run a native workload under Nsight Systems and join observed APIs with the static coverage map:

```bash
python3 tools/coverage/trace_workload.py \
    --name cuda-memcpy \
    -- tests/cuda_api/build/test_memcpy
```

This requires `nsys` in `PATH`. The tool uses CUDA, NVTX, NCCL, cuBLAS, and cuDNN tracing with CUDA all-API tracing enabled by default.

The trace tool writes:

- `target/coverage/traces/<name>/<name>.nsys-rep`
- `target/coverage/traces/<name>/<name>.sqlite`
- `target/coverage/traces/<name>/api-trace.md`
- `target/coverage/traces/<name>/api-trace.json`

Nsight traces include CUDA calls made inside CUDA libraries. For libraries that GPU-Remoting remotes at the library boundary, such as cuBLAS or cuDNN calls declared in `cudasys/src/hooks/`, those internal driver/runtime calls should be treated as prioritization signals rather than automatic client-surface requirements.

To parse an existing Nsight Systems SQLite export:

```bash
python3 tools/coverage/trace_workload.py \
    --name existing-run \
    --sqlite /path/to/report.sqlite
```

## Recommended Loop

1. Generate static coverage.
2. Trace the target workload natively.
3. Implement APIs that are both observed and not implemented.
4. Add focused native-vs-remoted tests for the API family.
5. Run the larger workload through remoting.
6. Use Compute Sanitizer for memory/API error checks when the focused tests are stable.
