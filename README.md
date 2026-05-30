# grand-pattern-cuda

CUDA implementation of the Grand Pattern primitives for NVIDIA GPUs.

## Overview

This library provides GPU-accelerated kernels and a C host API for the core
primitives used across the Grand Pattern ecosystem:

- **Vibe diffusion** – reaction-diffusion across rooms using CSR graph edges
- **JEPA prediction** – sliding-window linear extrapolation with surprise (MSE)
- **Murmur gossip** – TTL-limited parallel epidemic propagation
- **Conservation checks** – parallel double-entry bookkeeping invariant
- **Fleet reduction** – parallel averaging of room vibes and surprise
- **Anomaly detection** – threshold-based flagging of high-surprise rooms
- **Signal routing** – edge-parallel signal flow through graph edges

## Requirements

- CUDA Toolkit 12+
- `nvcc`
- GNU Make

## Build

```bash
make
```

## Run Tests

```bash
make test
```

30 tests exercise all kernels, the host API, edge cases, and parameter
validation.

## API

```c
#include "grand_pattern_cuda.h"

GpcContext* ctx = gpc_create();
gpc_graph_create(ctx, room_count, GPC_VIBE_DIM);

/* Set edges in CSR format */
GpcCsrEdges edges = {row_ptr, col_idx, weights};
gpc_set_edges_csr(ctx, &edges, edge_count);

/* Load initial state */
gpc_set_vibes(ctx, vibes);
gpc_set_signals(ctx, signals);
gpc_inject_murmurs(ctx, murmurs, n_murmurs);

/* Run one full simulation tick */
gpc_tick(ctx);

/* Query results */
gpc_get_vibes(ctx, out_vibes);
gpc_get_surprise(ctx, out_surprise);
gpc_get_fleet_state(ctx, &fleet_state);
gpc_get_anomalies(ctx, out_flags);
gpc_get_signals_out(ctx, out_signals);

gpc_destroy(ctx);
```

## Kernels

| Kernel | Description |
|--------|-------------|
| `vibe_diffusion_kernel` | Reaction-diffusion with CSR neighbors |
| `jepa_predict_kernel` | Linear extrapolation from sliding window |
| `murmur_gossip_kernel` | TTL-decayed parallel gossip propagation |
| `conservation_kernel` | Parallel L1 conservation check |
| `fleet_reduce_kernel` | Tree reduction for fleet averages |
| `anomaly_detect_kernel` | Per-room threshold flagging |
| `signal_route_kernel` | Edge-parallel signal routing (atomicAdd) |

## License

MIT
