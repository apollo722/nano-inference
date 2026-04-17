# Nano-Inference: Incremental Build Plan

## Vision
A simplified but scalable LLM inference framework — a "nano inference" — that demonstrates the complete inference path with production-grade design patterns. Start simple, grow incrementally, and gain complete understanding of every component.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                    │
│              OpenAI-compatible HTTP endpoints               │
└────────────────────────┬────────────────────────────────────┘
                         │ async add_request()
┌────────────────────────▼────────────────────────────────────┐
│                     Driver (CPU)                            │
│  Async request lifecycle: admit → allocate → schedule →     │
│  dispatch → collect output → respond                        │
│  Runs on CPU, never touches GPU directly                    │
└────────┬──────────────────────────┬─────────────────────────┘
         │ schedule()               │ generate()
┌────────▼───────────┐  ┌───────────▼─────────────────────────┐
│    Scheduler       │  │          Engine                     │
│ Continuous batching│  │  Parallelism orchestration (TP/PP)  │
│ Chunked prefill    │  │  Manages WorkerGroup                │
└────────────────────┘  └──────────┬──────────────────────────┘
                                   │ generate()
                        ┌──────────▼──────────────────────────┐
                        │         Worker (GPU)                │
                        │  Holds Inferencer + KV Cache        │
                        │  Executes on GPU device             │
                        └──────────┬──────────────────────────┘
                                   │ generate()
                        ┌──────────▼──────────────────────────┐
                        │       Inferencer                    │
                        │  Pluggable: TorchInferencer,        │
                        │            HuggingFaceInferencer    │
                        │  Model forward + sampling           │
                        └──────────┬──────────────────────────┘
                                   │
                        ┌──────────▼──────────────────────────┐
                        │     Model + Layers + Kernels        │
                        │  Modular: Attention, MLP, RMSNorm   │
                        │  Vision Encoder (Modular)           │
                        └──────────┬──────────────────────────┘
                                   │
                        ┌──────────▼──────────────────────────┐
                        │     KV Cache Manager                │
                        │  Paged attention block allocator    │
                        │  GPU ↔ CPU swap (future)            │
                        └─────────────────────────────────────┘
```

---

## Target Directory Structure

```
nano-inference/
├── src/nano_inference/
│   ├── api/                        # HTTP API layer
│   │   ├── server.py               # FastAPI app, OpenAI-compatible endpoints
│   │   └── protocol.py             # Request/Response data models (OpenAI format)
│   │
│   ├── core/                       # Core data structures & types
│   │   ├── request.py              # Request, GenerateQuery, GenerateOutput
│   │   ├── config.py               # All configuration dataclasses
│   │   └── sampling.py             # SamplingParams (temperature, top_k, top_p, etc.)
│   │
│   ├── driver/                     # CPU-side orchestration (Driver)
│   │   ├── driver.py               # AsyncDriver: request lifecycle management
│   │   └── output_processor.py     # Token → text conversion, stop condition checking
│   │
│   ├── scheduler/                  # Batching strategies
│   │   ├── scheduler.py            # Base scheduler + ContinuousBatchingScheduler
│   │   └── policy.py               # Scheduling policies (FCFS, priority, etc.)
│   │
│   ├── engine/                     # Parallelism orchestration
│   │   ├── engine.py               # Base engine + TPEngine
│   │   └── worker_group.py         # Spawn and manage worker processes
│   │
│   ├── worker/                     # GPU-side execution
│   │   └── worker.py               # Worker: holds inferencer, executes on device
│   │
│   ├── inferencer/                 # Pluggable inference backends
│   │   ├── base.py                 # InferencerBase ABC
│   │   ├── torch_inferencer.py     # PyTorch manual inference (our main backend)
│   │   ├── hf_inferencer.py        # HuggingFace pipeline (naive baseline)
│   │   └── factory.py              # InferencerFactory
│   │
│   ├── models/                     # Model implementations
│   │   ├── base.py                 # ModelBase ABC
│   │   ├── decoder.py              # Generic Decoder-only architecture
│   │   ├── vision.py               # Vision Encoder (Modular)
│   │   ├── model_loader.py         # Weight loading from HuggingFace
│   │   └── registry.py             # Model name → class mapping
│   │
│   ├── layers/                     # Reusable model layers
│   │   ├── attention.py            # Multi-head attention (with paged KV cache support)
│   │   ├── mlp.py                  # MLP / SwiGLU
│   │   ├── norm.py                 # RMSNorm
│   │   ├── embedding.py            # Token + positional embeddings
│   │   ├── rotary.py               # RoPE (Rotary Position Embedding)
│   │   └── linear.py               # Column/RowParallel linear (for TP)
│   │
│   ├── kv_cache/                   # KV cache management
│   │   ├── allocator.py            # PagedKVCacheAllocator (block-level GPU memory)
│   │   ├── block.py                # KVCacheBlock data structure
│   │   └── manager.py              # High-level cache manager (alloc/free/swap)
│   │
│   ├── sampling/                   # Token sampling
│   │   └── sampler.py              # Top-k, top-p, temperature, greedy, etc.
│   │
│   ├── quantization/               # Quantization support
│   │   ├── base.py                 # QuantizerBase
│   │   └── fp8.py                  # FP8 quantization (W8A8)
│   │
│   ├── distributed/                # Multi-GPU / multi-node
│   │   ├── parallel.py             # TP/PP primitives (all-reduce, all-gather)
│   │   └── comm.py                 # NCCL communicator setup
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py               # Logging infrastructure
│
├── tests/                          # Tests
├── configs/                        # Example YAML configs
├── pyproject.toml
└── PLAN.md
```

---

## Phased Implementation Plan

### Phase 1: Foundation — Single-Request, Single-GPU Inference
**Goal**: End-to-end inference for a single request on a single GPU (CPU-only first). Focus on Qwen3 (Text-only).

- [x] **1.1 Core data structures** — `Request`, `GenerateQuery`, `GenerateOutput`, `SamplingParams`, `Config`
- [x] **1.2 HuggingFace Inferencer** — Naive baseline using `transformers.AutoModelForCausalLM.generate()`
- [x] **1.3 Model layers** — `RMSNorm`, `SwiGLU MLP`, `RotaryEmbedding`, `MultiHeadAttention` (eager)
- [x] **1.4 Qwen3 Decoder model** — Manual PyTorch model using our layers (Text-only)
- [ ] **1.5 Torch Inferencer** — Manual forward + greedy sampling loop, no KV optimization
- [ ] **1.6 Worker** — Simple wrapper that holds an inferencer and runs on a device
- [ ] **1.7 Engine** — Single-worker engine, just forwards to worker
- [ ] **1.8 Driver** — Synchronous driver: receive request → build context → engine.generate → return output
- [ ] **1.9 API + endpoint** — `/v1/completions` endpoint that connects to driver
- [ ] **1.10 Verify** — Compare outputs between HF inferencer and Torch inferencer for correctness

**Deliverable**: `curl` a prompt → get a completion. Both inferencer backends produce matching output.

---

### Phase 2: Async Pipeline & Continuous Batching
**Goal**: Handle multiple concurrent requests efficiently. CPU/GPU separation becomes real.

- [ ] **2.1 Async Driver** — Convert driver to async: `add_request()` is non-blocking, background loop runs scheduling + generation
- [ ] **2.2 Scheduler base** — `SchedulerBase` with `add_tasks()`, `schedule()`, `abort_tasks()`
- [ ] **2.3 Continuous batching** — `ContinuousBatchingScheduler`: maintain running + waiting queues, dynamic batch composition
- [ ] **2.4 Generate context builder** — Batch multiple queries into a single `GenerateContext` for efficient GPU execution
- [ ] **2.5 Streaming output** — SSE streaming for `/v1/completions` and `/v1/chat/completions`
- [ ] **2.6 Output processor** — Incremental token detokenization, stop-condition checking, response assembly
- [ ] **2.7 Request lifecycle** — Proper admission control, cancellation, timeout handling

**Deliverable**: Multiple concurrent requests are batched and processed. Streaming responses work.

---

### Phase 3: Paged Attention & KV Cache Management
**Goal**: Memory-efficient KV cache with paged attention, enabling much higher throughput.

- [ ] **3.1 KV cache block** — `KVCacheBlock` structure with fixed block_size (e.g., 16 tokens)
- [ ] **3.2 Block allocator** — `PagedKVCacheAllocator`: free-list based block allocation on GPU
- [ ] **3.3 Paged attention layer** — Modify attention layer to use block tables (vLLM-style paged attention kernel)
- [ ] **3.4 Cache manager** — Integrate allocator with scheduler: allocate on prefill, extend on decode, free on completion
- [ ] **3.5 Scheduler KV-aware** — Scheduler respects available KV blocks; preemption when OOM (swap/recompute)

**Deliverable**: Paged attention working, KV utilization metrics visible, significantly more concurrent requests.

---

### Phase 4: Vision-Language Model Support (VLM)
**Goal**: Support multimodal inference by integrating vision encoders. Early integration leverages paged attention for high image token counts.

- [ ] **4.1 Vision Encoder** — Implement Qwen3-VL vision encoder / adapter
- [ ] **4.2 VLM Processor** — Processor integration for image tokenization
- [ ] **4.3 Unified Attention** — Support multimodal tokens in attention layers
- [ ] **4.4 Multimodal APIs** — OpenAI-compatible vision request support

**Deliverable**: Full VLM support for Qwen3-VL. Images can be processed alongside text.

---

### Phase 5: Chunked Prefill & Advanced Scheduling
**Goal**: Reduce TTFT variance with chunked prefill. Balance prefill and decode work per step.

- [ ] **5.1 Token-budget scheduler** — Max tokens per step budget, split long prefills into chunks
- [ ] **5.2 Chunked prefill** — Support partial prefill execution with inter-step state
- [ ] **5.3 Fused prefill-decode** — Mix prefill and decode requests in the same batch
- [ ] **5.4 Scheduling policies** — Priority-based, fairness policies, long-query management

**Deliverable**: Stable latency under mixed workloads. Predictable TTFT.

---

### Phase 6: CUDA Graph Support
**Goal**: Reduce CPU-side launch overhead during the decode phase by capturing execution graphs.

- [ ] **6.1 Graph capture utility** — Capture and replay mechanism for stable execution paths
- [ ] **6.2 Decode optimization** — Apply CUDA graphs to the decode forward pass
- [ ] **6.3 Lifecycle management** — Handle graph invalidation/recapture on configuration changes

**Deliverable**: Significant reduction in per-token latency (TPOT) for decode-bound workloads.

---

### Phase 7: Tensor Parallelism (Multi-GPU)
**Goal**: Scale to multiple GPUs with tensor parallelism.

- [ ] **7.1 NCCL communicator** — Initialize process groups, all-reduce / all-gather primitives
- [ ] **7.2 Parallel linear layers** — `ColumnParallelLinear`, `RowParallelLinear` with sharded weights
- [ ] **7.3 TP model** — Shard model across GPUs (attention heads + MLP columns)
- [ ] **7.4 Worker group** — Spawn N workers (one per GPU), coordinate via engine
- [ ] **7.5 TP Engine** — `TPEngine` that broadcasts context and collects results
- [ ] **7.6 Multi-process launch** — `torchrun` / `mp.spawn` integration

**Deliverable**: Model runs across 2+ GPUs with near-linear throughput scaling.

---

### Phase 8: FP8 Quantization
**Goal**: Reduce memory footprint and increase throughput with quantization.

- [ ] **8.1 Quantizer base** — `QuantizerBase` ABC with `quantize_weights()` and `dequantize()`
- [ ] **8.2 FP8 quantizer** — W8A8 FP8 quantization with per-tensor/per-channel scaling
- [ ] **8.3 Quantized linear** — `FP8Linear` layer that stores FP8 weights, computes in FP16/BF16
- [ ] **8.4 Model integration** — Config flag to load model with FP8 weights
- [ ] **8.5 Verify accuracy** — Compare FP8 vs FP16 outputs, measure perplexity delta

**Deliverable**: Model runs in FP8 with minimal accuracy loss, ~2x memory savings.

---

### Phase 9: Speculative Decoding
**Goal**: Reduce per-token latency with draft-then-verify.

- [ ] **9.1 Draft model** — Small draft model (or n-gram predictor)
- [ ] **9.2 Verify step** — Batch verify draft tokens with the main model
- [ ] **9.3 Rejection sampling** — Accept/reject draft tokens based on probability comparison
- [ ] **9.4 Integration** — Hook into the generate loop: draft → verify → accept → output

**Deliverable**: Measurable latency reduction on decode-bound workloads.

---

### Phase 10: Prefill-Decode Disaggregation
**Goal**: Separate prefill and decode into different GPU groups for optimal resource utilization.

- [ ] **10.1 Splitwise driver** — Route prefill vs decode to different engine instances
- [ ] **10.2 KV cache transfer** — Send KV cache from prefill GPUs to decode GPUs
- [ ] **10.3 Independent scaling** — Scale prefill and decode GPU pools independently

---

## Key Design Principles

1. **ABC-first**: Every major component has an abstract base class. This enables pluggability.
2. **Factory pattern**: `InferencerFactory`, `EngineFactory`, `SchedulerFactory` — select implementations via config.
3. **CPU/GPU separation**: Driver + Scheduler = CPU. Worker + Inferencer = GPU. No GPU ops in driver code.
4. **Async pipeline**: Driver runs async loops. GPU work is fire-and-forget from CPU perspective.
5. **Builder pattern for context**: `GenerateContextBuilder` with composable parts — cleanly separates concerns.
6. **Configuration-driven**: Everything configurable via dataclasses, loadable from YAML/CLI.

---

## Implementation Order Summary

- [ ] **Phase 1** (Foundation) → Correctness: single request works end-to-end (Qwen3 Text)
- [ ] **Phase 2** (Async + Batching) → Throughput: multiple requests batched
- [ ] **Phase 3** (Paged Attention) → Memory: efficient KV cache
- [ ] **Phase 4** (VLM Support) → Multimodal: vision integration
- [ ] **Phase 5** (Chunked Prefill) → Latency: stable TTFT under load
- [ ] **Phase 6** (CUDA Graph Support) → Latency: reduced CPU launch overhead
- [ ] **Phase 7** (Tensor Parallel) → Scale: multi-GPU inference
- [ ] **Phase 8** (FP8 Quant) → Efficiency: memory + compute savings
- [ ] **Phase 9** (Spec Decoding) → Latency: faster decode
- [ ] **Phase 10** (PD Disaggregation) → Architecture: independent scaling

Each phase builds on the previous one. Each phase produces a working system that can serve real requests. No phase is wasted — every component stays in the final system.
