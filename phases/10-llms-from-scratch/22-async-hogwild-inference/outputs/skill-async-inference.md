---
name: skill-async-inference
description: Decide when and how to apply async inference patterns -- Hogwild! shared KV cache, AsyncLM interrupt tokens, asynchronous speculative decoding -- to LLM systems, and when to leave decoding serial.
version: 1.0.0
phase: 10
lesson: 22
tags: [inference, async, hogwild, asynclm, speculative-decoding, serving]
---

# Async Inference Design Skill

Use this skill when you are tempted to "just parallelize decoding" and need a structured way to pick a primitive, size the payoff, and avoid shipping a slower system that looks parallel on paper.

## When this skill applies

- A request bundles multiple independent sub-tasks (multi-problem batches, multi-goal plans, multi-tool agents).
- Tool calls routinely block decode for hundreds of milliseconds to tens of seconds.
- Speculative decoding is already on but the draft/verify barrier is the new bottleneck.
- The critical path of a single response is measurably longer than the sum of its parallelizable parts.

## When to skip this skill

- The workload is strictly left-to-right narrative text.
- Batch size per GPU is already high and continuous batching + PagedAttention saturate memory bandwidth.
- The deployment does not expose KV cache internals (most hosted APIs).

## Decision tree

1. Is your latency bottleneck *reasoning* (many independent sub-thoughts) or *tool execution* (external calls)?
   - Reasoning -> evaluate Hogwild! Inference.
   - Tool execution -> evaluate AsyncLM.
   - Draft/verify gap in spec decoding -> evaluate SwiftSpec / AMUSD / PEARL-style async speculation.

2. For Hogwild! Inference, answer three questions:
   - Does your model support RoPE (required for cheap cache-position rotation)? Most modern open models do.
   - Is your inference engine KV-cache-native (vLLM, SGLang, TensorRT-LLM)? You need write access to cache regions.
   - Are your sub-tasks weakly coupled? Workers must be able to work on partially disjoint problems.
   If any answer is no, Hogwild! is not the right tool. Pick something else.

3. For AsyncLM, verify:
   - Your tool executor can push tokens back into a live stream (requires engine support).
   - You can enforce a grammar on outputs (logit processor or token monitor FSM).
   - Your model can either be fine-tuned on interrupt semantics or is large enough to pick it up in-context (paper reports GPT-4o-class does, small Llamas need fine-tuning).

4. For async speculative decoding, verify:
   - Draft and target can live on different streams or devices.
   - Your serving stack exposes a pipelined spec-decoding API (SGLang SpecForge v0.2, custom).

## Shared KV cache layout choices

| Layout | Sync discipline | Pros | Cons |
|---|---|---|---|
| Contiguous per-worker blocks | Barrier per round | Simple, easy to reason about | Workers do not see sub-round progress |
| Interleaved (A, B, A, B, ...) | Barrier per round | Position reflects commit order | Needs per-row synchronization |
| Token-wise concurrent | No barrier | Fastest; true Hogwild! | Requires RoPE rotation and chaos tolerance |

Default to contiguous for a first implementation. Upgrade to token-wise only when you have measured a barrier cost that justifies the engineering.

## Prompt scaffolding for cooperative workers

Ship three prompt ingredients, not two:

- A **system prompt** that names the workers and describes which region each owns.
- **Few-shot in-context examples** showing short cross-worker references ("Bob noted X, so I will do Y").
- A **periodic self-check** injected every N tokens: "Wait, is another worker doing this? (yes/no):". The explicit yes/no forces a discrete choice the worker must commit to.

Do not try to invent new special tokens if you are not fine-tuning. The paper works entirely in-context for reasoning-capable models.

## Tripwires before shipping

- Measure the critical path before and after. Async inference that does not shorten the critical path is pure regression.
- Confirm per-token cost. A Hogwild! 3-worker run has ~3x the per-step memory traffic; the win is in reducing wall-clock steps, not per-step cost.
- Bound worker count. Beyond ~4-8 workers, contention and redundant work dominate in practice.
- Instrument the self-check. If the self-check never fires, workers are probably doing independent work anyway and the cooperation prompt is wasted context.

## References you should be ready to cite

- Rodionov et al., "Hogwild! Inference: Parallel LLM Generation via Concurrent Attention" (NeurIPS 2025, arXiv:2504.06261) -- the shared-cache design and RoPE rotation trick.
- Gim, Lee, Zhong, "Asynchronous LLM Function Calling" (arXiv:2412.07017) -- interrupt tokens for non-blocking tool use.
- Zhang et al., "SwiftSpec: Ultra-Low Latency LLM Decoding by Scaling Asynchronous Speculative Decoding" (arXiv:2506.11309) -- draft/verify pipeline disaggregation.
- Recht, Re, Wright, Niu, "Hogwild!: A Lock-Free Approach to Parallelizing SGD" (NeurIPS 2011) -- the philosophical ancestor.
