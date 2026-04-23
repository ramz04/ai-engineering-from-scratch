# Async Inference and Hogwild!

> Stop waiting for your own tokens. Let many copies of the same model share one attention cache and think in parallel -- the way Hogwild! SGD let many CPUs share one parameter vector.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lessons 04 (Mini-GPT), 07-08 (RLHF / DPO), 12 (Inference Optimization)
**Time:** ~90 minutes

## Learning Objectives

- Explain why autoregressive decoding is sequential by default and what "async inference" changes about that contract.
- Reconstruct the Hogwild! Inference design: multiple LLM workers writing into a shared, concurrently-updated KV cache and reading each other's partial reasoning in real time.
- Contrast synchronous tool use with AsyncLM-style interrupt tokens and understand why interrupt timing is the hard part.
- Relate asynchronous speculative decoding (SwiftSpec, AMUSD, PEARL) to the broader "don't stall the critical path" pattern that unites all async inference.
- Simulate a 3-worker cooperative reasoning run with a shared cache using only the Python standard library.

## The Problem

A single LLM decode step reads the full model weights to produce exactly one token. On a 70B model at BF16 that is 140 GB of memory traffic per token. Then you do it again. Then again. Every token waits for the previous token. Your chain-of-thought is a strict dependency graph with width one.

Parallelism tricks we already know -- continuous batching, tensor parallelism, PagedAttention -- all exist *below* this dependency graph. They make the single stream faster, but the stream is still serial. If your task is "five GSM8k problems bundled together" or "plan a trip while also drafting the email," the model still reasons about each one in turn. A worker that could already be writing step 3 of the plan sits idle while another region of reasoning finishes step 2 of the email.

Async inference is the observation that the dependency graph does not have to be width one. Multiple workers can write into the same conversation at the same time if they can see each other's tokens while producing their own. That is a concurrency primitive, not a scheduling trick. Hogwild! Inference (Rodionov et al., NeurIPS 2025) shows that modern reasoning models already *know how* to use it, out of the box, if you just give them a shared cache and a prompt that explains the layout.

## The Concept

### Hogwild! SGD, then Hogwild! Inference

In 2011, Recht, Re, Wright, and Niu published Hogwild!, an SGD variant where many CPU threads update the same parameter vector without locks. The updates race; some writes are lost. It still converged, because sparse gradients on shared parameters behave like a form of approximate coordination. The lesson: shared mutable state plus a loose coordination signal beats a tightly serialized queue when the work is parallel-friendly.

Hogwild! Inference ports that lesson to LLM decoding. N instances of the same model (same weights, same tokenizer) share one attention cache. Each step, every worker:

1. Reads the current cache -- including tokens the other workers wrote microseconds ago.
2. Computes its own query over the shared keys and values.
3. Writes its new K, V, and output token into its own region of the cache.

There is no lock. Two workers writing at the same position would be a conflict, so the cache is partitioned by region (worker A owns positions 100..199, worker B owns 200..299) while the *prompt* region is shared. Attention ignores no one: worker A's query attends to the prompt, to worker A's own history, and to worker B's latest tokens.

### The shared KV cache layout

The paper describes three arrangements. The simplest is **contiguous**: a shared prompt block, then one contiguous block per worker.

```
Positions:    [0 .. P-1]  [P .. P+W_A-1]  [P+W_A .. P+W_A+W_B-1]
Contents:      prompt       worker A tokens  worker B tokens
Owned by:      all          worker A only    worker B only
Read by:       all          all              all
```

The **interleaved** layout alternates tokens (A, B, A, B, ...) so position ordering reflects real-time arrival. It needs step-wise synchronization -- every worker emits one token, then all workers see the new row.

The **token-wise synchronous** layout is the full Hogwild!: every worker sees every other worker's token as soon as it is committed, with no barrier. This is the fastest and the most chaotic; it requires that RoPE positional encodings can be *rotated* to their new positions without recomputation, because a token's logical position changes the moment another worker's token arrives.

### RoPE makes it cheap

Standard positional encodings would break the shared cache: moving a token from position 203 to 204 would change its embedding, which would change its K and V, which would invalidate the cache. You would have to recompute everything.

RoPE (Rotary Position Embeddings) has a useful property. For two vectors `q` and `k` rotated by angles `theta_q` and `theta_k`, the attention score depends only on the *difference* `theta_q - theta_k`. If you shift both by the same constant, the score is unchanged. That means a cached K at logical position p can be virtually moved to position p+delta just by rotating it by delta more -- no matmul needed, just a rotation on the already-computed K vector.

Hogwild! exploits this to relocate entire cache blocks as workers add tokens, which is what lets the token-wise layout work without a stop-the-world recomputation.

### Prompting workers to cooperate

No fine-tuning. The paper prompts reasoning models (QwQ-32B, DeepSeek-R1) with three ingredients:

- A **system prompt** that names the workers (Alice, Bob, ...) and describes the shared cache layout.
- **Partial in-context examples** showing brief cross-worker references ("Bob already computed X, I will handle Y").
- Periodic **self-check prompts** -- the classic s1-style "Wait, am I doing redundant work? (yes/no):" -- that force a worker to notice when another worker has overtaken its subtask.

Workers do not need a new vocabulary or new control tokens. The conversation protocol emerges from the prompt and the fact that each worker can literally read what the others just wrote.

### AsyncLM: asynchronous function calling

Hogwild! parallelizes *reasoning*. AsyncLM (Gim, Lee, Zhong, arXiv:2412.07017) parallelizes *tool use*. The synchronous contract is: model emits a tool call, model stops decoding, tool runs, tool returns, model resumes. A 30-second tool call is 30 seconds of idle GPU.

AsyncLM introduces **interrupt tokens**. The model keeps decoding after emitting a function call. When the tool finishes, the executor *interrupts* the token stream by injecting a special `[INTR]` token that carries the tool result. The hard parts:

- **Timing**: the interrupt must not land in the middle of another function's arguments.
- **Syntax enforcement**: a token monitor (a finite-state machine on top of logits) prevents the LLM from emitting `[INTR]` itself and prevents the executor from interrupting during fragile regions.
- **Training or prompting**: small models need fine-tuning to handle interrupts; GPT-4o handles them with few-shot prompting. The reported speedup is 1.6x to 5.4x on BFCL.

Hogwild! and AsyncLM are cousins. Both relax a sequential contract. Hogwild! relaxes *"one token at a time from one model"*. AsyncLM relaxes *"one tool call blocks the stream."*

### Async speculative decoding

Speculative decoding (Leviathan et al., 2023) already creates a draft/target pipeline, but the standard version is synchronous: draft produces K tokens, target verifies, repeat. SwiftSpec (Zhang et al., arXiv:2506.11309) and AMUSD (McDanel, ISCAS 2025) and PEARL (Liu et al., 2025) make this pipeline *asynchronous*:

- The draft model keeps drafting while the target verifies the previous batch.
- Verification and drafting live on different devices or different streams.
- A tree of drafts is pre-built so the next verification can start on *any* branch the target picks.

The pattern is identical to Hogwild! and AsyncLM: remove the barrier, tolerate some wasted work, let the critical path finish faster.

### When does async inference actually help?

| Workload | Async win? | Why |
|---|---|---|
| Single short prompt, single answer | No | The critical path is already one token wide. |
| Multi-problem batch (N GSM8k at once) | Yes | Workers claim disjoint problems; cross-reference for shared sub-facts. |
| Long plan with independent sub-goals | Yes | Workers draft different sub-goals in parallel. |
| Code generation with long test runs | Yes | AsyncLM interrupts keep decode busy during test execution. |
| Strictly left-to-right narrative writing | No | No real parallelism in the work itself. |

The substrate is free. The payoff comes from whether the *task* has independent work streams.

## Build It

We will simulate a 3-worker Hogwild!-style cooperative reasoning run using only the Python standard library. No real LLM -- we use a tiny stub that emits scripted tokens -- but the concurrency primitive (a shared, lock-protected KV cache that every worker reads each step) is real.

### Step 1: The shared cache

A `SharedCache` holds three regions: a shared prompt, and one owned region per worker. Readers see everything; writers only touch their own region.

See `code/main.py`, `SharedCache` class. The key method is `snapshot()`, which returns an immutable view of all committed tokens in real-time order. Every worker calls it once per step.

### Step 2: The worker

A `Worker` has an id, a private scripted plan (its "reasoning trajectory"), and a reference to the shared cache. Each step it:

1. Calls `cache.snapshot()`.
2. Checks whether any other worker has already covered its current subtask. If yes, it skips to the next subtask -- this is the "am I doing redundant work?" self-check, reduced to a string match for the simulation.
3. Emits one token into its own region via `cache.append(worker_id, token)`.

### Step 3: The scheduler

The scheduler advances all workers in lock-step rounds. Within a round, worker order is shuffled to simulate the fact that on real hardware no worker has priority -- writes race. After each round it prints the new row of the cache.

### Step 4: A toy cooperative task

Three workers solve three independent sub-questions of a compound problem:

- Worker A: "What is 17 * 23?"
- Worker B: "Spell 'asynchronous' backwards."
- Worker C: "Name the capital of Mongolia."

Each worker has a scripted token stream for its own answer. The interesting part is that each worker also watches for another worker finishing early and, if it sees a completed answer that overlaps its plan, it reroutes.

The output shows three answers interleaved in the order tokens were committed -- exactly the shape a real Hogwild! run would produce.

## Use It

The reference implementation (PyTorch) is `eqimp/hogwild_llm`. The minimum knobs:

```python
from hogwild_llm import HogwildRunner

runner = HogwildRunner(
    model="Qwen/QwQ-32B",
    num_workers=3,
    cache_layout="token_wise_sync",
    system_prompt_template="hogwild/default_v2",
)

plan = "Solve these three GSM8k problems together: ..."
out = runner.generate(plan, max_new_tokens=1024)
for w in out.workers:
    print(w.worker_id, w.text)
```

For AsyncLM-style tool use with an existing serving stack, the interrupt-token pattern currently lives inside the paper's reference fork of SGLang. The key hooks you would need in any engine:

- A next-token logit processor that enforces a CML grammar via FSM.
- An executor callback that can push a special token into the live token stream.
- A retraining or prompting pass that teaches the model what the interrupt token means.

For async speculative decoding, SwiftSpec and the production-ready SpecForge v0.2 release (LMSYS, 2025) are the closest open implementations.

## Ship It

This lesson produces:

- `outputs/skill-async-inference.md` -- a reusable skill for designing async inference layouts (Hogwild! shared cache, AsyncLM interrupts, SwiftSpec pipelines) and for deciding when *not* to use them.

## Exercises

1. **Easy.** Extend the simulator to 5 workers and re-run the cooperative task with 5 sub-questions. Confirm that total wall-clock rounds stays roughly constant as you add workers, as long as the subtasks are independent.

2. **Medium.** Implement the "redundant work" self-check with a real embedding lookup (use `collections.Counter` over tokens for a cheap surrogate similarity). Measure how often a worker correctly reroutes when another worker finishes its subtask first.

3. **Medium.** Add an `AsyncToolCall` worker that emits a `[CALL foo(x)]` token, then continues decoding. Simulate the tool returning after K rounds by injecting an `[INTR foo=value]` token from outside. Verify the decode stream never blocks.

4. **Hard.** Replace the contiguous cache layout with the token-wise synchronous layout described in the Hogwild! paper -- every worker sees every token the instant it is committed, not at round boundaries. Measure how it changes the interleaving pattern.

5. **Hard.** Build a toy asynchronous speculative decoder: a draft worker that emits K scripted tokens into a candidate region, and a target worker that asynchronously accepts or rejects each one by writing into a verification region. The critical path is the target stream; the draft never blocks it.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Hogwild! Inference | "Parallel LLM decoding" | N instances of the same model sharing one attention cache, each reading the others' tokens in real time and writing only its own region. |
| Shared KV cache | "A big cache" | A cache whose positions are partitioned by writer but visible to every reader -- the substrate for cross-worker attention. |
| RoPE rotation trick | "Move tokens around" | Because RoPE attention depends only on positional *differences*, cached K/V can be relocated to new positions by rotating, without recomputing. |
| Self-check prompt | "Are you sure?" | A periodic prompt (like s1-style "Wait,") that triggers a worker to notice another worker has done its subtask. |
| AsyncLM interrupt token | "Callback" | A special token injected by the executor into the live decode stream when a tool call finishes, so the model never has to block on the tool. |
| Token monitor | "Output constraint" | An FSM on top of the logit distribution that enforces valid CML / interrupt-token placement. |
| Async speculative decoding | "Faster spec decoding" | Drafting and verifying run on different streams or devices; the draft never stalls waiting for a verification result. |
| Critical path | "The slow thing" | The longest chain of token-level dependencies in a decode. Async techniques shorten it by removing barriers. |

## Further Reading

- Rodionov, Garipov, Shutova, Yakushev, Schultheis, Egiazarian, Sinitsin, Kuznedelev, Alistarh, "Hogwild! Inference: Parallel LLM Generation via Concurrent Attention" (NeurIPS 2025, [arXiv:2504.06261](https://arxiv.org/abs/2504.06261)) -- the paper this lesson reconstructs; introduces the shared KV cache + RoPE rotation + collaboration prompt design.
- Gim, Lee, Zhong, "Asynchronous LLM Function Calling" ([arXiv:2412.07017](https://arxiv.org/abs/2412.07017)) -- AsyncLM, the interrupt-token mechanism for non-blocking tool use, with 1.6x-5.4x latency reductions on BFCL.
- Recht, Re, Wright, Niu, "Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent" (NeurIPS 2011) -- the original Hogwild! whose shared-mutable-state design philosophy the inference paper ports to decoding.
- Zhang et al., "SwiftSpec: Ultra-Low Latency LLM Decoding by Scaling Asynchronous Speculative Decoding" ([arXiv:2506.11309](https://arxiv.org/abs/2506.11309)) -- async, disaggregated speculative decoding; the standard reference for pipelined draft/target overlap.
- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding" (ICML 2023) -- the synchronous baseline that SwiftSpec and AMUSD/PEARL later make asynchronous.
- Su, Lu, Pan, Murtadha, Wen, Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding" ([arXiv:2104.09864](https://arxiv.org/abs/2104.09864)) -- the original RoPE paper whose rotational invariance makes Hogwild!'s cache relocation free.
