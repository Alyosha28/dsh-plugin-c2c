<div align="center">

# dsh-plugin-c2c

**Cache-to-Cache for DeepSeek Harness.**
Two local LLMs that share a KV-cache instead of exchanging text.

</div>

---

## What this is

A DSH plugin that exposes [thu-nics/C2C](https://github.com/thu-nics/C2C)
(*Cache-to-Cache: Direct Semantic Communication Between Large Language Models*,
ICLR'26) as six `c2c_*` agent tools.

C2C runs two models at once — a **receiver** that answers and one or more
**sharers** — and projects the sharer's KV-cache into the receiver's cache space,
fusing it before the receiver generates anything. The sharer contributes its
internal state directly. No text passes between them.

## Why / the idea behind it

When one model hands work to another today, the handoff is text: model A writes a
completion, model B reads it as tokens. That channel is lossy by construction —
whatever the tokens cannot spell, the receiver never learns.

The C2C paper's observation is that the KV-cache *is* the model's working state, so
the handoff does not have to go through language at all. Project the cache into the
receiver's space, fuse, and the receiver starts from a richer position. The paper
reports 8.5–10.5% higher accuracy than either model alone and 3.0–5.0% over
text-based communication.

This plugin does not invent any of that; it packages the upstream runtime so a DSH
session can drive it. See [Accuracy claims](#accuracy-claims) for what is and is
not verified here.

## Install

```bash
# 1. set up the C2C runtime (clones upstream, builds a venv, pins transformers)
python python/dsh_c2c_setup.py --root ~/factor_digging/c2c --verify

# 2. install the plugin
dsh plugin --profile web add Alyosha28/dsh-plugin-c2c
```

The setup step is one command and does not download weights. The ~3.2 GB of model
files arrive on the first `c2c_load`, cached under `<root>/models/hf`.

Then restart the session — profiles on `patchReload: startup` load plugins at boot.

## Use

```
c2c_load    pair: qwen3_0.6b+qwen2.5_0.5b     # build the fused model
c2c_chat    prompt: "..."                      # answer through the fusion
c2c_compare prompt: "..."                      # fused vs receiver-alone, side by side
c2c_unload                                     # release device memory
```

`c2c_compare` is the interesting one: it runs the same prompt twice, once with the
sharer fused in and once with the receiver alone, and shows both answers. The two
conditions differ only in whether the projection is applied, so the difference you
see is attributable to the fusion.

Six tools total: `c2c_status`, `c2c_models`, `c2c_load`, `c2c_chat`, `c2c_compare`,
`c2c_unload`.

Seven fuser pairs ship upstream, from 0.6B+0.5B up to 8B+7B. The smallest runs
comfortably on a laptop.

## What it does that is actually useful

- **Answers with combined knowledge.** Two small models fused can resolve prompts
  neither handles alone. In testing, a receiver that stalled mid-derivation on a
  multi-step arithmetic prompt produced a different, more complete chain of thought
  once the sharer's cache was fused in.
- **Faster than decoder-level ensembling.** The sharer and the projectors run once
  during the prompt pass, not once per token, so fused decoding costs the same as
  receiver-only decoding. Measured: 64 tok/s fused vs 39 tok/s receiver-only on the
  same prompt.
- **Local and hermetic.** Weights, projection, and generation all stay on the
  machine. The daemon binds loopback only and makes no outbound calls except the
  Hugging Face download on first use.
- **Weights stay warm.** The daemon outlives the plugin realm, so a profile patch
  reload does not drop a loaded pair, and repeated questions skip the load cost
  (~7 s here). Idle pairs are released after `idleUnloadSeconds`.
- **Runs on Apple Silicon.** Upstream targets Linux+CUDA. This was ported to and
  verified on the MPS backend; see below.

## Verified numbers

arm64 Mac, 48 GB unified memory, macOS 26, `torch` 2.14.0, `transformers` 4.52.4,
float16 on MPS, smallest pair (Qwen3-0.6B receiver + Qwen2.5-0.5B-Instruct sharer):

| | |
|:--|:--|
| Load, weights on disk | 6.9–7.7 s (2 models + 28 projectors) |
| Decode, receiver only | 39 tok/s |
| Decode, C2C fused | 64 tok/s |
| Prefill + 96 tokens, fused | 1.5 s |
| Weights | ~3.2 GB |

## Architecture

```
DSH agent ── c2c_* tools ──HTTP/loopback──► c2c_daemon.py ──► rosetta (PyTorch)
  (Node)        (this plugin)                  (Python)        receiver + sharer
                                                               + C2C projectors
```

- **The Node plugin never imports Python or torch.** It spawns the daemon on demand
  and forwards HTTP calls, so plugin load stays fast and a broken Python environment
  cannot break DSH startup.
- **The daemon owns the weights.** A cold load costs seconds and the underlying
  models are not safe for concurrent generation, so work is serialized behind one
  lock instead of paying the load cost per call.

## Requirements

- Python 3.10–3.13 (PyTorch wheel availability).
- macOS (MPS), Linux (CUDA), or CPU. The 8B+7B pair wants ~32 GB+ of device memory.
- **`transformers` is pinned to 4.52.4 by the setup script — this is an upper
  bound, not a floor.** `rosetta` clones KV caches by appending to
  `DynamicCache.key_cache`, which is a plain list only through 4.54.x;
  transformers 4.55 wraps it without `.append()` and 4.56 removes the attribute.
  Both break the entire fusion path, and the failure only appears at model-load
  time. Installing current `transformers` looks fine right up until it is not.
- `accelerate` is required: every upstream loader passes `device_map=`, which
  transformers rejects without it. Upstream lists it only under an optional extra.

## Configuration

Defaults work with no editing — `repoRoot` and `pythonPath` are auto-detected.
Override in the profile's `cordis.patch.yml`:

```yaml
- id: c2c
  config:
    repoRoot: /path/to/c2c           # default: auto-detect
    pythonPath: /path/to/python      # default: <repoRoot>/.venv/bin/python
    port: 8765                       # loopback daemon port
    idleUnloadSeconds: 900           # 0 disables idle release
    loadTimeoutSeconds: 600
```

### Dependency model

`@deepseek-ai/*` packages are declared as **peerDependencies**, matching the other
published DSH plugins, and the plugin ships **no `node_modules`**. At runtime they
resolve through the installation's shared closure at `$DSH_HOME/profiles/node_modules`.

A **symlinked development checkout** is the exception: `dsh plugin add <path>`
installs a symlink, and Node resolves it to its real path, so the module walk starts
in your checkout and never reaches the shared closure. The bundled `.npmrc` sets
`auto-install-peers=true` so a plain `pnpm install` in the checkout materializes them.

### Running the daemon by hand

The daemon is a standalone HTTP service and is usable without DSH:

```bash
python python/c2c_daemon.py --repo-root /path/to/c2c --port 8765
curl -s localhost:8765/models
curl -s -X POST localhost:8765/generate \
     -H 'content-type: application/json' \
     -d '{"prompt":"Say hello.","mode":"c2c","max_new_tokens":48}'
```

## MPS notes

- **Do not enable sampled decoding on MPS.** `torch.multinomial` is known to return
  zero-probability indices there
  ([pytorch#192577](https://github.com/pytorch/pytorch/issues/192577)), which
  silently corrupts sampled output. Greedy decoding is unaffected and is the
  default. The tool description records this.
- dtype resolves to `float16` on MPS, `bfloat16` on CUDA, `float32` on CPU. The
  published projectors are stored bfloat16; they are re-cast to the receiver's dtype
  at construction.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically, so an op with no MPS kernel
  falls back to CPU rather than raising mid-generation.

## Accuracy claims

The paper reports 8.5–10.5% accuracy gains over individual models. **This plugin has
not measured that.** What is verified here is that the pipeline runs end to end and
that the fusion changes the output — a smoke test, not a benchmark. Treating a
single `c2c_compare` result as evidence of the paper's numbers would be wrong; that
needs a benchmark subset evaluated per condition.

## Limitations

- Only the smallest pair is verified end to end. The 1.7B and 8B registry entries
  use the same code path but have not been run here.
- One sharer per load. Upstream supports multi-sharer fusion (a bitmask in
  `kv_cache_index`); this plugin does not expose it yet.
- No automated test suite. `python/verify_c2c.py` is a manual end-to-end check.
- Training, multi-GPU, and the SGLang-backed evaluation harness are out of scope —
  they are CUDA-only and not reproduced.

## Credits

The fusion method and the `rosetta` runtime are
[thu-nics/C2C](https://github.com/thu-nics/C2C) by Tianyu Fu, Zihan Min, Hanling
Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang and Yu Wang. If you use the method,
cite their paper:

```bibtex
@article{fu2025c2c,
    title={Cache-to-Cache: Direct Semantic Communication Between Large Language Models},
    author={Tianyu Fu and Zihan Min and Hanling Zhang and Jichao Yan and Guohao Dai and Wanli Ouyang and Yu Wang},
    journal={arXiv preprint arXiv:2510.03215},
    year={2025},
}
```

This repository packages that runtime for DSH. It contains no model weights and
vendors none of their code.

## License

MIT. Upstream C2C is MIT as well.
