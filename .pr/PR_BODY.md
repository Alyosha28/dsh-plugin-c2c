Adds `Alyosha28/dsh-plugin-c2c` under **Tools & Capabilities**.

## What it does

Exposes [thu-nics/C2C](https://github.com/thu-nics/C2C) — *Cache-to-Cache: Direct
Semantic Communication Between Large Language Models* (ICLR'26) — as six `c2c_*`
agent tools. Two local models run together: the sharer's KV-cache is projected into
the receiver's cache space and fused before the receiver generates. No text passes
between the two models.

| Tool | Purpose |
|:--|:--|
| `c2c_status` | Is the runtime up, which pair is loaded, which device/dtype/interpreter |
| `c2c_models` | The seven published fuser pairs and which one is loaded |
| `c2c_load` | Build a receiver/sharer/projector triple (downloads weights on first use) |
| `c2c_chat` | Answer with the sharer fused (`mode: c2c`) or receiver alone (`mode: baseline`) |
| `c2c_compare` | Run both conditions on one prompt and return the answers side by side |
| `c2c_unload` | Release device memory |

Plus one `tool:c2c` section in the global system prompt.

## Requirements check

- ✅ `package.json` declares `dsh.bundle.patch` → `./cordis.patch.yml`, which sits
  next to it and inserts the row.
- ✅ Real, working code — 1897 lines: a tool layer, a loopback HTTP daemon that owns
  the weights, a one-command bootstrap for the upstream runtime, and an end-to-end
  verification script.
- ✅ Topics include `dsh-plugin`.

## Installation

```bash
python python/dsh_c2c_setup.py --root ~/factor_digging/c2c --verify   # once
dsh plugin --profile web add Alyosha28/dsh-plugin-c2c
```

The setup step clones upstream and builds the pinned virtual environment; it does
not download weights. The ~3.2 GB of model files arrive on the first `c2c_load`.

## On `peerDependencies` — a deliberate omission

`contributing.md` recommends declaring official `@deepseek-ai/*` packages as
peerDependencies with an explicit prerelease branch. I tried that and hit a wall
worth reporting:

```
"@deepseek-ai/dsh-tools": ">=0.0.1-rc.1 <0.1.0 || >=0.1.0-rc.1 <0.2.0-0"
```

This range — the documented form — **rejects `0.1.5-rc.2`**, which is the currently
released harness. Verified with node-semver 7.8.5:

| range | `0.1.5-rc.2` | `0.1.0-rc.6` | `0.1.9-rc.1` |
|:--|:--|:--|:--|
| `"*"` | ❌ | ❌ | ❌ |
| `">=0.1.0-rc.1 <0.2.0-0"` | ❌ | ✅ | ❌ |
| per-tuple enumeration | ✅ | ✅ | ✅ |

The prerelease rule needs a comparator carrying a prerelease tag on the *same*
`major.minor.patch` tuple, so only the tuple carrying the prerelease is covered —
and `*` excludes prereleases as well. The only correct range is a per-tuple
enumeration, which is ~500 characters for `@deepseek-ai/dsh-tools` alone.

So I omitted the harness peers rather than ship a range that produces `ERESOLVE`
for current harness users. The plugin does not need them declared to work:
`@deepseek-ai/dsh-tools` and `@deepseek-ai/schemastery` resolve from the
installation's shared closure at `$DSH_HOME/profiles/node_modules`, which Node
reaches by walking up from the installed package — verified by importing the
plugin from a published-style install layout with no local `node_modules`.

Happy to add them back in whatever form you prefer. If the intent is simply that
the packages be resolvable at runtime, the current form does that.

## Verified on

arm64 macOS (48 GB unified memory, macOS 26), `torch` 2.14.0,
`transformers` 4.52.4, float16 on MPS, smallest published pair:

| | |
|:--|:--|
| Load, weights on disk | 6.9–7.7 s (2 models + 28 projectors) |
| Decode, receiver only | 39 tok/s |
| Decode, fused | 64 tok/s |
| Weights | ~3.2 GB |

Upstream targets Linux+CUDA; this was ported to and verified on the MPS backend.
One portability finding is baked into the setup script: `transformers` must stay
at ≤ 4.54.x, because `rosetta` clones KV caches by appending to
`DynamicCache.key_cache`, which 4.55 wraps without `.append()` and 4.56 removes.
Installing current `transformers` passes every import and then fails at
model-load time.

## Not claimed

The paper reports 8.5–10.5% accuracy gains. **This plugin has not measured that.**
What is verified is that the pipeline runs and that the fusion changes the output.
A single `c2c_compare` result is a smoke test, not a benchmark.

Only the smallest pair is verified end to end; one sharer per load (upstream
supports multi-sharer fusion); no automated test suite.
