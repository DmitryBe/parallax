# Operations

How to deploy, redeploy, debug, and reason about costs.

## Initial setup

```bash
# Authenticate Modal (one-time)
modal token new
# OR set MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, MODAL_PROFILE in your env

# Create the shared API key Secret
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > .api_key.local
modal secret create parallax-api-key PARALLAX_API_KEY=$(cat .api_key.local)
```

## Deploy a model

Via the CLI (recommended):

```bash
parallax deploy config/qwen2.5-7b.yaml                    # default Modal app name
parallax deploy config/qwen2.5-7b.yaml --name my-prod     # custom name
parallax serve  config/qwen2.5-7b.yaml --name dev         # dev URL with hot reload
parallax info   config/qwen2.5-7b.yaml                    # inspect resolved config
parallax stop   my-prod                                    # stop deployment
```

Or directly with `modal` (CLI is just a thin env-var wrapper):

```bash
MODEL_CONFIG=config/qwen2.5-7b.yaml modal deploy src/parallax/app.py
MODEL_CONFIG=config/qwen2.5-7b.yaml PARALLAX_APP_NAME=my-prod modal deploy src/parallax/app.py
```

Deploying again with the same config updates the existing app (zero downtime: Modal swaps in the new image, old containers drain).

## Dev / debug

`parallax serve` (or `modal serve src/parallax/app.py`) gives a hot-reload dev URL + streaming logs. URL has `-dev` suffix and dies when the command exits.

```bash
parallax serve config/qwen2.5-7b.yaml
# or with custom name:
parallax serve config/qwen2.5-7b.yaml --name dev-test
```

The URL pattern is:
```
https://<workspace>--<app-name-with-dots-as-dashes>-serve-dev.modal.run
```

E.g. app `parallax-qwen2.5-7b` → URL `https://<workspace>--parallax-qwen2-5-7b-serve-dev.modal.run`.

## Useful Modal commands

```bash
modal app list                          # all apps in the workspace
modal app stop <APP_ID>                 # stop a deployment
modal app logs <APP_ID>                 # recent logs (may hit resource limits on chatty apps)
modal volume ls                         # check our caches: parallax-hf-cache, parallax-vllm-cache
modal secret list                       # confirm parallax-api-key exists
modal profile list                      # confirm auth profile / workspace
```

## Cold start budget

| Phase | Time |
|---|---|
| Container scheduling | 5-15s |
| Image pull (cached) | 5-10s |
| Volume mount + weight load (Qwen-7B, ~15GB) | 3-15s on warm volume; 60-90s on first download |
| vLLM engine init (CUDA graphs, prefix cache) | 15-30s |
| **Total (warm volume)** | **~30-60s** |
| **Total (first-ever start)** | **~3-5 min** |

After scale-down, the volume stays — subsequent cold starts only cost the warm path.

## Tuning

The knobs that matter most, in [`config/*.yaml`](../config):

| Knob | Effect | Tuning advice |
|---|---|---|
| `max_num_seqs` | Concurrent sequences in a single engine | Start at 256 for short contexts; if OOM, drop to 128 |
| `max_model_len` | Max tokens per request | Smaller = more concurrent seqs fit in KV cache. Set to your actual prompt size + output budget + margin |
| `gpu_memory_utilization` | Fraction of VRAM vLLM may use | Default 0.92. Drop to 0.85 if you see OOMs |
| `enable_prefix_caching` | Reuse computation for shared prompt prefixes | **Always on** for workloads with repeated system prompts (labeling, classification) |
| `allow_concurrent_inputs` (Modal) | How many parallel requests a container handles | Match to `max_num_seqs` |
| `scaledown_window` | Seconds idle before scale-to-zero | 300 (5 min) for batch workloads; raise for chattier traffic |
| GPU type | bigger model fits, possibly faster | A10G ($1.10/hr) for 7-9B bf16; L4 ($0.80/hr) cheaper; H100 ($3.95/hr) for 30B+ |

## Costs

Modal bills GPU-time at second granularity. The math for an 8-hour labeling batch on A10G:

- Container active for 8h × $1.10/hr = **$8.80** GPU
- Storage (~30GB cached weights on volume) ≈ $0.06 / month
- Network egress: typically negligible for inference

Add cold-start overhead (~1 min per scaledown cycle): if your batch sustains traffic continuously, irrelevant. If you have idle gaps, tune `scaledown_window` to balance container start cost vs idle GPU billing.

**Setting a budget cap:** Modal UI → workspace settings → spending limit. Recommended: set a monthly soft cap on first deploy (e.g. $50) so a misconfigured infinite loop doesn't surprise you.

## Common failure modes

### `AttributeError: 'State' object has no attribute 'log_stats'`
You hit vLLM 0.8.x's lifespan check before `init_app_state` ran. Make sure `parallax/app.py` uses `vllm_app.router.lifespan_context = lifespan` and not `@on_event("startup")`. See [architecture.md](./architecture.md).

### `No supported config format found in <model>`
The model architecture is newer than what your `vllm` or `transformers` version supports. Either pin to a model the installed vLLM knows, or pin a newer vLLM.

### `Logs query hit resource limit`
The web `modal app logs` endpoint can choke on chatty apps. Workaround: use `modal app logs <ID> --since 5m` or check the Modal web console which paginates better.

### Curl returns 303 on first request
Modal sometimes redirects pre-warm. Retry, or hit a public endpoint first (`/health`) to warm the container before authed requests.

### Container OOM on weight load
The model is too big for the GPU. Either:
- Drop to a smaller model (e.g. Qwen2.5-7B → Qwen2.5-4B-ish)
- Use AWQ/GPTQ 4-bit quantized weights (`hf_model_id: Qwen/Qwen2.5-7B-Instruct-AWQ`, add `quantization: awq` if needed)
- Upgrade GPU class

### "Unauthorized" from caller despite correct key
Confirm the Bearer header format: `Authorization: Bearer <KEY>`. LiteLLM does this for you; raw curl needs the exact prefix.

## Tear-down

```bash
# Stop a single model
modal app stop parallax-qwen2.5-7b

# Delete the secret
modal secret delete parallax-api-key

# Delete the volumes (releases storage cost)
modal volume delete parallax-hf-cache
modal volume delete parallax-vllm-cache
```
