"""Parallax — OpenAI-compatible vLLM endpoint on Modal.

Deploy:
  MODEL_CONFIG=config/qwen3-8b.yaml modal deploy parallax/app.py --name parallax-qwen3-8b

The model config is read at *deploy time* and baked into the image.
One Modal app == one model. Add a new model = new config + new deploy.
"""
from __future__ import annotations
import os
import modal

from parallax.config import ModelConfig

# ---------------------------------------------------------------------------
# Config (read at deploy time)
# ---------------------------------------------------------------------------
CFG = ModelConfig.from_env()
APP_NAME = os.environ.get("MODAL_APP_NAME", f"parallax-{CFG.name}")

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
VLLM_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",  # current stable (0.8.x)
        "fastapi[standard]==0.115.6",
        "huggingface_hub[hf_transfer]==0.27.0",
        "pyyaml==6.0.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("parallax")
    .add_local_dir(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"),
        remote_path="/root/config",
    )
)

# ---------------------------------------------------------------------------
# Persistent storage: weights cache + vLLM compile cache
# ---------------------------------------------------------------------------
HF_CACHE = modal.Volume.from_name("parallax-hf-cache", create_if_missing=True)
VLLM_CACHE = modal.Volume.from_name("parallax-vllm-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Auth secret
# ---------------------------------------------------------------------------
API_KEY_SECRET = modal.Secret.from_name(
    "parallax-api-key",
    required_keys=["PARALLAX_API_KEY"],
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App(APP_NAME)


@app.function(
    image=VLLM_IMAGE,
    gpu=f"{CFG.gpu}:{CFG.gpu_count}" if CFG.gpu_count > 1 else CFG.gpu,
    volumes={
        "/root/.cache/huggingface": HF_CACHE,
        "/root/.cache/vllm": VLLM_CACHE,
    },
    secrets=[API_KEY_SECRET],
    scaledown_window=CFG.scaledown_window,
    timeout=CFG.timeout,
    min_containers=CFG.min_containers,
)
@modal.concurrent(max_inputs=CFG.allow_concurrent_inputs)
@modal.asgi_app()
def serve():
    """Mount vLLM's OpenAI server behind a Bearer-auth shim (vLLM 0.8.x).

    Uses vLLM's own `build_async_engine_client` context manager wired into a
    FastAPI lifespan — the same pattern vLLM's `run_server` uses internally.
    This avoids the `app.state.log_stats` startup race.
    """
    import os as _os
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware

    from vllm.entrypoints.openai.api_server import (
        build_app,
        build_async_engine_client,
        init_app_state,
    )
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils import FlexibleArgumentParser

    # -- Parse vLLM CLI args from config ------------------------------------
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    cli_args = parser.parse_args(CFG.vllm_args())
    validate_parsed_serve_args(cli_args)

    # -- Build vLLM FastAPI app + override lifespan -------------------------
    vllm_app: FastAPI = build_app(cli_args)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async with build_async_engine_client(cli_args) as engine_client:
            model_config = await engine_client.get_model_config()
            await init_app_state(engine_client, model_config, _app.state, cli_args)
            yield

    vllm_app.router.lifespan_context = lifespan

    # -- Auth middleware ----------------------------------------------------
    expected_key = _os.environ["PARALLAX_API_KEY"]
    PUBLIC_PATHS = {"/health", "/version", "/docs", "/openapi.json", "/redoc"}

    class BearerAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.url.path in PUBLIC_PATHS:
                return await call_next(request)
            auth = request.headers.get("authorization", "")
            if not auth.lower().startswith("bearer "):
                return JSONResponse({"error": "missing bearer token"}, status_code=401)
            token = auth.split(None, 1)[1].strip()
            if token != expected_key:
                return JSONResponse({"error": "invalid api key"}, status_code=401)
            return await call_next(request)

    vllm_app.add_middleware(BearerAuth)
    vllm_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @vllm_app.get("/version")
    async def version():
        return {
            "app": APP_NAME,
            "model": CFG.served_model_name,
            "hf_model_id": CFG.hf_model_id,
            "max_model_len": CFG.max_model_len,
        }

    return vllm_app
