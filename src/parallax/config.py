"""YAML config loader for Parallax model deployments."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    # identity
    name: str
    hf_model_id: str
    served_model_name: str

    # hardware
    gpu: str = "A10G"
    gpu_count: int = 1

    # vLLM engine
    dtype: str = "bfloat16"
    max_model_len: int = 2048
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.92
    enable_prefix_caching: bool = True
    guided_decoding_backend: str = "xgrammar"

    # Modal runtime
    min_containers: int = 0
    scaledown_window: int = 300
    allow_concurrent_inputs: int = 256
    timeout: int = 600

    # Cold-start acceleration (opt-in)
    enable_memory_snapshot: bool = False   # snapshot CPU state after imports
    enable_gpu_snapshot: bool = False      # snapshot post-engine-init (experimental)

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        with open(p) as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_env(cls, default: str = "config/qwen2.5-7b.yaml") -> "ModelConfig":
        """Load config path from MODEL_CONFIG env var, fall back to default.

        Used by the Modal app when invoked directly (no CLI). For CLI-driven
        deploys, the CLI sets MODEL_CONFIG before running `modal deploy`.
        """
        path = os.environ.get("MODEL_CONFIG", default)
        p = Path(path).expanduser()
        if not p.is_absolute():
            # Try several anchors: CWD, then package root
            candidates = [
                Path.cwd() / path,
                Path(__file__).resolve().parent.parent.parent / path,
            ]
            p = next((c for c in candidates if c.exists()), candidates[0])
        cfg = cls.load(p)
        # Env overrides — set by the CLI (--gpu, --gpu-count) or by hand
        if (g := os.environ.get("PARALLAX_GPU")):
            cfg.gpu = g
        if (gc := os.environ.get("PARALLAX_GPU_COUNT")):
            try:
                cfg.gpu_count = int(gc)
            except ValueError as e:
                raise ValueError(f"PARALLAX_GPU_COUNT must be int, got {gc!r}") from e
        return cfg

    def vllm_args(self) -> list[str]:
        """Convert config into vLLM CLI/engine-arg list."""
        args = [
            "--model", self.hf_model_id,
            "--served-model-name", self.served_model_name,
            "--dtype", self.dtype,
            "--max-model-len", str(self.max_model_len),
            "--max-num-seqs", str(self.max_num_seqs),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--guided-decoding-backend", self.guided_decoding_backend,
        ]
        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")
        return args
