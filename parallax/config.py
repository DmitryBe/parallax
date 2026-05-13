"""YAML config loader for Parallax model deployments."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
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
    gpu: str = "L4"
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

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_env(cls, default: str = "config/qwen2.5-7b.yaml") -> "ModelConfig":
        """Load config path from MODEL_CONFIG env var, fall back to default."""
        path = os.environ.get("MODEL_CONFIG", default)
        # resolve relative to repo root if needed
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / p
        return cls.load(p)

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
