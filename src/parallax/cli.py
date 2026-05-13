"""Parallax CLI — thin wrapper around `modal deploy/serve` that:

1. Takes a config YAML path
2. Optionally overrides the Modal app name
3. Sets MODEL_CONFIG / PARALLAX_APP_NAME env vars before invoking Modal

Usage:
    parallax deploy config/qwen2.5-7b.yaml
    parallax deploy config/qwen2.5-7b.yaml --name my-custom-app
    parallax serve config/qwen2.5-7b.yaml --name dev-test
    parallax stop my-custom-app
    parallax info config/qwen2.5-7b.yaml
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

import click

# Resolve path to src/parallax/app.py regardless of where the CLI is invoked from
APP_MODULE_PATH = (Path(__file__).resolve().parent / "app.py").as_posix()


def _validate_config(config_path: str) -> Path:
    p = Path(config_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.exists():
        raise click.ClickException(f"config not found: {p}")
    if p.suffix not in (".yaml", ".yml"):
        raise click.ClickException(f"expected .yaml/.yml, got: {p.suffix}")
    return p


def _run_modal(modal_subcmd: str, config: Path, name: str | None) -> int:
    env = os.environ.copy()
    env["MODEL_CONFIG"] = str(config)
    if name:
        env["PARALLAX_APP_NAME"] = name
    cmd = ["modal", modal_subcmd, APP_MODULE_PATH]
    click.echo(f"$ MODEL_CONFIG={config.name}"
               f"{' PARALLAX_APP_NAME=' + name if name else ''}"
               f" {' '.join(cmd)}", err=True)
    return subprocess.call(cmd, env=env)


@click.group()
@click.version_option(package_name="parallax")
def cli() -> None:
    """Parallax — OpenAI-compatible vLLM endpoints on Modal."""


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--name", "-n", help="Override Modal app name (default: parallax-<config.name>)")
def deploy(config: str, name: str | None) -> None:
    """Deploy a model defined by CONFIG (path to YAML) to Modal."""
    cfg = _validate_config(config)
    sys.exit(_run_modal("deploy", cfg, name))


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--name", "-n", help="Override Modal app name")
def serve(config: str, name: str | None) -> None:
    """Hot-reload dev server (ephemeral URL, dies on Ctrl+C)."""
    cfg = _validate_config(config)
    sys.exit(_run_modal("serve", cfg, name))


@cli.command()
@click.argument("app_name")
def stop(app_name: str) -> None:
    """Stop a deployed app by name."""
    sys.exit(subprocess.call(["modal", "app", "stop", app_name]))


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
def info(config: str) -> None:
    """Print the resolved config (validates YAML parsing)."""
    from parallax.config import ModelConfig

    cfg_path = _validate_config(config)
    cfg = ModelConfig.load(cfg_path)
    click.echo(f"Config: {cfg_path}")
    click.echo(f"  app name:        parallax-{cfg.name}")
    click.echo(f"  model:           {cfg.hf_model_id}")
    click.echo(f"  served as:       {cfg.served_model_name}")
    click.echo(f"  gpu:             {cfg.gpu}" + (f" x{cfg.gpu_count}" if cfg.gpu_count > 1 else ""))
    click.echo(f"  dtype:           {cfg.dtype}")
    click.echo(f"  max_model_len:   {cfg.max_model_len}")
    click.echo(f"  max_num_seqs:    {cfg.max_num_seqs}")
    click.echo(f"  prefix caching:  {cfg.enable_prefix_caching}")
    click.echo(f"  memory snapshot: {cfg.enable_memory_snapshot}")
    click.echo(f"  gpu snapshot:    {cfg.enable_gpu_snapshot}")
    click.echo(f"  scaledown:       {cfg.scaledown_window}s")


if __name__ == "__main__":
    cli()
