from __future__ import annotations

"""CLI config export helpers for LLMTuning + TurboQuant + llama-cli.

This module converts the high-level LLMTuning/TurboQuant policy and model
metadata into a compact JSON description that shell scripts can consume to
launch `llama-cli` with matching KV cache and context settings.

Usage (Python):
    from turboquant.cli_config_export import build_cli_config, print_cli_config

    cfg = build_cli_config(
        model_size_b=32,
        model_choice="32B",
        mem_mode="balanced",  # "performance" | "balanced" | "ultra-eco"
    )
    print(cfg)

Usage (shell):
    # Example:
    #   python -m turboquant.cli_config_export --model-choice 2 --mem-choice 2 \\
    #       > tq_cli_config.json
    #   ./run_turboquant_demo_macos.sh --config tq_cli_config.json
"""

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Literal, Optional

from .llmtuning_bridge import CachePolicy, policy_for_model_size


MemMode = Literal["performance", "balanced", "ultra-eco"]


@dataclass
class CLIConfig:
    """Configuration summary for launching llama-cli.

    Fields are intentionally minimal and stable so that shell scripts can parse
    them using `jq`, `python -c`, or simple text tools without importing
    project-specific Python.
    """

    # High-level identifiers
    model_choice: str  # e.g. "1", "2", "3", "5", "6" or labels like "8B"
    model_size_b: float
    mem_mode: MemMode

    # Architecture parameters
    num_layers: int
    num_heads: int
    head_dim: int

    # KV / context policy (mirrors CachePolicy)
    cache_type_k: str  # "q8_0", "turbo2", "turbo3", "turbo4"
    cache_type_v: str
    ctx_len: int
    boundary_n_layers: int

    # NGL recommendation (per-device safe GPU layer count)
    # This is an *upper bound* that shell scripts can intersect with the
    # on-device Metal budget calculation from run_turboquant_demo_macos.sh.
    ngl_hint: int


def _arch_for_model_choice(model_choice: str) -> tuple[float, int, int, int]:
    """Return (model_size_b, num_layers, num_heads, head_dim) for a choice.

    The mapping mirrors both `StreamedInferenceManager.for_model_size` and the
    shell script model table so that LLMTuning and llama-cli see consistent
    architectures.
    """
    choice = str(model_choice).lower()

    # Defaults (Qwen 0.5B demo)
    size_b = 0.5
    num_layers = 24
    num_heads = 8
    head_dim = 64

    if choice in {"1", "8b", "8"}:
        size_b, num_layers, num_heads, head_dim = 8.0, 32, 32, 128
    elif choice in {"2", "32b", "32"}:
        size_b, num_layers, num_heads, head_dim = 32.0, 64, 40, 128
    elif choice in {"3", "100b", "100"}:
        size_b, num_layers, num_heads, head_dim = 104.0, 96, 128, 128
    elif choice in {"5", "500b", "405b", "405"}:
        size_b, num_layers, num_heads, head_dim = 405.0, 126, 128, 128
    elif choice in {"6", "20b", "20"}:
        size_b, num_layers, num_heads, head_dim = 20.0, 24, 8, 64

    return size_b, num_layers, num_heads, head_dim


def _mem_mode_from_choice(mem_choice: int) -> MemMode:
    if mem_choice == 1:
        return "performance"
    if mem_choice == 3:
        return "ultra-eco"
    return "balanced"


def _ctx_for(model_size_b: float, policy: CachePolicy, mem_mode: MemMode) -> int:
    """Derive a conservative context length for llama-cli from policy + mode."""
    # Start from CachePolicy.max_context (LLMTuning side), then tighten a bit
    # for more aggressive modes to leave headroom for Metal graph buffers.
    base = policy.max_context

    if model_size_b <= 8:
        # Small models: let llama-cli use up to 4k safely
        return min(base, 4096)

    if mem_mode == "performance":
        return base
    if mem_mode == "ultra-eco":
        return max(base // 4, 256)
    # balanced
    return max(base // 2, 512)


def _cache_types_from_policy(policy: CachePolicy, model_size_b: float) -> tuple[str, str]:
    """Map CachePolicy bits to llama-cli cache-type names."""
    # For K/V: 2 → "turbo2", 3 → "turbo3", 4 → "turbo4"
    def bits_to_name(bits: int) -> str:
        if bits == 2:
            return "turbo2"
        if bits == 3:
            return "turbo3"
        if bits >= 4:
            return "turbo4"
        return "q8_0"

    k_name = bits_to_name(policy.k_bits)
    v_name = bits_to_name(policy.v_bits)

    # For very small models (<1B) it is often preferable to stay in q8_0.
    if model_size_b < 1.0:
        k_name = v_name = "q8_0"

    return k_name, v_name


def build_cli_config(
    model_choice: str,
    mem_choice: int,
    *,
    model_size_b_override: Optional[float] = None,
) -> CLIConfig:
    """Construct a CLIConfig for a given model + memory choice.

    Args:
        model_choice: Same value as used in run_turboquant_demo_macos.sh (1–6 or label).
        mem_choice: 1=performance, 2=balanced, 3=ultra-eco.
        model_size_b_override: Optional manual model size in billions.
    """
    size_b, num_layers, num_heads, head_dim = _arch_for_model_choice(model_choice)
    if model_size_b_override is not None:
        size_b = model_size_b_override

    mem_mode = _mem_mode_from_choice(mem_choice)
    policy = policy_for_model_size(size_b)

    ctx_len = _ctx_for(size_b, policy, mem_mode)
    cache_type_k, cache_type_v = _cache_types_from_policy(policy, size_b)

    # NGL hint: rough upper bound assuming all layers can move to GPU.
    # Shell still runs its own Metal-based `calculate_ngl`, but this hint can
    # be used in the future to clamp that result per-policy.
    if size_b <= 8:
        ngl_hint = num_layers
    elif size_b <= 32:
        ngl_hint = int(num_layers * 0.75)
    elif size_b <= 104:
        ngl_hint = int(num_layers * 0.6)
    else:
        ngl_hint = int(num_layers * 0.5)

    return CLIConfig(
        model_choice=str(model_choice),
        model_size_b=size_b,
        mem_mode=mem_mode,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        ctx_len=ctx_len,
        boundary_n_layers=policy.boundary_n_layers,
        ngl_hint=ngl_hint,
    )


def print_cli_config(config: CLIConfig) -> None:
    """Print CLI config as a single-line JSON object to stdout."""
    print(json.dumps(asdict(config), separators=(",", ":"), sort_keys=True))


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export TurboQuant + LLMTuning config for llama-cli.")
    p.add_argument(
        "--model-choice",
        type=str,
        required=True,
        help="Model selection (matches run_turboquant_demo_macos.sh: 1..6, 8B, 32B, etc.)",
    )
    p.add_argument(
        "--mem-choice",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="Memory optimisation mode: 1=performance, 2=balanced, 3=ultra-eco.",
    )
    p.add_argument(
        "--model-size-b",
        type=float,
        default=None,
        help="Optional explicit model size in billions (overrides heuristic).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = build_cli_config(
        model_choice=args.model_choice,
        mem_choice=args.mem_choice,
        model_size_b_override=args.model_size_b,
    )
    print_cli_config(cfg)


if __name__ == "__main__":
    main()

