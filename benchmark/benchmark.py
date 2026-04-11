#!/usr/bin/env python3
import subprocess
import re
import os
import sys
import platform
import argparse
from typing import Dict, Optional, List

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def get_time_cmd() -> List[str]:
    """Return the platform-specific wrapper command to measure memory usage."""
    if platform.system() == "Darwin":
        return ["/usr/bin/time", "-l"]
    else:
        return ["/usr/bin/time", "-v"]

def can_run_time_cmd() -> bool:
    """Check if the time command exists on the OS."""
    return os.path.exists(get_time_cmd()[0])

def run_benchmark(name: str, cmd: List[str], env: dict) -> Dict[str, float]:
    print(f"\n--- Running {name} Benchmark ---")
    print(f"> Command: '{' '.join(cmd)}'")
    
    time_cmd = get_time_cmd()
    full_cmd = time_cmd + cmd if can_run_time_cmd() else cmd
    
    process = subprocess.Popen(
        full_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    # Send /exit to stdin to ensure llama-cli exits if it defaults to interactive mode
    stdout, stderr = process.communicate(input="/exit\n")
    raw_output = stdout + stderr

    # Print to user so they see progress
    for line in raw_output.split('\n'):
        if "[ Prompt:" in line or "llama_print_timings" in line or "llama_new_context_with_model" in line:
             pass # Skip spamming to console
    # Actually, we don't need to print everything. 
    
    # Strip ANSI escape codes to ensure clean regex matching
    output = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', raw_output)
    
    if process.returncode != 0:
        print(f"Warning: {name} command exited with non-zero exit status {process.returncode}")
        
    results = {}
    
    # Try classic timings first
    prompt_eval_re = re.findall(r"prompt eval time.*?=\s*[\d.]+\s*ms\s*/.*?,\s*([\d.]+)\s*tokens per second", output, re.IGNORECASE)
    if prompt_eval_re:
        results["prompt_eval_tokens_per_sec"] = float(prompt_eval_re[-1])
    else:
        # Fallback to interactive bar
        prompt_alt_re = re.findall(r"\[\s*Prompt:\s*([\d.]+)\s*t/s", output, re.IGNORECASE)
        if prompt_alt_re:
            results["prompt_eval_tokens_per_sec"] = float(prompt_alt_re[-1])

    eval_re = re.findall(r"eval time.*?=\s*[\d.]+\s*ms\s*/.*?,\s*([\d.]+)\s*tokens per second", output, re.IGNORECASE)
    if eval_re:
        results["eval_tokens_per_sec"] = float(eval_re[-1])
    else:
        eval_alt_re = re.findall(r"\|\s*Generation:\s*([\d.]+)\s*t/s", output, re.IGNORECASE)
        if eval_alt_re:
            results["eval_tokens_per_sec"] = float(eval_alt_re[-1])
        
    calc_re = re.findall(r"compute buffer total size =\s*([\d.]+)\s*MiB", output, re.IGNORECASE)
    if calc_re:
        results["llama_compute_buffer_mb"] = float(calc_re[-1])

    # Parse KV buffer size reported by llama (e.g. "Metal KV buffer size = 512.00 MiB")
    # Sum all KV buffer lines — there may be multiple (K + V split, or per-backend)
    kv_buf_re = re.findall(r"KV buffer size\s*=\s*([\d.]+)\s*MiB", output, re.IGNORECASE)
    if kv_buf_re:
        results["kv_buffer_mb"] = sum(float(v) for v in kv_buf_re)
        
    if can_run_time_cmd():
        if platform.system() == "Darwin":
            mem_re = re.search(r"^\s*(\d+)\s+maximum resident set size", output, re.MULTILINE)
            if mem_re:
                results["max_rss_mb"] = float(mem_re.group(1)) / (1024 * 1024)
        else:
            mem_re = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", output, re.IGNORECASE)
            if mem_re:
                results["max_rss_mb"] = float(mem_re.group(1)) / 1024

    return results

def print_results_table(all_results: Dict[str, Dict[str, float]], config_names: List[str]):
    print("\n" + "="*80)
    print(" " * 30 + "BENCHMARK RESULTS")
    print("="*80)
    
    header = f"{'Metric':<20} | " + " | ".join([f"{name:<15}" for name in config_names])
    print(header)
    print("-" * len(header))
    
    metrics = [
        ("prompt_eval_tokens_per_sec", "Prefill (t/s)"),
        ("eval_tokens_per_sec", "Generation (t/s)"),
        ("max_rss_mb",          "Peak RSS (MB)"),
        ("kv_buffer_mb",        "KV Cache (MiB)"),
    ]
    
    for key, display_name in metrics:
        row = f"{display_name:<20} | "
        for name in config_names:
            val = all_results[name].get(key)
            val_str = f"{val:.2f}" if val is not None else "N/A"
            row += f"{val_str:<15} | "
        print(row[:-3])
    print("="*80 + "\n")

def plot_results(all_results: Dict[str, Dict[str, float]], config_names: List[str], model_path: str):
    if not MATPLOTLIB_AVAILABLE:
        print("\nNote: 'matplotlib' is not installed. To generate a scientific graph, please run:")
        print("  pip install matplotlib numpy")
        print("and run this benchmark again.\n")
        return

    # Determine if we have KV buffer data for a dedicated panel
    has_kv_data = any(all_results[n].get('kv_buffer_mb') for n in config_names)
    ncols = 3 if has_kv_data else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols + 2, 6))
    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if has_kv_data else None

    # Color palette: grey for Baseline, green for TurboTuning, blue for CPU variants
    palette = ['#7f7f7f', '#2ca02c', '#1f77b4', '#d62728', '#9467bd']
    colors = {name: palette[i % len(palette)] for i, name in enumerate(config_names)}

    width = 0.25 if len(config_names) > 2 else 0.35

    # ── Panel 1: Speed ──────────────────────────────────────────────────────────
    labels_speed = ['Prefill (t/s)', 'Generation (t/s)']
    x_speed = np.arange(len(labels_speed))

    for i, name in enumerate(config_names):
        speed_vals = [all_results[name].get('prompt_eval_tokens_per_sec', 0),
                      all_results[name].get('eval_tokens_per_sec', 0)]
        offset = width * i - (width * len(config_names) / 2) + width / 2
        bars = ax1.bar(x_speed + offset, speed_vals, width, label=name,
                       color=colors[name], edgecolor='black')
        for bar, val in zip(bars, speed_vals):
            if val:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    ax1.set_ylabel('Tokens per Second (Higher is Better)')
    ax1.set_title('Inference Speed')
    ax1.set_xticks(x_speed)
    ax1.set_xticklabels(labels_speed)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # ── Panel 2: Peak RSS ────────────────────────────────────────────────────────
    x_mem = np.arange(1)
    for i, name in enumerate(config_names):
        mem_val = all_results[name].get('max_rss_mb') or all_results[name].get('llama_compute_buffer_mb', 0)
        offset = width * i - (width * len(config_names) / 2) + width / 2
        bar = ax2.bar(x_mem + offset, [mem_val], width, label=name,
                      color=colors[name], edgecolor='black')
        if mem_val:
            ax2.text(bar[0].get_x() + bar[0].get_width() / 2, mem_val + 10,
                     f'{mem_val:.0f}', ha='center', va='bottom', fontsize=7)

    ax2.set_ylabel('RSS Memory in MB (Lower is Better)')
    ax2.set_title('Peak System RAM (RSS)\n(includes model weights — KV is a subset)')
    ax2.set_xticks(x_mem)
    ax2.set_xticklabels(['Peak RSS (MB)'])
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # ── Panel 3: KV Cache Size (only when data available) ───────────────────────
    if ax3 is not None:
        for i, name in enumerate(config_names):
            kv_val = all_results[name].get('kv_buffer_mb', 0)
            offset = width * i - (width * len(config_names) / 2) + width / 2
            bar = ax3.bar(x_mem + offset, [kv_val], width, label=name,
                          color=colors[name], edgecolor='black')
            if kv_val:
                ax3.text(bar[0].get_x() + bar[0].get_width() / 2, kv_val + 1,
                         f'{kv_val:.1f}', ha='center', va='bottom', fontsize=7)

        ax3.set_ylabel('KV Cache Size in MiB (Lower is Better)')
        ax3.set_title('Actual KV Cache Size\n(from llama log — GPU or CPU buffer)')
        ax3.set_xticks(x_mem)
        ax3.set_xticklabels(['KV Cache (MiB)'])
        ax3.legend(fontsize=8)
        ax3.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    basename = os.path.basename(model_path).lower()
    match = re.search(r'([\d.]+b)', basename)
    model_size = match.group(1).upper() if match else "Unknown"
    title_suffix = f" ({model_size} Model)" if model_size != "Unknown" else ""

    plt.suptitle(f'TurboTuning vs Baseline Benchmarks{title_suffix}', y=1.02, fontsize=15, fontweight='bold')

    filename_size = f"_{model_size.lower()}" if model_size != "Unknown" else ""
    output_filename = f"benchmark_results{filename_size}.png"

    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"=> Scientific benchmark graph successfully saved to: {output_filename}\n")

def main():
    parser = argparse.ArgumentParser(description="TurboTuning Modular Benchmark")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    default_cli = os.path.join(project_root, "llama-cpp-turboquant", "build", "bin", "llama-cli")
    default_model = os.path.join(project_root, "models", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

    parser.add_argument("--model", type=str, help="Path to a custom GGUF model (overrides --size)")
    parser.add_argument("--size", type=str, choices=["0.5B", "8B", "32B", "70B", "104B", "all"], default="8B", help="Run benchmark on specific built-in model size or 'all'")
    parser.add_argument("--prompt", type=str, default="Write a 300 word essay about the future of artificial intelligence in space exploration. Be creative and very detailed.", help="Prompt for benchmarking")
    parser.add_argument("--n-predict", type=int, default=256, help="Number of tokens to generate")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="Number of threads")
    parser.add_argument("--ngl", type=int, default=99, help="Number of GPU layers (for offloading)")
    parser.add_argument("--llama-cli", type=str, default=default_cli, help="Path to llama-cli executable")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.llama_cli):
        print(f"Error: llama-cli not found at '{args.llama_cli}'. Please compile it first.")
        sys.exit(1)
        
    MODEL_PRESETS = {
        "0.5B": "qwen2.5-0.5b-q4_k_m.gguf",
        "8B": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "32B": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        "70B": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        "104B": "c4ai-command-r-plus-08-2024.Q2_K.gguf"
    }

    to_run = []
    if args.model:
        if not os.path.exists(args.model):
            print(f"Error: Model not found at '{args.model}'.")
            sys.exit(1)
        to_run.append(("Custom", args.model))
    elif args.size == "all":
        to_run = [(k, os.path.join(project_root, "models", v)) for k, v in MODEL_PRESETS.items()]
    else:
        to_run = [(args.size, os.path.join(project_root, "models", MODEL_PRESETS[args.size]))]

    env = os.environ.copy()
    
    for size_label, m_path in to_run:
        if not os.path.exists(m_path):
            print(f"--------------------------------------------------")
            print(f"Skipping {size_label} test: model file not found -> {os.path.basename(m_path)}")
            print(f"Run one of the demo scripts to download it first.")
            print(f"--------------------------------------------------\n")
            continue

        print(f"\n========================================================")
        print(f"    BENCHMARKING PIPELINE FOR MODEL: {size_label} ")
        print(f"========================================================")
        
        turbo_cmd = [
            args.llama_cli,
            "-m", m_path,
            "-p", args.prompt,
            "-n", str(args.n_predict),
            "-c", str(args.n_ctx),
            "-t", str(args.threads),
            "-ngl", str(args.ngl),
            "--no-display-prompt",
            "--simple-io"
        ]
        
        baseline_cli = os.environ.get("BASELINE_CLI")
        if not baseline_cli:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            baseline_cli = os.path.join(project_root, "llama-cpp-turboquant-original", "build", "bin", "llama-cli")
            if not os.path.exists(baseline_cli):
                print(f"Warning: Baseline CLI not found at '{baseline_cli}'. Falling back to default llama-cli.")
                baseline_cli = args.llama_cli

        baseline_cmd = turbo_cmd.copy()
        baseline_cmd[0] = baseline_cli

        # Build CPU-mode commands (--ngl 0): KV cache stays in RAM so RSS diff is real
        cpu_baseline_cmd = [c for c in baseline_cmd]
        cpu_baseline_cmd_idx_ngl = next(
            (i for i, v in enumerate(cpu_baseline_cmd) if v == "-ngl"), None
        )
        if cpu_baseline_cmd_idx_ngl is not None:
            cpu_baseline_cmd[cpu_baseline_cmd_idx_ngl + 1] = "0"

        cpu_turbo_cmd = [c for c in turbo_cmd]
        cpu_turbo_cmd_idx_ngl = next(
            (i for i, v in enumerate(cpu_turbo_cmd) if v == "-ngl"), None
        )
        if cpu_turbo_cmd_idx_ngl is not None:
            cpu_turbo_cmd[cpu_turbo_cmd_idx_ngl + 1] = "0"

        run_configs = {
            # GPU-offloaded runs (fast; KV on GPU so RSS diff is small)
            "Baseline (GPU)": baseline_cmd,
            "TurboTuning (GPU)": turbo_cmd + [
                "--cache-type-k", "turbo4",
                "--cache-type-v", "turbo3",
                "--turbo-async"
            ],
            # CPU-only runs (slow; KV in RAM so we measure TRUE memory savings)
            "Baseline (CPU)": cpu_baseline_cmd,
            "TurboTuning (CPU)": cpu_turbo_cmd + [
                "--cache-type-k", "turbo4",
                "--cache-type-v", "turbo3",
                "--turbo-async"
            ],
        }
        
        config_names = list(run_configs.keys())
        all_results = {}
        
        for name in config_names:
            all_results[name] = run_benchmark(name, run_configs[name], env)
            
        print_results_table(all_results, config_names)
        plot_results(all_results, config_names, m_path)

if __name__ == "__main__":
    main()
