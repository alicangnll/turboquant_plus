import json
import argparse
import sys

def get_optimal_config(model_choice, mem_choice):
    # Default values - LLMTuning Universal Minimization
    config = {
        "ctx_len": 2048,
        "cache_type_k": "turbo4", # Standardize on high-compression
        "cache_type_v": "turbo4",
        "num_layers": 32,
        "num_heads": 32,
        "head_dim": 128,
        "chat_template": "",
        "extra_args": ""
    }

    # Model specific architectures (LLMTuning Intelligence)
    if model_choice in ["1", "8B", "8b"]: # Llama 3.1 8B
        config.update({
            "num_layers": 32,
            "num_heads": 32,
            "head_dim": 128,
            "ctx_len": 4096 if mem_choice == "1" else (2048 if mem_choice == "2" else 1024),
            "cache_type_k": "turbo4", # Minimized even on performance mode for 8B
            "cache_type_v": "turbo4"
        })

    elif model_choice in ["2", "32B", "32b"]: # Qwen 2.5 32B
        config.update({
            "num_layers": 64,
            "num_heads": 40,
            "head_dim": 128,
            "ctx_len": 1024 if mem_choice == "1" else (512 if mem_choice == "2" else 256),
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo2" if mem_choice == "3" else "turbo4", 
            "chat_template": "qwen2"
        })

    elif model_choice in ["3", "100B", "100b"]: # Command R+ 104B
        config.update({
            "num_layers": 96,
            "num_heads": 128,
            "head_dim": 128,
            "ctx_len": 1024 if mem_choice == "1" else (512 if mem_choice == "2" else 256),
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo2" if mem_choice == "3" else "turbo4",
            "chat_template": "command-r"
        })

    elif model_choice in ["6", "20B", "20b"]: # GPT 20B OSS (MoE)
        config.update({
            "num_layers": 24,
            "num_heads": 8,
            "head_dim": 64,
            "ctx_len": 2048 if mem_choice == "1" else 1024,
            "cache_type_k": "turbo4", 
            "cache_type_v": "turbo4",
            "chat_template": "none"
        })

    elif model_choice in ["7", "31B", "31b"]: # Gemma 4 31B
        config.update({
            "num_layers": 48,
            "num_heads": 32,
            "head_dim": 128,
            "ctx_len": 4096 if mem_choice == "1" else (2048 if mem_choice == "2" else 512),
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo2" if mem_choice == "3" else "turbo4",
            "chat_template": "gemma",
            "extra_args": "-fa on"
        })

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-choice", required=True)
    parser.add_argument("--mem-choice", required=True)
    args = parser.parse_args()

    config = get_optimal_config(args.model_choice, args.mem_choice)
    print(json.dumps(config))
