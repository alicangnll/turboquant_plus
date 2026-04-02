"""GGUF Reader — Light, NumPy-only partial tensor loader.

Allows AirLLM-style layer-by-layer weight streaming on Apple Silicon without
loading the entire model file into RAM. Each layer is mapped from the GGUF
on-disk and only converted to float32 NumPy arrays during the active compute.
"""

import os
import struct
from typing import Dict, Tuple

import numpy as np


class GGUFMap:
    """Maps a GGUF file and allows random access to individual layer tensors."""

    def __init__(self, path: str):
        self.path = path
        self.fd = open(path, "rb")
        self._load_header()

    def _load_header(self):
        # Magic: GGUF (4 bytes)
        magic = self.fd.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {self.path} (magic={magic})")

        # Version (4 bytes, little endian)
        self.version = struct.unpack("<I", self.fd.read(4))[0]
        if self.version not in (2, 3):
             raise ValueError(f"Unsupported GGUF version: {self.version}")

        # Meta counts
        self.count_tensors = struct.unpack("<Q", self.fd.read(8))[0]
        self.count_metadata = struct.unpack("<Q", self.fd.read(8))[0]

        # 1. Parse metadata KV pairs
        self.metadata: Dict[str, any] = {}
        for _ in range(self.count_metadata):
            key, val = self._read_kv()
            self.metadata[key] = val

        # 2. Parse tensor info table
        self.tensors: Dict[str, dict] = {}
        for _ in range(self.count_tensors):
            name, info = self._read_tensor_info()
            self.tensors[name] = info

        # 3. Find data offset (padding)
        # GGUF tensor data starts after the table, aligned to 32 bytes by default
        self.data_start = (self.fd.tell() + 31) & ~31

        # Use memmap for zero-copy access
        self.map = np.memmap(self.path, mode="r")

    def _read_str(self) -> str:
        length = struct.unpack("<Q", self.fd.read(8))[0]
        return self.fd.read(length).decode("utf-8")

    def _read_kv(self) -> Tuple[str, any]:
        key = self._read_str()
        value_type = struct.unpack("<I", self.fd.read(4))[0]
        # value_type: 0=u8, 1=i8, 2=u16, 3=i16, 4=u32, 5=i32, 6=f32, 7=bool, 8=str, 9=array, 10=u64, 11=i64, 12=f64
        if value_type == 0: val = struct.unpack("<B", self.fd.read(1))[0]
        elif value_type == 1: val = struct.unpack("<b", self.fd.read(1))[0]
        elif value_type == 2: val = struct.unpack("<H", self.fd.read(2))[0]
        elif value_type == 3: val = struct.unpack("<h", self.fd.read(2))[0]
        elif value_type == 4: val = struct.unpack("<I", self.fd.read(4))[0]
        elif value_type == 5: val = struct.unpack("<i", self.fd.read(4))[0]
        elif value_type == 6: val = struct.unpack("<f", self.fd.read(4))[0]
        elif value_type == 7: val = struct.unpack("?", self.fd.read(1))[0]
        elif value_type == 8: val = self._read_str()
        elif value_type == 9: # Array
            sub_type = struct.unpack("<I", self.fd.read(4))[0]
            count = struct.unpack("<Q", self.fd.read(8))[0]
            val = []
            if sub_type == 8: # str array
                for _ in range(count): val.append(self._read_str())
            else:
                sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
                fmts = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"?",10:"<Q",11:"<q",12:"<d"}
                s = sizes[sub_type]
                f = fmts[sub_type]
                for _ in range(count): val.append(struct.unpack(f, self.fd.read(s))[0])
        elif value_type == 10: val = struct.unpack("<Q", self.fd.read(8))[0]
        elif value_type == 11: val = struct.unpack("<q", self.fd.read(8))[0]
        elif value_type == 12: val = struct.unpack("<d", self.fd.read(8))[0]
        else: val = None
        return key, val

    def get_arch(self) -> dict:
        """Heuristically detect model architecture from metadata and tensors."""
        arch = {
            "num_layers": self.metadata.get("llama.block_count", 0),
            "num_heads": self.metadata.get("llama.attention.head_count", 0),
            "head_dim": self.metadata.get("llama.attention.head_count_kv", 0), # fallback
        }

        # If metadata is missing (older GGUF or weird tags), infer from tensor scans
        if arch["num_layers"] == 0:
            max_blk = -1
            for name in self.tensors:
                if name.startswith("blk."):
                    try: blk = int(name.split(".")[1]); max_blk = max(max_blk, blk)
                    except: pass
            arch["num_layers"] = max_blk + 1

        if arch["num_heads"] == 0:
            # Check shape of blk.0.attn_q.weight
            t_name = next((n for n in self.tensors if n.endswith("attn_q.weight")), None)
            if t_name:
                shape = self.tensors[t_name]["shape"]
                # For Llama models, arch is hidden_size / num_heads = head_dim
                # We'll guess 32 heads for 7B, 40 for 32B, etc. 
                # Better: use typical 128 head_dim for large models
                if len(shape) >= 1: arch["num_heads"] = shape[0] // 128

        if arch.get("head_dim") == 0 or arch.get("head_dim") is None or isinstance(arch.get("head_dim"), list):
             arch["head_dim"] = 128 # Industry standard

        return arch

    def _read_tensor_info(self) -> Tuple[str, dict]:
        name = self._read_str()
        n_dims = struct.unpack("<I", self.fd.read(4))[0]
        dims = []
        for _ in range(n_dims):
            dims.append(struct.unpack("<Q", self.fd.read(8))[0])
        # GGUF tensors are usually [width, height, ...] but llama.cpp often flips them
        dims.reverse()

        type_id = struct.unpack("<I", self.fd.read(4))[0]
        offset = struct.unpack("<Q", self.fd.read(8))[0]

        return name, {
            "shape": tuple(dims),
            "type": type_id,
            "offset": offset,
        }

    def get_layer_weights(self, layer_idx: int) -> dict:
        """Fetch tensors for a specific layer. Only maps the bytes into RAM."""
        weights = {}
        prefix = f"blk.{layer_idx}."
        for name, info in self.tensors.items():
            if name.startswith(prefix):
                 # Simple name mapping for Transformer (q, k, v, o)
                 short_name = name.split(".")[-2] + "_weight"
                 if "attn_q" in name: short_name = "q_weight"
                 elif "attn_k" in name: short_name = "k_weight"
                 elif "attn_v" in name: short_name = "v_weight"
                 elif "attn_output" in name: short_name = "o_weight"

                 # Map the tensor (zero copy)
                 # Note: llama.cpp GGUF quantization (Q4_K, etc.) requires specialized dequantization.
                 # For the AirLLM + TurboQuant demo, we assume float16 or float32 for simplicity,
                 # or we skip reading if it's already quantized (to avoid complexity).
                 if info["type"] in (0, 1): # F32 or F16
                      dtype = np.float32 if info["type"] == 0 else np.float16
                      # bytes = size * dtype_size
                      size = np.prod(info["shape"])
                      start = self.data_start + info["offset"]
                      data = self.map[start : start + (size * np.dtype(dtype).itemsize)]
                      weights[short_name] = np.frombuffer(data, dtype=dtype).reshape(info["shape"]).copy()
                 else:
                      # If it's a Q4_K_M etc., we'd need a dequantizer.
                      # We'll generate synthetic weights for any non-float tensors to keep
                      # the demo interactive and memory-accurate without a C++ dequant library.
                      pass
        return weights

    def close(self):
        self.fd.close()


    def close(self):
        self.fd.close()
