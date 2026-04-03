"""GGUF Reader — Light, NumPy-only partial tensor loader.

Allows LLMTuning-style layer-by-layer weight streaming on Apple Silicon without
loading the entire model file into RAM. Each layer is mapped from the GGUF
on-disk and only converted to float32 NumPy arrays during the active compute.
"""

import os
import struct
from typing import Dict, Tuple

import numpy as np


def dequantize_q4_k(data: np.ndarray, shape: tuple) -> np.ndarray:
    """Dequantize GGML Q4_K blocks (144 bytes per 256 weights)."""
    n_elements = np.prod(shape)
    n_blocks = n_elements // 256
    if n_blocks == 0: return np.zeros(shape, dtype=np.float16)

    # Reshape into blocks
    blocks = data.reshape(n_blocks, 144)
    
    # Super-block scale and min (fp16)
    d = blocks[:, 0:2].view(np.float16).astype(np.float32)
    dmin = blocks[:, 2:4].view(np.float16).astype(np.float32)
    
    # Scales (12 bytes for 8 groups)
    sc = blocks[:, 4:16]
    # Unpack 6-bit scales and mins
    scales = np.zeros((n_blocks, 8), dtype=np.float32)
    mins = np.zeros((n_blocks, 8), dtype=np.float32)
    
    # Bytes 0-3: low 4 bits of scales 0-3 and mins 0-3
    scales[:, 0] = (sc[:, 0] & 0x0F) | ((sc[:, 4] & 0x03) << 4)
    scales[:, 1] = (sc[:, 1] & 0x0F) | ((sc[:, 4] & 0x0C) << 2)
    scales[:, 2] = (sc[:, 2] & 0x0F) | ((sc[:, 4] & 0x30))
    scales[:, 3] = (sc[:, 3] & 0x0F) | ((sc[:, 4] & 0xC0) >> 2)
    
    mins[:, 0] = (sc[:, 0] >> 4) | ((sc[:, 5] & 0x03) << 4)
    mins[:, 1] = (sc[:, 1] >> 4) | ((sc[:, 5] & 0x0C) << 2)
    mins[:, 2] = (sc[:, 2] >> 4) | ((sc[:, 5] & 0x30))
    mins[:, 3] = (sc[:, 3] >> 4) | ((sc[:, 5] & 0xC0) >> 2)
    
    # Bytes 6-11: low 4 bits of scales 4-7 and mins 4-7
    scales[:, 4] = (sc[:, 6] & 0x0F) | ((sc[:, 10] & 0x03) << 4)
    scales[:, 5] = (sc[:, 7] & 0x0F) | ((sc[:, 10] & 0x0C) << 2)
    scales[:, 6] = (sc[:, 8] & 0x0F) | ((sc[:, 10] & 0x30))
    scales[:, 7] = (sc[:, 9] & 0x0F) | ((sc[:, 10] & 0xC0) >> 2)
    
    mins[:, 4] = (sc[:, 6] >> 4) | ((sc[:, 11] & 0x03) << 4)
    mins[:, 5] = (sc[:, 7] >> 4) | ((sc[:, 11] & 0x0C) << 2)
    mins[:, 6] = (sc[:, 8] >> 4) | ((sc[:, 11] & 0x30))
    mins[:, 7] = (sc[:, 9] >> 4) | ((sc[:, 11] & 0xC0) >> 2)

    # 4-bit quants (128 bytes)
    # Dequantize
    out = np.zeros((n_blocks, 256), dtype=np.float16)
    for i in range(8): # 8 groups of 32
        group_scales = d * scales[:, i:i+1]
        group_mins = dmin * mins[:, i:i+1]
        
        off = i * 16
        group_data = blocks[:, 16+off : 16+off+16]
        v1 = group_data & 0xF
        v2 = group_data >> 4
        
        out[:, i*32 : i*32+16] = (v1 * group_scales - group_mins).astype(np.float16)
        out[:, i*32+16 : i*32+32] = (v2 * group_scales - group_mins).astype(np.float16)
        
    return out.reshape(shape)


def dequantize_q6_k(data: np.ndarray, shape: tuple) -> np.ndarray:
    """Dequantize GGML Q6_K blocks (210 bytes per 256 weights)."""
    n_elements = np.prod(shape)
    n_blocks = n_elements // 256
    if n_blocks == 0: return np.zeros(shape, dtype=np.float16)

    blocks = data.reshape(n_blocks, 210)
    
    ql = blocks[:, 0:128]      # 128 bytes
    sc = blocks[:, 192:208].astype(np.int8)  # 16 bytes
    d  = blocks[:, 208:210].view(np.float16).astype(np.float32)
    
    out = np.zeros((n_blocks, 256), dtype=np.float16)
    for i in range(16): # 16 scales for 16 blocks of 16
        scale = d * sc[:, i:i+1]
        off = i * 8
        group = ql[:, off : off+8]
        v1 = group & 0xF
        v2 = group >> 4
        out[:, i*16 : i*16+8] = ((v1.astype(np.float32) - 32) * scale).astype(np.float16)
        out[:, i*16+8 : i*16+16] = ((v2.astype(np.float32) - 32) * scale).astype(np.float16)
        
    return out.reshape(shape)


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
            key = self._read_str()
            value_type = struct.unpack("<I", self.fd.read(4))[0]
            self.metadata[key] = self._read_value(value_type)

        # 2. Parse tensor info table
        self.tensors: Dict[str, dict] = {}
        for _ in range(self.count_tensors):
            name, info = self._read_tensor_info()
            self.tensors[name] = info

        # 3. Find data offset (padding)
        self.data_start = (self.fd.tell() + 31) & ~31

        # Use memmap for zero-copy access
        self.map = np.memmap(self.path, mode="r")

    def _read_str(self) -> str:
        length = struct.unpack("<Q", self.fd.read(8))[0]
        return self.fd.read(length).decode("utf-8")

    def _read_value(self, value_type: int) -> any:
        if value_type in (0, 7): return struct.unpack("<B", self.fd.read(1))[0]
        elif value_type == 1: return struct.unpack("<b", self.fd.read(1))[0]
        elif value_type == 2: return struct.unpack("<H", self.fd.read(2))[0]
        elif value_type == 3: return struct.unpack("<h", self.fd.read(2))[0]
        elif value_type == 4: return struct.unpack("<I", self.fd.read(4))[0]
        elif value_type == 5: return struct.unpack("<i", self.fd.read(4))[0]
        elif value_type == 6: return struct.unpack("<f", self.fd.read(4))[0]
        elif value_type == 10: return struct.unpack("<Q", self.fd.read(8))[0]
        elif value_type == 11: return struct.unpack("<q", self.fd.read(8))[0]
        elif value_type == 12: return struct.unpack("<d", self.fd.read(8))[0]
        elif value_type == 8: return self._read_str()
        elif value_type == 9: # Array
            sub_type = struct.unpack("<I", self.fd.read(4))[0]
            count = struct.unpack("<Q", self.fd.read(8))[0]
            vals = []
            for _ in range(count):
                vals.append(self._read_value(sub_type))
            return vals
        return None

    def _read_tensor_info(self) -> Tuple[str, dict]:
        name = self._read_str()
        n_dims = struct.unpack("<I", self.fd.read(4))[0]
        dims = []
        for _ in range(n_dims):
            dims.append(struct.unpack("<Q", self.fd.read(8))[0])
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
                 short_name = name.split(".")[-2] + "_weight"
                 if "attn_q" in name: short_name = "q_weight"
                 elif "attn_k" in name: short_name = "k_weight"
                 elif "attn_v" in name: short_name = "v_weight"
                 elif "attn_output" in name: short_name = "o_weight"

                 start = self.data_start + info["offset"]
                 
                 if info["type"] in (0, 1): # F32 or F16
                      dtype = np.float32 if info["type"] == 0 else np.float16
                      size = np.prod(info["shape"])
                      data = self.map[start : start + (size * np.dtype(dtype).itemsize)]
                      weights[short_name] = np.frombuffer(data, dtype=dtype).reshape(info["shape"]).astype(np.float16)
                 elif info["type"] == 12: # Q4_K
                      size_bytes = (np.prod(info["shape"]) // 256) * 144
                      data = self.map[start : start + size_bytes]
                      weights[short_name] = dequantize_q4_k(data, info["shape"])
                 elif info["type"] == 14: # Q6_K
                      size_bytes = (np.prod(info["shape"]) // 256) * 210
                      data = self.map[start : start + size_bytes]
                      weights[short_name] = dequantize_q6_k(data, info["shape"])
                 else:
                      weights[short_name] = np.zeros(info["shape"], dtype=np.float16)
        return weights

    def get_weight(self, name: str) -> np.ndarray:
        """Fetch any named tensor from the GGUF and dequantize it to FP16."""
        if name not in self.tensors:
             return None
        
        info = self.tensors[name]
        start = self.data_start + info["offset"]
        
        if info["type"] in (0, 1): # F32 or F16
             dtype = np.float32 if info["type"] == 0 else np.float16
             size = np.prod(info["shape"])
             data = self.map[start : start + (size * np.dtype(dtype).itemsize)]
             return np.frombuffer(data, dtype=dtype).reshape(info["shape"]).astype(np.float16)
        elif info["type"] == 12: # Q4_K
             size_bytes = (np.prod(info["shape"]) // 256) * 144
             data = self.map[start : start + size_bytes]
             return dequantize_q4_k(data, info["shape"])
        elif info["type"] == 14: # Q6_K
             size_bytes = (np.prod(info["shape"]) // 256) * 210
             data = self.map[start : start + size_bytes]
             return dequantize_q6_k(data, info["shape"])
        
        return np.zeros(info["shape"], dtype=np.float16)

    def close(self):
        self.fd.close()
