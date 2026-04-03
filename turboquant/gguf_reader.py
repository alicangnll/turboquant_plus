"""GGUF Reader — Light, NumPy-only partial tensor loader.

Allows LLMTuning-style layer-by-layer weight streaming on Apple Silicon without
loading the entire model file into RAM. Each layer is mapped from the GGUF
on-disk and only converted to float32 NumPy arrays during the active compute.
"""

import os
import struct
from typing import Dict, Tuple

import numpy as np

QK_K = 256


def _get_scale_min_k4(j: int, q: np.ndarray) -> tuple[float, float]:
    """Match ggml get_scale_min_k4(j, scales, &sc, &m); q is 12-byte scales row."""
    if j < 4:
        return float(q[j] & 63), float(q[j + 4] & 63)
    return (
        float((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)),
        float((q[j + 4] >> 4) | ((q[j] >> 6) << 4)),
    )


def dequantize_q4_k(data: np.ndarray, shape: tuple) -> np.ndarray:
    """Dequantize GGML Q4_K blocks (144 bytes per 256 weights).

    Must match ggml ``dequantize_row_q4_K`` / ``get_scale_min_k4`` (not legacy layouts).
    """
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // QK_K
    if n_blocks == 0:
        return np.zeros(shape, dtype=np.float16)

    blocks = data.reshape(n_blocks, 144)
    d_all = blocks[:, 0:2].view(np.float16).astype(np.float32).reshape(-1)
    min_all = blocks[:, 2:4].view(np.float16).astype(np.float32).reshape(-1)
    scales12 = blocks[:, 4:16]
    qs_all = blocks[:, 16:144]

    out = np.zeros((n_blocks, QK_K), dtype=np.float32)
    for bi in range(n_blocks):
        d = float(d_all[bi])
        min_v = float(min_all[bi])
        sc = scales12[bi]
        qbuf = qs_all[bi]
        y = out[bi]
        yo = 0
        is_ = 0
        q = qbuf
        for _ in range(0, QK_K, 64):
            s0, m0 = _get_scale_min_k4(is_ + 0, sc)
            s1, m1 = _get_scale_min_k4(is_ + 1, sc)
            d1 = d * s0
            m1f = min_v * m0
            d2 = d * s1
            m2f = min_v * m1
            for l in range(32):
                y[yo + l] = d1 * (int(q[l]) & 0xF) - m1f
            for l in range(32):
                y[yo + 32 + l] = d2 * (int(q[l]) >> 4) - m2f
            yo += 64
            q = q[32:]
            is_ += 2

    return out.reshape(shape).astype(np.float16)


def dequantize_q6_k(data: np.ndarray, shape: tuple) -> np.ndarray:
    """Dequantize GGML Q6_K blocks (210 bytes per 256 weights).

    Layout: ql[128], qh[64], scales[16], d (fp16). Matches ``dequantize_row_q6_K``.
    """
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // QK_K
    if n_blocks == 0:
        return np.zeros(shape, dtype=np.float16)

    blocks = data.reshape(n_blocks, 210)
    ql_all = blocks[:, 0:128].astype(np.uint8)
    qh_all = blocks[:, 128:192].astype(np.uint8)
    sc_all = blocks[:, 192:208].astype(np.int8)
    d_all = blocks[:, 208:210].view(np.float16).astype(np.float32).reshape(-1)

    out = np.zeros((n_blocks, QK_K), dtype=np.float32)
    for bi in range(n_blocks):
        d = float(d_all[bi])
        ql = ql_all[bi]
        qh = qh_all[bi]
        sc = sc_all[bi]
        y = out[bi]
        y_off = 0
        ql_off = 0
        qh_off = 0
        sc_off = 0
        for _ in range(0, QK_K, 128):
            for l in range(32):
                is_l = l // 16
                q1 = int((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) - 32
                q2 = int((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) - 32
                q3 = int((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) - 32
                q4 = int((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) - 32
                y[y_off + l + 0] = d * int(sc[sc_off + is_l + 0]) * q1
                y[y_off + l + 32] = d * int(sc[sc_off + is_l + 2]) * q2
                y[y_off + l + 64] = d * int(sc[sc_off + is_l + 4]) * q3
                y[y_off + l + 96] = d * int(sc[sc_off + is_l + 6]) * q4
            y_off += 128
            ql_off += 64
            qh_off += 32
            sc_off += 8

    return out.reshape(shape).astype(np.float16)


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
                 elif "ffn_gate" in name: short_name = "gate_weight"
                 elif "ffn_up" in name: short_name = "up_weight"
                 elif "ffn_down" in name: short_name = "down_weight"

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
