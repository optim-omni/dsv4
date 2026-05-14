#!/usr/bin/env python3
"""Parameter-count calculator for a DeepSeek-V4-shaped 35A3 nano config.

The script intentionally avoids importing heavyweight model code. It mirrors
the modules visible in the upstream modeling files and reports logical weight
counts, which is the useful architecture-level count before quantization scale
tensors or runtime KV-cache buffers.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


QWEN_CONFIG_URL = "https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json"
QWEN_MODELING_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers/main/"
    "src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py"
)
QWEN_CONFIGURATION_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers/main/"
    "src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py"
)
DSV4_CONFIG_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json"
DSV4_INFERENCE_CONFIG_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/inference/config.json"
DSV4_MODELING_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/inference/model.py"


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def b(n: int) -> float:
    return n / 1_000_000_000


def make_compress_ratios(n_layers: int, n_mtp_layers: int) -> list[int]:
    ratios = [0, 0]
    while len(ratios) < n_layers + n_mtp_layers:
        ratios.extend([4, 128])
    ratios = ratios[: n_layers + n_mtp_layers]
    ratios[-1] = 0
    return ratios


@dataclass(frozen=True)
class DSV4Config:
    vocab_size: int = 248_320
    hidden_size: int = 2048
    moe_intermediate_size: int = 512
    num_hidden_layers: int = 40
    num_hash_layers: int = 3
    num_nextn_predict_layers: int = 1
    num_attention_heads: int = 32
    head_dim: int = 256
    q_lora_rank: int = 512
    o_lora_rank: int = 512
    o_groups: int = 8
    qk_rope_head_dim: int = 64
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4

    @property
    def compress_ratios(self) -> list[int]:
        return make_compress_ratios(self.num_hidden_layers, self.num_nextn_predict_layers)


def dsv4_attention_params(cfg: DSV4Config, layer_id: int) -> dict[str, int]:
    d = cfg.hidden_size
    h = cfg.num_attention_heads
    hd = cfg.head_dim
    q_rank = cfg.q_lora_rank
    o_rank = cfg.o_lora_rank
    o_groups = cfg.o_groups
    rope = cfg.qk_rope_head_dim
    idx_h = cfg.index_n_heads
    idx_hd = cfg.index_head_dim
    cr = cfg.compress_ratios[layer_id]

    parts: dict[str, int] = {
        "attn_sink": h,
        "q_lora_a": d * q_rank,
        "q_norm": q_rank,
        "q_lora_b": q_rank * h * hd,
        "kv": d * hd,
        "kv_norm": hd,
        "o_lora_a": o_rank * h * hd,
        "o_lora_b": d * o_groups * o_rank,
    }

    if cr:
        coff = 2 if cr == 4 else 1
        parts.update(
            {
                "compressor_ape": cr * coff * hd,
                "compressor_wkv": d * coff * hd,
                "compressor_wgate": d * coff * hd,
                "compressor_norm": hd,
            }
        )
        if cr == 4:
            parts.update(
                {
                    "indexer_q_b": q_rank * idx_h * idx_hd,
                    "indexer_weights_proj": d * idx_h,
                    "indexer_compressor_ape": cr * 2 * idx_hd,
                    "indexer_compressor_wkv": d * 2 * idx_hd,
                    "indexer_compressor_wgate": d * 2 * idx_hd,
                    "indexer_compressor_norm": idx_hd,
                }
            )
    return parts


def dsv4_moe_params(cfg: DSV4Config, layer_id: int, active: bool) -> dict[str, int]:
    d = cfg.hidden_size
    inter = cfg.moe_intermediate_size
    routed = cfg.n_routed_experts
    shared = cfg.n_shared_experts
    active_routed = cfg.num_experts_per_tok if active else routed
    gate = routed * d
    if layer_id < cfg.num_hash_layers:
        gate += cfg.vocab_size * cfg.num_experts_per_tok
    else:
        gate += routed
    return {
        "router": gate,
        "routed_experts": active_routed * 3 * d * inter,
        "shared_experts": shared * 3 * d * inter,
    }


def dsv4_hc_params(cfg: DSV4Config) -> int:
    hc = cfg.hc_mult
    mix_hc = (2 + hc) * hc
    hc_dim = hc * cfg.hidden_size
    return 2 * (mix_hc * hc_dim + mix_hc + 3)


def dsv4_block_params(cfg: DSV4Config, layer_id: int, active: bool, mtp_extra: bool = False) -> dict[str, int]:
    d = cfg.hidden_size
    parts: dict[str, int] = {}
    for key, value in dsv4_attention_params(cfg, layer_id).items():
        parts[f"attention.{key}"] = value
    for key, value in dsv4_moe_params(cfg, layer_id, active).items():
        parts[f"moe.{key}"] = value
    parts["norms"] = 2 * d
    parts["hyper_connections"] = dsv4_hc_params(cfg)
    if mtp_extra:
        hc = cfg.hc_mult
        parts["mtp_extra"] = 2 * d * d + 3 * d + hc * (hc * d) + hc + 1
    return parts


def sum_parts(parts: dict[str, int]) -> int:
    return sum(parts.values())


def dsv4_total_params(cfg: DSV4Config, active: bool) -> dict[str, Any]:
    global_parts: dict[str, int] = {
        "embed_tokens": cfg.vocab_size * cfg.hidden_size,
        "final_norm": cfg.hidden_size,
        "lm_head": cfg.vocab_size * cfg.hidden_size,
        "lm_head_hyper_connection": cfg.hc_mult * (cfg.hc_mult * cfg.hidden_size) + cfg.hc_mult + 1,
    }
    layer_summaries = []
    blocks_total = 0
    for layer_id in range(cfg.num_hidden_layers):
        parts = dsv4_block_params(cfg, layer_id, active)
        total = sum_parts(parts)
        blocks_total += total
        layer_summaries.append(
            {
                "layer_id": layer_id,
                "compress_ratio": cfg.compress_ratios[layer_id],
                "params": total,
                "parts": parts,
            }
        )

    mtp_total = 0
    mtp_summaries = []
    for mtp_id in range(cfg.num_nextn_predict_layers):
        layer_id = cfg.num_hidden_layers + mtp_id
        parts = dsv4_block_params(cfg, layer_id, active, mtp_extra=True)
        total = sum_parts(parts)
        mtp_total += total
        mtp_summaries.append(
            {
                "mtp_id": mtp_id,
                "layer_id": layer_id,
                "compress_ratio": cfg.compress_ratios[layer_id],
                "params": total,
                "parts": parts,
            }
        )

    total = sum_parts(global_parts) + blocks_total + mtp_total
    return {
        "active": active,
        "total": total,
        "total_b": b(total),
        "global_parts": global_parts,
        "transformer_blocks_total": blocks_total,
        "mtp_total": mtp_total,
        "layer_summaries": layer_summaries,
        "mtp_summaries": mtp_summaries,
    }


def qwen_text_params_from_config(config: dict[str, Any], active: bool = False) -> dict[str, Any]:
    text = config["text_config"]
    d = text["hidden_size"]
    vocab = text["vocab_size"]
    n_layers = text["num_hidden_layers"]
    n_experts = text["num_experts"]
    topk = text["num_experts_per_tok"]
    inter = text["moe_intermediate_size"]
    shared_inter = text["shared_expert_intermediate_size"]
    layer_types = text["layer_types"]

    linear_key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    linear_value_dim = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    conv_dim = linear_key_dim * 2 + linear_value_dim
    linear_attn = {
        "conv1d": conv_dim * text["linear_conv_kernel_dim"],
        "dt_bias": text["linear_num_value_heads"],
        "a_log": text["linear_num_value_heads"],
        "gated_norm": text["linear_value_head_dim"],
        "out_proj": linear_value_dim * d,
        "in_proj_qkv": d * conv_dim,
        "in_proj_z": d * linear_value_dim,
        "in_proj_b": d * text["linear_num_value_heads"],
        "in_proj_a": d * text["linear_num_value_heads"],
    }
    full_attn = {
        "q_proj": d * text["num_attention_heads"] * text["head_dim"] * 2,
        "k_proj": d * text["num_key_value_heads"] * text["head_dim"],
        "v_proj": d * text["num_key_value_heads"] * text["head_dim"],
        "o_proj": d * text["num_attention_heads"] * text["head_dim"],
        "q_norm": text["head_dim"],
        "k_norm": text["head_dim"],
    }
    active_experts = topk if active else n_experts
    moe = {
        "router": n_experts * d,
        "routed_experts": active_experts * (2 * inter * d + d * inter),
        "shared_expert": 2 * shared_inter * d + d * shared_inter,
        "shared_expert_gate": d,
    }
    layer_norms = 2 * d
    layer_summaries = []
    total_layers = 0
    for i, layer_type in enumerate(layer_types):
        attn = linear_attn if layer_type == "linear_attention" else full_attn
        parts = {f"attention.{k}": v for k, v in attn.items()}
        parts.update({f"moe.{k}": v for k, v in moe.items()})
        parts["norms"] = layer_norms
        total = sum_parts(parts)
        total_layers += total
        layer_summaries.append({"layer_id": i, "layer_type": layer_type, "params": total, "parts": parts})
    global_parts = {"embed_tokens": vocab * d, "final_norm": d, "lm_head": vocab * d}
    total = sum_parts(global_parts) + total_layers
    return {
        "active": active,
        "total": total,
        "total_b": b(total),
        "global_parts": global_parts,
        "layers_total": total_layers,
        "layer_summaries": layer_summaries,
    }


def main() -> None:
    out_dir = Path("remote_param_count_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    fetched = {
        "qwen_config": fetch_text(QWEN_CONFIG_URL),
        "qwen_modeling": fetch_text(QWEN_MODELING_URL),
        "qwen_configuration": fetch_text(QWEN_CONFIGURATION_URL),
        "dsv4_config": fetch_text(DSV4_CONFIG_URL),
        "dsv4_inference_config": fetch_text(DSV4_INFERENCE_CONFIG_URL),
        "dsv4_modeling": fetch_text(DSV4_MODELING_URL),
    }
    for name, text in fetched.items():
        (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")

    qwen_config = json.loads(fetched["qwen_config"])
    dsv4_config = json.loads(fetched["dsv4_config"])
    dsv4_inference_config = json.loads(fetched["dsv4_inference_config"])

    nano_cfg = DSV4Config()
    report = {
        "sources": {
            name: {
                "url": url,
                "sha256": sha256_text(fetched[name]),
                "bytes": len(fetched[name].encode("utf-8")),
            }
            for name, url in {
                "qwen_config": QWEN_CONFIG_URL,
                "qwen_modeling": QWEN_MODELING_URL,
                "qwen_configuration": QWEN_CONFIGURATION_URL,
                "dsv4_config": DSV4_CONFIG_URL,
                "dsv4_inference_config": DSV4_INFERENCE_CONFIG_URL,
                "dsv4_modeling": DSV4_MODELING_URL,
            }.items()
        },
        "upstream_key_configs": {
            "qwen_35a3_text": qwen_config["text_config"],
            "deepseek_v4_flash_hf": dsv4_config,
            "deepseek_v4_flash_inference": dsv4_inference_config,
        },
        "qwen_35a3_text_count": {
            "total": qwen_text_params_from_config(qwen_config, active=False),
            "active": qwen_text_params_from_config(qwen_config, active=True),
            "note": "Text-only count from HF Qwen3.5 MoE modeling classes; excludes vision tower.",
        },
        "deepseek_v4_flash_count": {
            "total": dsv4_total_params(
                DSV4Config(
                    vocab_size=dsv4_inference_config["vocab_size"],
                    hidden_size=dsv4_inference_config["dim"],
                    moe_intermediate_size=dsv4_inference_config["moe_inter_dim"],
                    num_hidden_layers=dsv4_inference_config["n_layers"],
                    num_hash_layers=dsv4_inference_config["n_hash_layers"],
                    num_nextn_predict_layers=dsv4_inference_config.get("n_mtp_layers", 1),
                    num_attention_heads=dsv4_inference_config["n_heads"],
                    head_dim=dsv4_inference_config["head_dim"],
                    q_lora_rank=dsv4_inference_config["q_lora_rank"],
                    o_lora_rank=dsv4_inference_config["o_lora_rank"],
                    o_groups=dsv4_inference_config["o_groups"],
                    qk_rope_head_dim=dsv4_inference_config["rope_head_dim"],
                    n_routed_experts=dsv4_inference_config["n_routed_experts"],
                    n_shared_experts=dsv4_inference_config["n_shared_experts"],
                    num_experts_per_tok=dsv4_inference_config["n_activated_experts"],
                    index_n_heads=dsv4_inference_config["index_n_heads"],
                    index_head_dim=dsv4_inference_config["index_head_dim"],
                    index_topk=dsv4_inference_config["index_topk"],
                    hc_mult=dsv4_inference_config["hc_mult"],
                ),
                active=False,
            ),
            "active": dsv4_total_params(
                DSV4Config(
                    vocab_size=dsv4_inference_config["vocab_size"],
                    hidden_size=dsv4_inference_config["dim"],
                    moe_intermediate_size=dsv4_inference_config["moe_inter_dim"],
                    num_hidden_layers=dsv4_inference_config["n_layers"],
                    num_hash_layers=dsv4_inference_config["n_hash_layers"],
                    num_nextn_predict_layers=dsv4_inference_config.get("n_mtp_layers", 1),
                    num_attention_heads=dsv4_inference_config["n_heads"],
                    head_dim=dsv4_inference_config["head_dim"],
                    q_lora_rank=dsv4_inference_config["q_lora_rank"],
                    o_lora_rank=dsv4_inference_config["o_lora_rank"],
                    o_groups=dsv4_inference_config["o_groups"],
                    qk_rope_head_dim=dsv4_inference_config["rope_head_dim"],
                    n_routed_experts=dsv4_inference_config["n_routed_experts"],
                    n_shared_experts=dsv4_inference_config["n_shared_experts"],
                    num_experts_per_tok=dsv4_inference_config["n_activated_experts"],
                    index_n_heads=dsv4_inference_config["index_n_heads"],
                    index_head_dim=dsv4_inference_config["index_head_dim"],
                    index_topk=dsv4_inference_config["index_topk"],
                    hc_mult=dsv4_inference_config["hc_mult"],
                ),
                active=True,
            ),
            "note": "Logical architecture count from DeepSeek-V4 inference/model.py modules; excludes quantization scale tensors and runtime KV-cache buffers.",
        },
        "deepseek_v4_nano_35a3_config": asdict(nano_cfg) | {"compress_ratios": nano_cfg.compress_ratios},
        "deepseek_v4_nano_35a3_count": {
            "total": dsv4_total_params(nano_cfg, active=False),
            "active": dsv4_total_params(nano_cfg, active=True),
            "note": "DSV4 architecture scaled to Qwen3.5-35B-A3B text dimensions: hidden 2048, 40 layers, 256 experts, top-8, expert intermediate 512, vocab 248320.",
        },
    }

    (out_dir / "param_count_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "qwen_35a3_text_total_b": report["qwen_35a3_text_count"]["total"]["total_b"],
        "qwen_35a3_text_active_b": report["qwen_35a3_text_count"]["active"]["total_b"],
        "dsv4_flash_total_b": report["deepseek_v4_flash_count"]["total"]["total_b"],
        "dsv4_flash_active_b": report["deepseek_v4_flash_count"]["active"]["total_b"],
        "dsv4_nano_total_b": report["deepseek_v4_nano_35a3_count"]["total"]["total_b"],
        "dsv4_nano_active_b": report["deepseek_v4_nano_35a3_count"]["active"]["total_b"],
        "dsv4_nano_exact_total": report["deepseek_v4_nano_35a3_count"]["total"]["total"],
        "dsv4_nano_exact_active": report["deepseek_v4_nano_35a3_count"]["active"]["total"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
