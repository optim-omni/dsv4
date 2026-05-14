#!/usr/bin/env python3
"""128k KV/state/offload calculator for DSV4-nano vs Qwen3.5-35B-A3B.

The script keeps the arithmetic behind the markdown reports reproducible. It
does not import model code; the constants below are the architecture-level
shapes read from the source snapshots in this repository.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MIB = 1024**2


@dataclass(frozen=True)
class DSV4NanoShape:
    layers: int = 40
    head_dim: int = 256
    rope_dim: int = 64
    index_head_dim: int = 128
    sliding_window: int = 128
    compress_ratio_0_layers: int = 2
    compress_ratio_4_layers: int = 19
    compress_ratio_128_layers: int = 19

    @property
    def main_bytes_per_element(self) -> float:
        non_rope = self.head_dim - self.rope_dim
        return (non_rope * 1 + self.rope_dim * 2) / self.head_dim

    @property
    def indexer_bytes_per_element(self) -> float:
        # Report/storage assumption: non-RoPE FP8 + RoPE BF16.
        non_rope = self.index_head_dim - self.rope_dim
        return (non_rope * 1 + self.rope_dim * 2) / self.index_head_dim


@dataclass(frozen=True)
class Qwen35Shape:
    layers: int = 40
    linear_layers: int = 30
    full_attention_layers: int = 10
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    conv_kernel_size: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 256
    recurrent_state_bytes_per_element: int = 2
    conv_state_bytes_per_element: int = 2
    full_kv_bytes_per_element: int = 1

    @property
    def recurrent_elements_per_layer(self) -> int:
        return self.linear_num_value_heads * self.linear_key_head_dim * self.linear_value_head_dim

    @property
    def conv_dim(self) -> int:
        key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        value_dim = self.linear_num_value_heads * self.linear_value_head_dim
        return key_dim * 2 + value_dim

    @property
    def full_kv_elements_per_token_per_layer(self) -> int:
        return 2 * self.num_key_value_heads * self.head_dim


def bytes_to_mib(value: float) -> float:
    return value / MIB


def mib_to_gib(value: float) -> float:
    return value / 1024


def dsv4_nano_cache(seq_len: int, checkpoint_interval: int) -> dict[str, Any]:
    shape = DSV4NanoShape()
    checkpoints = seq_len // checkpoint_interval

    def compressor_state_mib(compress_ratio: int, head_dim: int) -> float:
        coff = 2 if compress_ratio == 4 else 1
        elements_per_buffer = coff * compress_ratio * coff * head_dim
        return bytes_to_mib(elements_per_buffer * 2 * 4)

    compressor_state_main_r4_mib = shape.compress_ratio_4_layers * compressor_state_mib(4, shape.head_dim)
    compressor_state_main_r128_mib = shape.compress_ratio_128_layers * compressor_state_mib(128, shape.head_dim)
    compressor_state_indexer_mib = shape.compress_ratio_4_layers * compressor_state_mib(4, shape.index_head_dim)
    compressor_incremental_state_mib = (
        compressor_state_main_r4_mib + compressor_state_main_r128_mib + compressor_state_indexer_mib
    )

    window_mib = bytes_to_mib(
        shape.layers * shape.sliding_window * shape.head_dim * shape.main_bytes_per_element
    )
    all_checkpoint_windows_mib = bytes_to_mib(
        shape.layers * seq_len * shape.head_dim * shape.main_bytes_per_element
    )

    cr0_main_mib = bytes_to_mib(
        shape.compress_ratio_0_layers * shape.sliding_window * shape.head_dim * shape.main_bytes_per_element
    )
    cr4_main_mib = bytes_to_mib(
        shape.compress_ratio_4_layers
        * (shape.sliding_window + seq_len / 4)
        * shape.head_dim
        * shape.main_bytes_per_element
    )
    cr128_main_mib = bytes_to_mib(
        shape.compress_ratio_128_layers
        * (shape.sliding_window + seq_len / 128)
        * shape.head_dim
        * shape.main_bytes_per_element
    )
    main_compressed_mib = cr0_main_mib + cr4_main_mib + cr128_main_mib
    # Source anchor:
    # Indexer.__init__ uses Compressor(..., compress_ratio, ..., rotate=True)
    # and registers kv_cache with shape max_seq_len // compress_ratio.
    indexer_slots_per_layer = seq_len / 4
    indexer_mib = bytes_to_mib(
        shape.compress_ratio_4_layers
        * indexer_slots_per_layer
        * shape.index_head_dim
        * shape.indexer_bytes_per_element
    )
    compressed_backing_without_window_mib = main_compressed_mib - window_mib + indexer_mib

    resident_with_compressor_mib = window_mib + compressor_incremental_state_mib
    external_ckpt128_mib = all_checkpoint_windows_mib + compressed_backing_without_window_mib
    restore_read_min_mib = resident_with_compressor_mib
    restore_read_materialized_worst_mib = (
        compressed_backing_without_window_mib + window_mib + compressor_incremental_state_mib
    )

    return {
        "shape": asdict(shape),
        "checkpoints": checkpoints,
        "resident": {
            "window_mib": window_mib,
            "compressor_incremental_state": {
                "main_ratio4_mib": compressor_state_main_r4_mib,
                "main_ratio128_mib": compressor_state_main_r128_mib,
                "indexer_ratio4_mib": compressor_state_indexer_mib,
                "total_mib": compressor_incremental_state_mib,
            },
            "with_compressor_state_mib": resident_with_compressor_mib,
        },
        "external_exact_prefix": {
            "indexer_slots_per_layer": indexer_slots_per_layer,
            "main_compressed_kv_mib": main_compressed_mib,
            "indexer_kv_mib": indexer_mib,
            "total_with_window_mib": main_compressed_mib + indexer_mib,
            "total_without_window_mib": compressed_backing_without_window_mib,
        },
        "external_ckpt128": {
            "all_checkpoint_windows_mib": all_checkpoint_windows_mib,
            "compressed_backing_without_window_mib": compressed_backing_without_window_mib,
            "total_mib": external_ckpt128_mib,
            "total_gib": mib_to_gib(external_ckpt128_mib),
        },
        "restore_read": {
            "offset_pointer_min_mib": restore_read_min_mib,
            "materialized_128k_worst_mib": restore_read_materialized_worst_mib,
        },
    }


def qwen35_cache(seq_len: int, checkpoint_interval: int) -> dict[str, Any]:
    shape = Qwen35Shape()
    checkpoints = seq_len // checkpoint_interval

    recurrent_one_mib = bytes_to_mib(
        shape.linear_layers * shape.recurrent_elements_per_layer * shape.recurrent_state_bytes_per_element
    )
    conv_one_mib = bytes_to_mib(
        shape.linear_layers
        * shape.conv_dim
        * (shape.conv_kernel_size - 1)
        * shape.conv_state_bytes_per_element
    )
    linear_state_one_mib = recurrent_one_mib + conv_one_mib
    full_kv_mib = bytes_to_mib(
        shape.full_attention_layers
        * seq_len
        * shape.full_kv_elements_per_token_per_layer
        * shape.full_kv_bytes_per_element
    )
    external_recurrent_ckpt128_mib = recurrent_one_mib * checkpoints
    external_conv_ckpt128_mib = conv_one_mib * checkpoints
    external_linear_ckpt128_mib = external_recurrent_ckpt128_mib + external_conv_ckpt128_mib
    resident_with_full_kv_mib = full_kv_mib + linear_state_one_mib
    external_with_full_kv_mib = external_linear_ckpt128_mib + full_kv_mib

    return {
        "shape": asdict(shape),
        "checkpoints": checkpoints,
        "resident": {
            "linear_current_mib": recurrent_one_mib,
            "conv_current_mib": conv_one_mib,
            "linear_plus_conv_current_mib": linear_state_one_mib,
            "full_attention_kv_int8_mib": full_kv_mib,
            "with_full_kv_mib": resident_with_full_kv_mib,
            "with_full_kv_gib": mib_to_gib(resident_with_full_kv_mib),
        },
        "external_exact_prefix": {
            "linear_state_mib": linear_state_one_mib,
            "with_full_kv_mib": linear_state_one_mib + full_kv_mib,
            "with_full_kv_gib": mib_to_gib(linear_state_one_mib + full_kv_mib),
        },
        "external_ckpt128": {
            "recurrent_mib": external_recurrent_ckpt128_mib,
            "recurrent_gib": mib_to_gib(external_recurrent_ckpt128_mib),
            "conv_mib": external_conv_ckpt128_mib,
            "conv_gib": mib_to_gib(external_conv_ckpt128_mib),
            "linear_plus_conv_mib": external_linear_ckpt128_mib,
            "linear_plus_conv_gib": mib_to_gib(external_linear_ckpt128_mib),
            "with_full_kv_mib": external_with_full_kv_mib,
            "with_full_kv_gib": mib_to_gib(external_with_full_kv_mib),
        },
        "restore_read": {
            "linear_state_only_mib": linear_state_one_mib,
            "with_full_kv_128k_worst_mib": linear_state_one_mib + full_kv_mib,
            "with_full_kv_128k_worst_gib": mib_to_gib(linear_state_one_mib + full_kv_mib),
        },
    }


def build_results(seq_len: int = 131_072, checkpoint_interval: int = 128) -> dict[str, Any]:
    return {
        "assumptions": {
            "batch_size": 1,
            "seq_len": seq_len,
            "checkpoint_interval": checkpoint_interval,
            "checkpoints": seq_len // checkpoint_interval,
            "units": "MiB/GiB are binary units",
            "dsv4_precision": "non-RoPE FP8 + RoPE BF16",
            "qwen_full_kv_precision": "INT8",
            "qwen_linear_state_precision": "BF16",
        },
        "dsv4_nano": dsv4_nano_cache(seq_len, checkpoint_interval),
        "qwen3_5_35b_a3b": qwen35_cache(seq_len, checkpoint_interval),
    }


def format_mib(value: float) -> str:
    if value.is_integer():
        return f"{int(value)} MiB"
    return f"{value} MiB"


def format_summary(results: dict[str, Any]) -> str:
    dsv4 = results["dsv4_nano"]
    qwen = results["qwen3_5_35b_a3b"]
    rows = [
        (
            "DSV4-nano, CSA offload",
            format_mib(dsv4["resident"]["with_compressor_state_mib"]),
            f'{dsv4["external_ckpt128"]["total_mib"]} MiB = {dsv4["external_ckpt128"]["total_gib"]} GiB',
            format_mib(dsv4["restore_read"]["offset_pointer_min_mib"]),
        ),
        (
            "Qwen3.5, linear state offload, full KV resident",
            f'{qwen["resident"]["with_full_kv_mib"]} MiB = {qwen["resident"]["with_full_kv_gib"]} GiB',
            f'{format_mib(qwen["external_ckpt128"]["linear_plus_conv_mib"])} = '
            f'{qwen["external_ckpt128"]["linear_plus_conv_gib"]} GiB',
            format_mib(qwen["restore_read"]["linear_state_only_mib"]),
        ),
        (
            "Qwen3.5, linear state + full KV offload",
            format_mib(qwen["resident"]["linear_plus_conv_current_mib"]),
            f'{format_mib(qwen["external_ckpt128"]["with_full_kv_mib"])} = '
            f'{qwen["external_ckpt128"]["with_full_kv_gib"]} GiB',
            f'{qwen["restore_read"]["with_full_kv_128k_worst_mib"]} MiB = '
            f'{qwen["restore_read"]["with_full_kv_128k_worst_gib"]} GiB',
        ),
    ]
    lines = [
        "ckpt128 arbitrary-boundary prefix cache recovery",
        "",
        "| scheme | resident memory | external backing | per-hit restore read |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(f"| {scheme} | `{resident}` | `{external}` | `{restore}` |" for scheme, resident, external, restore in rows)
    return "\n".join(lines)


def check_reports(results: dict[str, Any], repo_root: Path) -> None:
    report_paths = [
        repo_root / "reports" / "kv_cache_128k_nano_vs_qwen35.md",
        repo_root / "reports" / "prefix_cache_offload_128k_nano_vs_qwen35.md",
    ]
    text = "\n".join(path.read_text() for path in report_paths)
    dsv4 = results["dsv4_nano"]
    qwen = results["qwen3_5_35b_a3b"]
    expected = [
        f'{dsv4["external_exact_prefix"]["total_with_window_mib"]} MiB',
        f'{dsv4["external_ckpt128"]["total_mib"]} MiB',
        f'{dsv4["restore_read"]["materialized_128k_worst_mib"]} MiB',
        f'{int(qwen["external_ckpt128"]["recurrent_mib"])} MiB',
        f'{int(qwen["external_ckpt128"]["linear_plus_conv_mib"])} MiB',
        f'{int(qwen["external_ckpt128"]["with_full_kv_mib"])} MiB',
        f'{qwen["external_ckpt128"]["with_full_kv_gib"]} GiB',
    ]
    missing = [value for value in expected if value not in text]
    if missing:
        raise SystemExit(f"Report check failed; missing values: {missing}")


def check_source_anchors(repo_root: Path) -> None:
    dsv4_modeling = (repo_root / "sources" / "deepseek_v4_flash" / "modeling_deepseek_v4.py").read_text()
    expected = [
        "self.compressor = Compressor(args, compress_ratio, self.head_dim, True)",
        "args.max_seq_len // compress_ratio",
        "self.kv_cache[:bsz, :end_pos // ratio]",
    ]
    missing = [snippet for snippet in expected if snippet not in dsv4_modeling]
    if missing:
        raise SystemExit(f"Source anchor check failed; missing snippets: {missing}")


def check_all(results: dict[str, Any], repo_root: Path) -> None:
    check_source_anchors(repo_root)
    check_reports(results, repo_root)
    print("cache_memory_reports_ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=131_072)
    parser.add_argument("--checkpoint-interval", type=int, default=128)
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.add_argument("--check-reports", action="store_true")
    args = parser.parse_args()

    if args.seq_len % args.checkpoint_interval != 0:
        raise SystemExit("seq-len must be divisible by checkpoint-interval")

    results = build_results(args.seq_len, args.checkpoint_interval)
    if args.check_reports:
        check_all(results, Path(__file__).resolve().parents[1])
        return
    if args.format == "text":
        print(format_summary(results))
    else:
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
