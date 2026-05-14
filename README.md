# DSV4 Nano Configs

Minimal DeepSeek V4 style nano configurations, parameter-count scripts, upstream modeling references, and remote count evidence.

The counts were computed on `cudo-h100` from the DeepSeek V4 Flash `inference/model.py` module structure and the Qwen3.5-35B-A3B config/modeling references.

## Layout

- `configs/`: packaged model variants. Each variant includes `config.json`, `inference/config.json`, `modeling_deepseek_v4.py`, and a parameter-count note.
- `sources/`: upstream reference files split by model family.
- `reports/`: remote count reports and sweeps.
- `scripts/`: reproducible parameter-count script.

## Variants

| Variant | Vocab | Routed experts | Top-k active experts | Total params | Active params |
| --- | ---: | ---: | ---: | ---: | ---: |
| `configs/dsv4-nano-35a3` | 248320 | 256 | 8 | 35.089736272B | 3.103973968B |
| `configs/dsv4-nano-35a2_6b` | 248320 | 256 | 4 | 35.086756432B | 2.585094736B |
| `configs/dsv4-nano-35a3-vocab150k` | 150016 | 256 | 10 | 34.685623888B | 2.957811280B |
| `configs/dsv4-nano-35a2_6b-vocab150k` | 150016 | 256 | 7 | 34.684273744B | 2.569536592B |
| `configs/dsv4-nano-80a3-vocab150k` | 150016 | 640 | 10 | 84.244223824B | 2.990069584B |

`150000` vocab targets are rounded up to `150016` so `vocab_size % 128 == 0`.

## Source Anchors

- Qwen3.5-35B-A3B config and modeling: `sources/qwen3_5_35a3/`
- DeepSeek V4 Flash config and modeling: `sources/deepseek_v4_flash/`
- Full fetched remote snapshot: `reports/remote_param_count_out/`

## Reports

- `reports/kv_cache_128k_nano_vs_qwen35.md`: 128k KV/cache footprint comparison for DSV4-nano and Qwen3.5.
- `reports/prefix_cache_offload_128k_nano_vs_qwen35.md`: offload sizing for instant-recoverable prefix cache.

## Reproduce

```bash
python3 scripts/param_count_dsv4.py
```

The script fetches the upstream sources, writes `remote_param_count_out/`, and prints the count summary.

Cache/offload calculations:

```bash
python3 scripts/cache_memory_128k.py --format text
python3 scripts/cache_memory_128k.py --check-reports
```
