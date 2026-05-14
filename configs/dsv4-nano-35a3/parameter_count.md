# DeepSeek V4 Nano 35A3 Parameter Count

This is a minimal configuration-only pass. The target keeps the DeepSeek V4 architecture and scales the model to the Qwen3.5-35B-A3B text size.

## Remote Evidence

The parameter count was run on `cudo-h100` (`lux-2-edmonton-node-2`) under `~/dsv4_nano_param_count`.

Remote command:

```bash
cd ~/dsv4_nano_param_count && python3 param_count_dsv4.py
```

Remote summary:

```json
{
  "qwen_35a3_text_total_b": 34.660610688,
  "qwen_35a3_text_active_b": 3.454988928,
  "dsv4_flash_total_b": 290.944616402,
  "dsv4_flash_active_b": 14.120552402,
  "dsv4_nano_total_b": 35.089736272,
  "dsv4_nano_active_b": 3.103973968,
  "dsv4_nano_exact_total": 35089736272,
  "dsv4_nano_exact_active": 3103973968
}
```

Full machine-generated report is in `docs/remote_param_count_report.json`.

## Source Anchors

- Qwen config: `https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json`
- Qwen modeling: `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`
- DeepSeek V4 Flash config: `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json`
- DeepSeek V4 Flash inference config: `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/inference/config.json`
- DeepSeek V4 Flash modeling: `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/inference/model.py`

Remote SHA prefixes:

```text
qwen_config              5e4d7f74fec2f360
qwen_modeling            94959429ca270ae0
qwen_configuration       a5051abd2b915ecb
dsv4_config              b628e63398a645ab
dsv4_inference_config    6cc6f816ca73a8d3
dsv4_modeling            ce962f1face79d4f
```

## Chosen Nano Config

The config uses the Qwen 35A3 scale knobs where they matter for parameter count:

- `hidden_size = 2048`
- `num_hidden_layers = 40`
- `n_routed_experts = 256`
- `num_experts_per_tok = 8`
- `moe_intermediate_size = 512`
- `vocab_size = 248320`

The architecture knobs remain DeepSeek V4 style:

- MLA with `q_lora_rank = 512`, `head_dim = 256`, `o_lora_rank = 512`
- DSV4 compression pattern with 41 entries for 40 layers plus 1 MTP layer
- Hyper-Connections with `hc_mult = 4`
- 3 hash-routing layers and 1 MTP layer

## Counting Assumption

Counts are logical architecture parameters. They include embeddings, untied LM head, transformer blocks, MTP block, MLA/compressor/indexer/Hyper-Connections, all routed experts for total count, top-8 routed plus shared expert for active count, and hash `tid2eid` tensors. They exclude quantization scale tensors and runtime KV-cache buffers.
