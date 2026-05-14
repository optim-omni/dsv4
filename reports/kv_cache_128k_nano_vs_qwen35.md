# 128k KV / State 内存：DSV4 Nano vs Qwen3.5-35B-A3B

本文只比较两个对象：

- `dsv4-nano-35a3`
- `Qwen/Qwen3.5-35B-A3B`

计算口径：

- batch size: `1`
- 序列长度: `128k = 131072`
- DSV4-nano KV 存储：非 RoPE 维度用 FP8，RoPE 维度用 BF16
- Qwen3.5 full-attention KV：INT8
- Qwen3.5 linear-attention recurrent state：按原版模型 dtype，即 BF16
- 单位使用二进制 `MiB/GiB`

## DSV4 Nano

源码形状：

- 层数：`40`
- attention latent `head_dim`: `256`
- RoPE 维度：`64`
- indexer `head_dim`: `128`
- sliding window: `128`
- 40 个主层的压缩比例分布：
  - `2` 层 `compress_ratio = 0`
  - `19` 层 `compress_ratio = 4`
  - `19` 层 `compress_ratio = 128`

单个 DSV4 attention 层的主 KV 公式：

```text
kv_slots(layer) = sliding_window + (seq_len / compress_ratio if compress_ratio > 0 else 0)
main_kv_bytes(layer) = kv_slots(layer) * head_dim * main_bytes_per_element
```

主 KV 的混合精度平均字节数：

```text
main_bytes_per_element = ((head_dim - rope_dim) * 1 + rope_dim * 2) / head_dim
                       = ((256 - 64) * 1 + 64 * 2) / 256
                       = 1.25 bytes
```

对于 `compress_ratio = 4` 的层，DSV4 还有 indexer KV。这里的 `/4` 不是额外假设，而是来自源码里的 indexer 自己带 `Compressor(args, compress_ratio, self.head_dim, True)`，并且它的 `kv_cache` shape 是 `args.max_seq_len // compress_ratio`。也就是说，indexer 保存的是压缩位置的 scoring cache，不是 raw-token 级别的 128k 全量 cache。

```text
indexer_slots(layer) = seq_len / 4
indexer_bytes(layer) = indexer_slots(layer) * index_head_dim * indexer_bytes_per_element
```

对应源码锚点：

```text
sources/deepseek_v4_flash/modeling_deepseek_v4.py:
  Indexer.__init__:
    self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
    self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim), persistent=False)
  Indexer.forward:
    index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
```

本文按 DSV4-nano compressed cache 的混合精度口径计算：非 RoPE 维度 FP8，RoPE 维度 BF16。

```text
indexer_bytes_per_element = ((index_head_dim - rope_dim) * 1 + rope_dim * 2) / index_head_dim
                          = ((128 - 64) * 1 + 64 * 2) / 128
                          = 1.5 bytes
```

结果：

```text
main KV    = 197.5 MiB
indexer KV = 114.0 MiB
total      = 311.5 MiB
```

## Qwen3.5-35B-A3B

源码形状：

- 层数：`40`
- 层模式：`30` 个 linear-attention 层 + `10` 个 full-attention 层
- linear attention checkpoint 间隔：`1024`
- 128k 下 checkpoint 数量：`131072 / 1024 = 128`
- linear value heads: `32`
- linear key head dim: `128`
- linear value head dim: `128`
- linear conv dim:

```text
conv_dim = linear_num_key_heads * linear_key_head_dim * 2
         + linear_num_value_heads * linear_value_head_dim
         = 16 * 128 * 2 + 32 * 128
         = 8192
```

每个 linear-attention 层、每个 checkpoint 的 recurrent state：

```text
recurrent_elements = linear_num_value_heads * linear_key_head_dim * linear_value_head_dim
                   = 32 * 128 * 128
                   = 524288 elements
```

linear recurrent checkpoint 内存：

```text
recurrent_bytes = linear_layers * checkpoints * recurrent_elements * bf16_bytes
                = 30 * 128 * 524288 * 2
                = 3840 MiB
                = 3.75 GiB
```

full-attention INT8 KV 内存：

```text
full_kv_elements_per_token_per_layer = 2 * num_key_value_heads * head_dim
                                     = 2 * 2 * 256
                                     = 1024 elements

full_kv_bytes = full_attention_layers * seq_len * full_kv_elements_per_token_per_layer * int8_bytes
              = 10 * 131072 * 1024 * 1
              = 1280 MiB
              = 1.25 GiB
```

如果也 checkpoint conv state：

```text
conv_state_bytes = linear_layers * checkpoints * conv_dim * (conv_kernel_size - 1) * bf16_bytes
                 = 30 * 128 * 8192 * 3 * 2
                 = 180 MiB
                 = 0.17578125 GiB
```

结果：

```text
linear recurrent checkpoints = 3840 MiB
full-attention INT8 KV       = 1280 MiB
total without conv state     = 5120 MiB = 5.0 GiB
conv state, if checkpointed  = 180 MiB = 0.17578125 GiB
total with conv state        = 5300 MiB = 5.17578125 GiB
```

## 结果

| 模型 / 口径 | 128k state memory |
| --- | ---: |
| DSV4-nano compressed KV，混合 FP8/BF16 | `311.5 MiB` |
| Qwen3.5 linear recurrent checkpoints + full-attention INT8 KV | `5120 MiB = 5.0 GiB` |
| Qwen3.5 再加 checkpointed conv state | `5300 MiB = 5.17578125 GiB` |

相对 DSV4-nano：

```text
Qwen without conv state = 5120 / 311.5 = 16.44x
Qwen with conv state    = 5300 / 311.5 = 17.01x
```

一句话结论：在 `128k / batch=1 / Qwen ckpt1024` 这个口径下，Qwen3.5 的 linear-attention checkpoint state 仍是主要内存开销之一。即使 full-attention KV 用 INT8，Qwen3.5 仍需要大约 `5.0 GiB` 保存 recurrent checkpoints + full-attention KV；DSV4-nano 的 compressed mixed-precision KV 只需要 `311.5 MiB`。
