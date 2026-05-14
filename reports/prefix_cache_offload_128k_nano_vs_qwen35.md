# 128k Prefix Cache Offload：DSV4 Nano vs Qwen3.5-35B-A3B

本文回答一个更接近系统实现的问题：

如果假设 DSV4 的 CSA 状态和 Qwen3.5 的 linear state 都可以卸载到外存，为了保证 prefix cache 可以瞬时恢复，显存/内存和外存各需要多少？

这里的“外存”只表示不常驻 GPU 的 prefix-cache backing store，可以是 CPU RAM、pinned host memory、mmap 文件或 NVMe。是否真的“瞬时”，取决于带宽和调度；本文先算容量。

下文的“常驻内存”指恢复后必须留在快速路径里的常驻工作集，通常就是 GPU resident state；如果实现支持从 CPU/pinned/NVMe 直接分块取数，compressed cache 可以留在外存 backing 里，用指针/offset 恢复逻辑状态。

## 计算口径

- batch size: `1`
- 序列长度: `128k = 131072`
- prefix checkpoint 间隔: `128`
- checkpoint 数: `131072 / 128 = 1024`
- DSV4-nano KV: 非 RoPE 维度 FP8，RoPE 维度 BF16
- Qwen3.5 full-attention KV: INT8
- Qwen3.5 linear recurrent state: 原版 BF16
- 单位: 二进制 `MiB/GiB`

本文把“瞬时可恢复”分成两种口径：

1. 固定完整 prefix 恢复：只恢复一个确定的 128k prefix 的最终态。
2. ckpt128 任意边界恢复：任意 128-token checkpoint 命中都不需要从头 replay。

主表使用第 2 种，因为它更符合 prefix cache 在长上下文里复用中间前缀的需求。

## DSV4 Nano

### 常驻内存

DSV4 decode 当前步至少要有 sliding-window KV。40 层，每层 window 为 128：

```text
main_bytes_per_element = ((256 - 64) * 1 + 64 * 2) / 256
                       = 1.25 bytes

resident_window_bytes = layers * window * head_dim * main_bytes_per_element
                      = 40 * 128 * 256 * 1.25
                      = 1.5625 MiB
```

如果为了任意 prefix 长度恢复后立刻继续压缩，还保留 compressor 的增量状态：

```text
compressor_state_main_ratio4   = 0.59375 MiB
compressor_state_main_ratio128 = 4.75 MiB
compressor_state_indexer_ratio4 = 0.296875 MiB

compressor_incremental_state = 0.59375 + 4.75 + 0.296875
                             = 5.640625 MiB

resident_with_compressor_state = 1.5625 + 5.640625
                               = 7.203125 MiB
```

### 外存：固定完整 128k prefix

固定完整 prefix 只需要保存最终 128k prefix 对应的一份 DSV4 compressed cache：

```text
main compressed KV = 195.9375 MiB
indexer KV         = 114.0 MiB
sliding window KV  = 1.5625 MiB

external_exact_prefix_total = 311.5 MiB
```

这里的 indexer KV 是压缩位置的 scoring cache，不是 raw-token 级别全量 cache。源码里 `Indexer` 自己有 `Compressor(args, compress_ratio, self.head_dim, True)`，并注册 `args.max_seq_len // compress_ratio` 长度的 `kv_cache`；attention 里只有 `compress_ratio == 4` 的层会创建 indexer。

如果 sliding window 常驻内存、不放外存，则外存是：

```text
external_exact_prefix_without_window = 195.9375 + 114.0
                                     = 309.9375 MiB
```

### 外存：ckpt128 任意边界可恢复

要让任意 128-token 边界都能瞬时恢复，DSV4 需要两类 backing：

1. main compressed KV / indexer compressed scoring KV 可以按 prefix 长度切片复用；
2. 每个 checkpoint 的最后 128 token sliding window 也要可恢复，否则恢复到中间前缀后还要 replay 最近 128 token。

由于 checkpoint 间隔正好等于 sliding window，1024 个 window 不重叠，等价于保存 128k 全量 latent window backing：

```text
all_checkpoint_windows = seq_len * layers * head_dim * main_bytes_per_element
                       = 131072 * 40 * 256 * 1.25
                       = 1600 MiB
                       = 1.5625 GiB
```

compressed backing：

```text
compressed_backing = main_compressed_KV + indexer_KV
                   = 195.9375 + 114.0
                   = 309.9375 MiB
```

因此：

```text
external_ckpt128_total = all_checkpoint_windows + compressed_backing
                       = 1600 + 309.9375
                       = 1909.9375 MiB
                       = 1.86517333984375 GiB
```

### 单次命中恢复读回量

如果 compressed backing 可以按 offset 恢复，只需要读回当前 checkpoint 的 window 和 compressor 增量状态：

```text
restore_read_min = window + compressor_incremental_state
                 = 1.5625 + 5.640625
                 = 7.203125 MiB
```

如果恢复时必须把当前 prefix 对应的 compressed cache 也物化回 GPU，最坏读回量为：

```text
restore_read_materialized_worst = compressed_backing + window + compressor_incremental_state
                                = 309.9375 + 1.5625 + 5.640625
                                = 317.140625 MiB
```

## Qwen3.5-35B-A3B

Qwen3.5 有 `30` 个 linear-attention 层和 `10` 个 full-attention 层。

### 常驻内存

如果 linear state 已经卸载，decode 当前步仍需要当前 recurrent state：

```text
recurrent_elements = 32 * 128 * 128
                   = 524288

resident_linear_current = linear_layers * recurrent_elements * bf16_bytes
                        = 30 * 524288 * 2
                        = 30 MiB
```

如果 conv state 也要当前步可继续：

```text
conv_dim = 16 * 128 * 2 + 32 * 128
         = 8192

resident_conv_current = linear_layers * conv_dim * (conv_kernel_size - 1) * bf16_bytes
                      = 30 * 8192 * 3 * 2
                      = 1.40625 MiB
```

full-attention KV 按 INT8 常驻：

```text
full_kv_elements_per_token_per_layer = 2 * num_key_value_heads * head_dim
                                     = 2 * 2 * 256
                                     = 1024

resident_full_attention_kv = full_attention_layers * seq_len * 1024 * int8_bytes
                           = 10 * 131072 * 1024 * 1
                           = 1280 MiB
                           = 1.25 GiB
```

所以 Qwen 的常驻内存是：

```text
resident_without_conv_current = 1280 + 30
                              = 1310 MiB
                              = 1.279296875 GiB

resident_with_conv_current = 1280 + 30 + 1.40625
                           = 1311.40625 MiB
                           = 1.280670166015625 GiB
```

如果 full-attention KV 也卸载，则 Qwen 常驻内存可降到：

```text
resident_linear_only = 30 + 1.40625
                     = 31.40625 MiB
```

### 外存：固定完整 128k prefix

如果只恢复一个确定的完整 128k prefix，linear attention 只需要最终 recurrent state，而不是 1024 份 checkpoint：

```text
external_exact_recurrent = 30 MiB
external_exact_conv      = 1.40625 MiB
external_exact_linear    = 31.40625 MiB
```

如果 full-attention KV 也卸载：

```text
external_exact_with_full_kv = 31.40625 + 1280
                            = 1311.40625 MiB
                            = 1.280670166015625 GiB
```

这个口径容量很小，但只能瞬时恢复这一个完整 prefix；它不能支持任意 128-token 中间前缀命中。

### 外存：ckpt128 任意边界可恢复

要让任意 128-token checkpoint 命中都不需要 replay，外存需要保存每个 checkpoint 的 recurrent state：

```text
external_recurrent_ckpt128 = linear_layers * checkpoints * recurrent_elements * bf16_bytes
                           = 30 * 1024 * 524288 * 2
                           = 30720 MiB
                           = 30.0 GiB
```

如果 conv state 也要瞬时恢复：

```text
external_conv_ckpt128 = linear_layers * checkpoints * conv_dim * (conv_kernel_size - 1) * bf16_bytes
                      = 30 * 1024 * 8192 * 3 * 2
                      = 1440 MiB
                      = 1.40625 GiB
```

因此：

```text
external_linear_ckpt128 = 30720 + 1440
                        = 32160 MiB
                        = 31.40625 GiB
```

如果 full-attention KV 也卸载：

```text
external_ckpt128_with_full_kv = 32160 + 1280
                              = 33440 MiB
                              = 32.65625 GiB
```

### 单次命中恢复读回量

如果 full-attention KV 常驻，任意 checkpoint 命中只需要读回一份 linear/conv state：

```text
restore_read_linear_state = 30 + 1.40625
                          = 31.40625 MiB
```

如果 full-attention KV 也卸载，最坏恢复到 128k prefix 还要读回 full-attention KV：

```text
restore_read_with_full_kv = 31.40625 + 1280
                          = 1311.40625 MiB
                          = 1.280670166015625 GiB
```

## 结果表

### 固定完整 128k prefix 恢复

| 模型 | 常驻内存 | 外存 |
| --- | ---: | ---: |
| DSV4-nano，CSA 外存，window 常驻 | `1.5625 MiB`，带 compressor state 为 `7.203125 MiB` | `309.9375 MiB` |
| DSV4-nano，整份 compressed cache 外存 | `0-7.203125 MiB` | `311.5 MiB` |
| Qwen3.5，linear state 外存，full KV 常驻 | `1311.40625 MiB = 1.28067 GiB` | `31.40625 MiB` |
| Qwen3.5，linear state + full KV 都外存 | `31.40625 MiB` | `1311.40625 MiB = 1.28067 GiB` |

### ckpt128 任意边界可恢复

| 模型 | 常驻内存 | 外存 |
| --- | ---: | ---: |
| DSV4-nano，CSA 外存，当前 window 常驻 | `1.5625 MiB`，带 compressor state 为 `7.203125 MiB` | `1909.9375 MiB = 1.86517 GiB` |
| Qwen3.5，linear state 外存，full KV 常驻 | `1311.40625 MiB = 1.28067 GiB` | `32160 MiB = 31.40625 GiB` |
| Qwen3.5，linear state + full KV 都外存 | `31.40625 MiB` | `33440 MiB = 32.65625 GiB` |

### ckpt128 单次命中读回量

| 模型 | 单次恢复读回量 |
| --- | ---: |
| DSV4-nano，compressed backing 只恢复 offset/pointer | `7.203125 MiB` |
| DSV4-nano，compressed cache 也物化回 GPU，128k 最坏 | `317.140625 MiB` |
| Qwen3.5，full KV 常驻，只读回 linear/conv state | `31.40625 MiB` |
| Qwen3.5，full KV 也外存，128k 最坏 | `1311.40625 MiB = 1.28067 GiB` |

## 恢复策略结论

如果目标是“prefix cache 命中后不 replay、立刻恢复可继续 decode”的 ckpt128 语义：

- DSV4-nano 的外存池约 `1.87 GiB`，常驻内存可以压到 `7.21 MiB` 量级；命中时最小读回也是 `7.21 MiB`。
- Qwen3.5 如果只卸载 linear state，外存池约 `31.41 GiB`，常驻仍有 `1.28 GiB` full-attention INT8 KV；命中读回 `31.41 MiB`。
- Qwen3.5 如果连 full-attention KV 也卸载，常驻可以降到 `31.41 MiB`，但外存池变成 `32.66 GiB`，128k 最坏命中读回 `1.28 GiB`。

所以，真正卡“瞬时可恢复”的不是容量能不能放下，而是恢复带宽：

- DSV4-nano 的 `~1.87 GiB` ckpt128 backing 还可以考虑 CPU RAM / pinned memory + 异步恢复；如果只恢复 offset/pointer，单次读回很小。
- Qwen3.5 的 `31-33 GiB` backing 如果在 NVMe 上，容量虽然能放，但 cache 池会很重；要接近瞬时，至少得把热 checkpoint 放在内存级外存里，并且做分层/分块预取。
