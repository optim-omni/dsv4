# DeepSeek V4 Nano 35A2.6B

This variant keeps the same 35B total architecture as `dsv4-nano-35a3`, but reduces active routed experts from 8 to 4.

Remote run on `cudo-h100`:

```text
topk=4  total=35.086756432B  active=2.585094736B
exact_total=35086756432
exact_active=2585094736
```

Minimal config delta from 35A3:

```json
{
  "num_experts_per_tok": 4,
  "n_activated_experts": 4
}
```

Everything else stays the same: `hidden_size=2048`, `num_hidden_layers=40`, `n_routed_experts=256`, `moe_intermediate_size=512`, `vocab_size=248320`, DeepSeek V4 MLA/compression/indexer/Hyper-Connections, and one MTP layer.
