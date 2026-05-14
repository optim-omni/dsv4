# dsv4-nano-35a3-vocab150k

35B total, A3-class active after reducing vocab to 150016; top-k raised from 8 to 10 to keep active params near 3B.

Remote count on `cudo-h100`:

```text
total=34.685623888B
active=2.957811280B
exact_total=34685623888
exact_active=2957811280
```

Key config:

```json
{
  "vocab_size": 150016,
  "n_routed_experts": 256,
  "num_experts_per_tok": 10,
  "n_activated_experts": 10,
  "hidden_size": 2048,
  "num_hidden_layers": 40,
  "moe_intermediate_size": 512
}
```

`150000` is rounded up to `150016` so `vocab_size % 128 == 0`. Token ids use DeepSeek-style `bos_token_id=0` and `eos_token_id=1` because Qwen's previous `248044` EOS is outside the 150016 vocabulary.
