# dsv4-nano-35a2_6b-vocab150k

35B total, A2.6-class active after reducing vocab to 150016; top-k raised from 4 to 7 to keep active params near 2.6B.

Remote count on `cudo-h100`:

```text
total=34.684273744B
active=2.569536592B
exact_total=34684273744
exact_active=2569536592
```

Key config:

```json
{
  "vocab_size": 150016,
  "n_routed_experts": 256,
  "num_experts_per_tok": 7,
  "n_activated_experts": 7,
  "hidden_size": 2048,
  "num_hidden_layers": 40,
  "moe_intermediate_size": 512
}
```

`150000` is rounded up to `150016` so `vocab_size % 128 == 0`. Token ids use DeepSeek-style `bos_token_id=0` and `eos_token_id=1` because Qwen's previous `248044` EOS is outside the 150016 vocabulary.
