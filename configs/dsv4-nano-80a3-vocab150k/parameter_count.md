# dsv4-nano-80a3-vocab150k

80B-range total with E=640 and A3-class active after reducing vocab to 150016.

Remote count on `cudo-h100`:

```text
total=84.244223824B
active=2.990069584B
exact_total=84244223824
exact_active=2990069584
```

Key config:

```json
{
  "vocab_size": 150016,
  "n_routed_experts": 640,
  "num_experts_per_tok": 10,
  "n_activated_experts": 10,
  "hidden_size": 2048,
  "num_hidden_layers": 40,
  "moe_intermediate_size": 512
}
```

`150000` is rounded up to `150016` so `vocab_size % 128 == 0`. Token ids use DeepSeek-style `bos_token_id=0` and `eos_token_id=1` because Qwen's previous `248044` EOS is outside the 150016 vocabulary.
