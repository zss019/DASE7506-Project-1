# MP1 checkpoint bundle (v3)

Frozen predictor for DASE7506 protocol `7506-mp1-wt2-v2`. Evaluation does **not** require retraining.

| Item | Value |
|---|---|
| File | `checkpoint.pt` |
| SHA-256 | `de648427b2ca17472beafc9d4bfae69772eb52adae1f6824d29fb60464dfd032` |
| Size | 26,035,093 bytes (26.0 MiB) |
| Implementation | `student` (`student.py` in the code repository) |
| Config | `config.json` (width 256, depth 8, GELU, RoPE, RMSNorm, dropout 0.2) |
| Reported full-test FP32 BPB | **1.48544** |

## Evaluate

Use the **matching code repository** (same `student.py` as SHA-256 `aa47af96d8487592a67441e0df218717bcee482ec3da60d40fb6745f45cbe9cd`) and the course scorer. From `code/`:

```bash
python evaluate.py --checkpoint /path/to/checkpoint.pt --device cpu --precision fp32 --split test --threads 4
```

`evaluate.py` reconstructs the model from `implementation` + `config` stored inside the checkpoint, then loads the weights. Do not pass a different `student.py`.

Verify the file before scoring:

```bash
sha256sum checkpoint.pt
# de648427b2ca17472beafc9d4bfae69772eb52adae1f6824d29fb60464dfd032
```

## What is inside the checkpoint

`protocol`, `implementation`, `config`, `model` state dict, `seed` 17, `train_tokens` 229376000. No optimizer state. RoPE cos/sin tables are not stored; they are rebuilt at load time.
