# DASE7506 MP1 — Small Language Model Challenge

Individual submission. Protocol `7506-mp1-wt2-v2`. All commands below run from `code/` after the virtual environment is active.

## Reported score

| Item | Value |
|---|---|
| **Full-test FP32 BPB** | **1.48544** |
| Evaluator JSON (`bpb`) | 1.485441599903059 |
| Validation FP32 BPB | 1.46716 |
| Token perplexity (test) | 22.314 |
| Parameters | 6,820,096 |
| Seed | 17 |
| Checkpoint SHA-256 | `de648427b2ca17472beafc9d4bfae69772eb52adae1f6824d29fb60464dfd032` |
| `student.py` SHA-256 | `aa47af96d8487592a67441e0df218717bcee482ec3da60d40fb6745f45cbe9cd` |
| Scorer SHA-256 | `128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d` (unchanged) |

The leaderboard shows five decimal places. Submit **1.48544**. The JSON value is the exact output of the supplied `evaluate.py` on the frozen checkpoint, CPU, FP32, full test split.

Code repository: https://github.com/zss019/DASE7506-Project-1  
Score issue: https://github.com/xudongwu-0/xudongwu-0.github.io/issues/107

**Report (≤10 pages):** [`report/MP1_report.pdf`](report/MP1_report.pdf) (source [`report/MP1_report.md`](report/MP1_report.md)). Method, matched-token comparison, ablation, and critical analysis as required by the guide.

## 1. Install

Python **3.12**. From this repository:

```bash
cd code
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
```

Install **one** PyTorch build, then the remaining pins:

```bash
# CPU (required for ranked evaluation)
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU training only — use this instead of the CPU wheel
# python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

No API key, pretrained weights, or extra dataset download. Data and the BPE-2048 tokenizer are in `code/data/`.

## 2. Evaluate the frozen checkpoint (no retraining)

Place the matching `checkpoint.pt` (SHA-256 above) on disk. Then:

```bash
python evaluate.py --checkpoint /path/to/checkpoint.pt --device cpu --precision fp32 --split test --threads 4
```

The printed `bpb` is the submission score. This run also writes `test_cpu_fp32.json` next to the checkpoint unless `--output` is set.

The files to upload as the public bundle are in `checkpoint-bundle/` (see that folder’s README). Weights are not stored in git. After the GitHub Release is published, download:

```
https://github.com/zss019/DASE7506-Project-1/releases/download/v3/checkpoint.pt
```

SHA-256 must remain `de648427b2ca17472beafc9d4bfae69772eb52adae1f6824d29fb60464dfd032`.

## 3. Train the submitted model

Training is not required to score the bundle. To rebuild the same predictor from random initialization:

```bash
python train.py --implementation student \
  --config configs/final_v3.json \
  --device cuda --compile \
  --seed 17 --steps 7000 --batch-size 128 \
  --lr 0.004 --warmup 300 --min-lr-ratio 0.02 \
  --weight-decay 0.3 --wd-exclude-1d \
  --ema 0.9995 --eval-every 1400 \
  --run-dir runs/final-v3
```

On CPU, drop `--compile` and use `--device cpu --precision fp32`. The saved `checkpoint.pt` stores the EMA weights. Then evaluate as in §2.

`configs/final_v3.json`:

```json
{
  "vocab": 2048,
  "width": 256,
  "heads": 4,
  "depth": 8,
  "context": 256,
  "norm": "rms",
  "pos": "rope",
  "mlp": "gelu",
  "bias": false,
  "dropout": 0.2
}
```

Processed training targets: `7000 × 128 × 256 = 229,376,000`.

## 4. Evaluation budget (same frozen predictor)

Measured with the supplied scorer, CPU FP32, 4 threads, protocol windows of 256 tokens.

| Limit | This predictor | Cap |
|---|---:|---:|
| CPU scoring time vs local baseline | about **3.4–4.1×** (shared-machine load varies; architecture is ~3.4× on a quieter host) | ≤ 5× |
| Peak evaluation RSS | **2.10 GiB** | ≤ 4 GiB |
| Uncompressed inference assets | **26.0 MiB** checkpoint / 28.0 MiB of FP32 parameters | ≤ 64 MiB |

Reproduced classroom baseline on this machine: test BPB **2.10126** (`runs/baseline/`, seed 17, 1,200 steps). Official reference is approximately 2.10.

## 5. Method

A GPT decoder with the same interfaces as the starter (`forward` logits, `predict_log_probs` normalized log-probabilities, `context=256`, vocab 2048). Changes relative to `model.py`:

- **RMSNorm** instead of LayerNorm; **no linear biases**.
- **RoPE** instead of learned absolute position embeddings.
- **GELU MLP** with 4× hidden width (same family as the baseline FFN).
- **Dropout 0.2** on embeddings, attention, and residual branches (training only).
- **Scaled residual-output init** \(0.02 / \sqrt{2\cdot\mathrm{depth}}\).
- Tied input/output embeddings (already in the baseline).
- Training: larger model and longer run, cosine decay to 2% of peak LR, AdamW weight decay 0.3 with 1-D parameters excluded, **EMA 0.9995** of the weights used for validation and the saved checkpoint.

Settings were chosen on **validation** only. The test split was scored after each freeze, never used to pick hyperparameters.

## 6. What must stay unchanged

`common.py`, `evaluate.py`, `tests/test_contract.py`, `data/`, `configs/baseline.json`, and `model.py` are the classroom baseline and scorer. Do not modify them when reproducing. Student code is `student.py`, `train.py`, and `configs/final_v3.json`.

## 7. Training and search cost

All development training used one NVIDIA GeForce RTX 3080 Ti (12 GiB). Search included architecture ablations, size/dropout/LR sweeps, GELU vs SwiGLU, and weight-decay confirmation across seeds 17/18/19. Cumulative GPU training time is on the order of **8 GPU-hours** of wall-clock job time (many runs overlapped on one GPU). CPU time was used for the official baseline, timing, and all ranked evaluations.

A 10-page report (method, matched-token comparison, ablation, and cost/quality trade-offs) will sit in this repository for the 30 September final submission.

## 8. AI assistance

Substantive help came from **Cursor Grok 4.6** (and earlier turns of the same coding agent) for reading the starter, implementing `student.py` / training flags, running sweeps, measuring the budget, and drafting this README. The submitted predictor, commands, hashes, and scores were executed locally with the supplied evaluator. I take responsibility for the implementation and the reported BPB.

Reused work: the starter GPT, trainer, scorer, WikiText-2 splits, and BPE-2048 tokenizer from the course package; RMSNorm / RoPE / residual scaling follow standard GPT-2 / Llama-style practice rather than a copied third-party checkpoint.

## 9. Data attribution

WikiText-2: Merity et al., [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). Text by Wikipedia contributors; upstream [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and [GFDL](https://www.gnu.org/licenses/fdl-1.3.html). Tokenizer fitted only on the supplied training split. See `code/README.md` §6.
