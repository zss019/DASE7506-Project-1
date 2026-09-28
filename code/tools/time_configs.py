"""Measure CPU FP32 validation scoring time of untrained configs relative to the baseline.

Usage: python tools/time_configs.py [--base configs/x.json] [--split test] width:depth:heads[:mlp_mult] ...
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from common import ROOT, load_data, setup
from evaluate import score
import model as baseline
import student


def timed(m, data, split, repeats=2):
    return min(score(m, *data[split], torch.device('cpu'), 'fp32')['seconds'] for _ in range(repeats))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=ROOT / 'configs/student_modern.json')
    p.add_argument('--split', default='validation')
    p.add_argument('specs', nargs='+')
    args = p.parse_args()
    setup('cpu', 'fp32', 4)
    data = load_data()
    base_cfg = json.loads((ROOT / 'configs/baseline.json').read_text())
    modern = json.loads(args.base.read_text())
    ref = baseline.build_model(base_cfg)
    for spec in args.specs:
        parts = spec.split(':')
        cfg = modern | dict(width=int(parts[0]), depth=int(parts[1]), heads=int(parts[2]))
        if len(parts) > 3:
            cfg['mlp_mult'] = float(parts[3])
        m = student.build_model(cfg)
        base_seconds = timed(ref, data, args.split)
        seconds = timed(m, data, args.split)
        params = sum(p.numel() for p in m.parameters())
        print(json.dumps(dict(spec=spec, params=params, fp32_mib=round(params * 4 / 2**20, 1),
                              seconds=round(seconds, 2), baseline_seconds=round(base_seconds, 2),
                              ratio=round(seconds / base_seconds, 2))), flush=True)


if __name__ == '__main__':
    main()
