"""Measure CPU FP32 validation scoring time of untrained configs relative to the baseline.

Usage: python tools/time_configs.py width:depth:heads[:mlp_mult] ...
"""
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


def timed(m, data, repeats=2):
    return min(score(m, *data['validation'], torch.device('cpu'), 'fp32')['seconds'] for _ in range(repeats))


def main():
    setup('cpu', 'fp32', 4)
    data = load_data()
    base_cfg = json.loads((ROOT / 'configs/baseline.json').read_text())
    modern = json.loads((ROOT / 'configs/student_modern.json').read_text())
    ref = baseline.build_model(base_cfg)
    for spec in sys.argv[1:]:
        parts = spec.split(':')
        cfg = modern | dict(width=int(parts[0]), depth=int(parts[1]), heads=int(parts[2]))
        if len(parts) > 3:
            cfg['mlp_mult'] = float(parts[3])
        m = student.build_model(cfg)
        base_seconds = timed(ref, data)
        seconds = timed(m, data)
        params = sum(p.numel() for p in m.parameters())
        print(json.dumps(dict(spec=spec, params=params, fp32_mib=round(params * 4 / 2**20, 1),
                              seconds=round(seconds, 2), baseline_seconds=round(base_seconds, 2),
                              ratio=round(seconds / base_seconds, 2))), flush=True)


if __name__ == '__main__':
    main()
