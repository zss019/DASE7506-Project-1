"""Run named training jobs on the GPU (several concurrently) and summarise validation BPB.

Each job: name, config overrides on configs/student_modern.json, and train.py arguments.
Usage: python tools/sweep.py sweep_file.json [--parallel 3]
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(job, out_dir):
    run_dir = out_dir / job['name']
    if (run_dir / 'metrics.json').exists():
        return job['name']
    cfg = json.loads((ROOT / 'configs/student_modern.json').read_text()) | job.get('config', {})
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / f"{job['name']}.config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + '\n')
    subprocess.run(['rm', '-rf', str(run_dir)])
    cmd = [sys.executable, 'train.py', '--implementation', 'student', '--config', str(cfg_path),
           '--device', 'cuda', '--compile', '--run-dir', str(run_dir), *map(str, job.get('args', []))]
    with open(out_dir / f"{job['name']}.log", 'w') as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return job['name']


def summary(out_dir, names):
    for name in names:
        m = json.loads((out_dir / name / 'metrics.json').read_text())
        curve = ' '.join(f"{v['step']}:{v['bpb']:.4f}" + (f"/{v['ema_bpb']:.4f}" if 'ema_bpb' in v else '')
                         for v in m['validation_history'])
        live = m['live_validation']['bpb'] if m.get('live_validation') else None
        print(f"{name:28s} final {m['validation']['bpb']:.4f} live {live and round(live, 4)} "
              f"params {m['parameters']} train {m['train_seconds']:.0f}s | {curve}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sweep', type=Path)
    p.add_argument('--parallel', type=int, default=3)
    args = p.parse_args()
    jobs = json.loads(args.sweep.read_text())
    out_dir = ROOT / 'runs' / args.sweep.stem
    with ThreadPoolExecutor(args.parallel) as pool:
        names = list(pool.map(lambda j: run(j, out_dir), jobs))
    summary(out_dir, names)


if __name__ == '__main__':
    main()
