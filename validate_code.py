
from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/validation'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(1)
    from rttp_grid.audit import run_preflight_audit
    run_preflight_audit(args.output / 'audits')
    suite = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], text=True, capture_output=True)
    (args.output / 'tests.txt').write_text(suite.stdout + suite.stderr)
    print(suite.stdout + suite.stderr)
    report = {
        'python': platform.python_version(), 'platform': platform.system(), 'device': 'cpu',
        'dependencies': {name: importlib.metadata.version(name) for name in ['torch','torchvision','datasets','numpy','pandas','scipy','matplotlib','tqdm','pillow']},
        'numerical_preflight': 'passed', 'unit_tests': 'passed' if suite.returncode == 0 else 'failed',
        'scope': 'Offline numerical and packaging validation. Does not rerun dataset training or reproduce GPU timing.'
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    raise SystemExit(suite.returncode)


if __name__ == '__main__':
    main()
