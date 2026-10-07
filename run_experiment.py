
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', choices=['grid', 'amortization', 'surrogates', 'selection'])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, help='Override output root; relative to the working directory.')
    args = p.parse_args()
    payload = json.loads(args.config.read_text())
    if args.output:
        payload.update(root_name=str(args.output.resolve()), use_google_drive=False)
    if args.experiment == 'grid':
        from rttp_grid.config import GridConfig
        from rttp_grid.runner import run_grid
        cfg, run = GridConfig(**payload), run_grid
    elif args.experiment == 'amortization':
        from rttp_grid.amortization import AmortizationConfig, run_amortization_experiment
        cfg, run = AmortizationConfig(**payload), run_amortization_experiment
    elif args.experiment == 'surrogates':
        from rttp_grid.surrogate_baselines import SurrogateBaselineConfig, run_surrogate_baseline_experiment
        cfg, run = SurrogateBaselineConfig(**payload), run_surrogate_baseline_experiment
    else:
        from rttp_grid.candidate_selection import CandidateSelectionConfig, run_candidate_selection_experiment
        cfg, run = CandidateSelectionConfig(**payload), run_candidate_selection_experiment
    root = run(cfg)
    print(json.dumps({'output_root': str(root)}, indent=2))
    if args.experiment == 'grid' and cfg.run_summary and cfg.run_response:
        import pandas as pd
        report = pd.read_csv(root / 'tables/quality_report.csv')
        failed = report[report.blocking & ~report.passed]
        if len(failed):
            raise SystemExit('Blocking quality checks failed: ' + ', '.join(failed.check))


if __name__ == '__main__':
    main()
