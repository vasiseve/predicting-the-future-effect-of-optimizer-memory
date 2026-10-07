# Mapping to the supplied v4 draft

Figure and table numbering below refers to **Predicting the Future Effect of Optimizer Memory**, version 4.

| Draft item | Experiment implementation | Evidence / caveat |
| --- | --- | --- |
| Section 3, tangent propagation | `rttp_grid/runner.py`: `sgd_tangent_step`, `adam_tangent_step`, `grad_hvp`, initial/perturbed Adam states | Production functions checked against multi-step finite differences |
| Figure 1; Tables 1 and 3 | `run_experiment.py grid --config configs/full_grid.json` | Archived `reference_results/full_grid/reanalysis/tables/sgd_h40_main_table.csv` and `adam_h20_component_table.csv`; all eight Table 1 endpoint errors match after rounding |
| Figure 2, magnitude/horizon regime | Full grid, Adam `rms_global`, ResNet-18, Split CIFAR-10 | `response_summary_by_scale.csv`; portable reference builder uses response learning rate 0.0003 |
| Figure 3 and Table 2, chronological versus frozen | Full grid, `time_varying` and `frozen_boundary` | See `frozen_comparison_by_scale.csv`, `frozen_best_comparison.csv`, terminal operating points from reanalysis. Avoid mixing boundary/response learning rates or different scale selections |
| Figure 4, dense log-v versus RMS-relative | `legacy/adam_paper_aligned_response.ipynb` and `legacy/adam_paper_aligned_rms_response_v2.ipynb` | Historical measurement CSVs in `reference_results/legacy/`; exact paired plotting selection is not established by the full-grid config |
| Figure 5, common-horizon Adam generalization | Full grid, H=10, first moment scale 0.01 and RMS-global scale 0.1 | Use fixed response LR 0.0003 for CIFAR-10 and 0.0001 for TinyImageNet rather than per-architecture best selection |
| Figure 6, retention/adaptation policies | `legacy/optimizer_response_cifar10_rttp_paper_drive_resumable.ipynb` (SmallCNN), `legacy/optimizer_response_cifar10_rttp_sgd_resnet18_paper_colab.ipynb` (ResNet-18) | Policy summary/raw CSVs included in `reference_results/legacy/`. The newer candidate-selection experiment evaluates ranking metrics and is distinct from this policy figure |
| Section 4.6 / Table 4, surrogate comparison | `run_experiment.py surrogates --config configs/surrogate_baselines.json` | Template based on the shipped dataclass defaults. Exact archived per-run config and timing measurements were not located; fresh runtime may differ |
| Table 5, candidate-count amortization | `run_experiment.py amortization --config configs/amortization.json` | Includes candidate counts 5, 10, 20, 50, 100, H=40 and rank 8. Exact original per-run timing config was not located; timing depends on hardware |
| Additional prospective ranking evaluation | `run_experiment.py selection --config configs/candidate_selection.json` | Auxiliary experiment, not a replacement for Figure 6 |

## Selection and uncertainty

The full-grid runner's `best_scale_summary.csv` selects scale by centered finite-difference error separately at each horizon. The paper-facing reanalysis selects operating points with endpoint error; terminal-selected figures hold the selected scale/LR fixed across horizons. These are different summaries. The regeneration scripts preserve their original selection rules.

Seed-level aggregation averages directions within seed before computing intervals across five seeds. The original reanalysis uses Student-t 95% intervals. Do not treat individual directions as independent seeds. The original Figure 2 caption/text and heatmap appear to describe different trends with perturbation magnitude; this package preserves the measurements and does not resolve manuscript interpretation.

## Validation boundary

Archived evidence is included for traceability. Numerical tests validate derivatives on small deterministic systems and test the production response functions. They do not certify exact agreement of a fresh CUDA/data run with every manuscript value. Dataset revisions were not recorded in the original run, and GPU model/total full-grid runtime were not recorded in the supplied manifest; these details cannot be reconstructed reliably.
