# Predicting the Future Effect of Optimizer Memory

## Running the code in Google Colab

### 1. Upload the code folder to Google Drive

Extract the code package ZIP on your computer. Upload the entire extracted `predicting-the-future-effect-of-optimizer-memory` folder into **My Drive**, keeping its subfolders intact.

The first notebook cell expects this location:

```text
My Drive/predicting-the-future-effect-of-optimizer-memory/
```

In Colab, this becomes:

```text
/content/drive/MyDrive/predicting-the-future-effect-of-optimizer-memory/
```

Upload the extracted folder, not just the ZIP or an individual notebook. The folder must contain `requirements.txt`, `rttp_grid/`, `configs/` and `notebooks/`. If you use another location, edit `CODEBASE_DIR` in the first notebook cell.

### 2. Open the main notebook and select a GPU

Open [Google Colab](https://colab.research.google.com/). Select **File → Upload notebook** and upload:

```text
notebooks/rttp_full_grid_colab.ipynb
```

Select **Runtime → Change runtime type**, choose a **GPU** hardware accelerator, and save. These experiments use PyTorch with CPU or CUDA; choose a GPU rather than a TPU. CPU execution is suitable for numerical checks and small runs, but is slow for the full grid.

Python 3.12 is the tested environment. Dependency versions are pinned in `requirements-validated.txt`; the versions recorded with the original paper run are separately listed in `requirements-paper.txt`.

### 3. Run the setup cell

Run the first cell and authorize access to your Google Drive. It mounts Drive, selects the code folder, and installs the dependencies. Wait for installation to finish before proceeding.

If Colab requests a session restart after installation, restart it and rerun the setup cell before continuing. To check GPU access, run this in a new cell:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

The second line should print `True` for a GPU run. The experiment also records its selected device in the output manifest.

### 4. Start with the smoke run

In the configuration cell, keep:

```python
RUN_MODE = 'smoke'
FORCE = False
RUN_PREFLIGHT_AUDIT = True
RUN_DATASET_PROBE = True
RUN_FROZEN_BASELINE = True
```

Run the remaining cells in order. The notebook smoke configuration uses Split CIFAR-10, SmallCNN, one seed, one boundary-training epoch, SGD and Adam, and prediction horizons 1 and 3. It downloads the public dataset automatically. This checks execution; it does not reproduce the full paper experiment.

The default notebook smoke output is saved in:

```text
My Drive/predicting-the-future-effect-of-optimizer-memory/rttp_full_grid_smoke/
```

The final cell displays tables and audit summaries. Open `tables/quality_report.csv` and confirm that every row with `blocking=True` has `passed=True`. Also inspect `audits/preflight_audit_summary.csv` and `diagnostics/dataset_probe.csv`.

### 5. Run the full grid or a selected subset

After the smoke run succeeds, set:

```python
RUN_MODE = 'full'
ROOT_NAME = 'results/rttp_full_grid'
FORCE = False
```

Rerun the configuration cells and then the experiment cell. Full mode uses both datasets, both architectures, both optimizers and seeds 0–4. The `DATASETS`, `ARCHITECTURES`, `OPTIMIZERS` and `SEEDS` variables in the notebook apply only to **custom** mode.

For a smaller selected run, use:

```python
RUN_MODE = 'custom'
DATASETS = ('split_cifar10',)
ARCHITECTURES = ('smallcnn',)
OPTIMIZERS = ('heavy_ball', 'adam')
SEEDS = (0,)
ROOT_NAME = 'results/cifar_smallcnn_seed0'
FORCE = False
```

Custom mode retains the full configuration's training and response settings for the selected blocks. Use a distinct `ROOT_NAME` whenever you change scientific settings. The runner rejects incompatible cached settings in an existing output folder.

### 6. Run the additional experiments

Open the appropriate notebook, run its setup cell, edit its configuration cell, and execute the remaining cells in order.

| Notebook in `notebooks/` | Experiment | Default output under the code folder |
| --- | --- | --- |
| `rttp_cifar_surrogate_baselines_colab.ipynb` | Compare trajectory surrogates | `results/rttp_cifar_surrogate_baselines/` |
| `rttp_cifar_amortized_cost_timing_colab.ipynb` | Replay versus shared-response timing | `results/rttp_amortization_cifar/` |
| `rttp_cifar_candidate_selection_colab.ipynb` | Rank candidate interventions | `results/rttp_cifar_candidate_selection/` |

The timing notebook defaults to candidate counts `(5, 10, 20, 50)`. Add `100` to `CANDIDATE_COUNTS` to include the largest count in the paper's timing table. Keep `FORCE_BOUNDARY = False` to reuse an existing compatible boundary checkpoint. GPU timings and memory measurements depend on the allocated hardware.

The historical notebooks in `legacy/` are separate implementations used for particular draft figures. Their code cells contain their own settings. See `docs/paper_experiment_map.md` before using them to reproduce a specific figure.

### 7. Find and resume your results

Full-grid outputs are saved in:

```text
My Drive/predicting-the-future-effect-of-optimizer-memory/results/rttp_full_grid/
```

The main files are:

- `config.json` and `manifest.json`: configuration and environment.
- `tables/response_raw.csv`: individual measurements.
- `tables/response_summary_by_scale.csv`: aggregated measurements.
- `tables/quality_report.csv`: completeness and numerical checks.
- `figures/`: generated figures when enabled.
- `cache/`: boundary checkpoints and completed response chunks.
- `archives/`: exported full-grid results.

After a Colab disconnection, reconnect, rerun setup, restore the same configuration and `ROOT_NAME`, and leave `FORCE = False`. The main grid reuses completed checkpoints and chunks. Auxiliary experiments reuse their boundary checkpoint but rerun their evaluations. Drive preserves saved outputs; computation interrupted before saving must be repeated. Colab runtime availability and limits vary; see the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).

### Optional: run the supplied JSON configurations

After the setup cell, run these commands in a new Colab cell:

```python
!python run_experiment.py grid --config configs/smoke.json
!python run_experiment.py grid --config configs/full_grid.json
```

Run the smoke command first, then run the full-grid command separately after checking its outputs. The JSON smoke configuration is smaller than the notebook smoke configuration and saves to `results/smoke/`. The JSON full-grid configuration saves to `results/full_grid/`. Use either the notebook settings or the JSON configuration consistently for a given run.

For offline numerical checks, run:

```python
!python validate_code package.py
```

To regenerate analysis from the included archived measurements without training, run:

```python
!python scripts/make_paper_reanalysis.py --root reference_results/full_grid
!python scripts/make_frozen_ci_figures.py --root reference_results/full_grid
!python scripts/build_reference_artifacts.py
```

### Troubleshooting and reproduction limits

- **Code folder not found:** check `CODEBASE_DIR` and remove any extra nested folder introduced while extracting the ZIP.
- **Import errors after installation:** restart the session if requested, then rerun setup.
- **GPU unavailable:** check the runtime accelerator and reconnect; the code otherwise uses CPU.
- **Out of memory:** run a smaller custom subset or choose a runtime with more GPU memory. Changing batch size or tangent directions changes the experiment; use a new output folder and record the changes.
- **Existing settings conflict:** choose a new `ROOT_NAME`. Use `FORCE = True` only when you intend to recompute that run.
- **Nonfinite frozen baseline:** long-horizon frozen SGD can diverge. These rows are diagnostic; nonfinite time-varying response metrics are blocking failures.

The package passed numerical tests, synthetic pipeline tests and a small real-data CIFAR-10 run with the pinned dependencies. Full-grid training and GPU timings were not rerun during packaging. Exact original per-run configurations for the two timing tables were not located; the supplied auxiliary configurations are run templates. Consult `docs/paper_experiment_map.md` for figure-specific provenance and limitations.
