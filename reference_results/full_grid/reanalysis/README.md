# Paper Reanalysis Bundle

This directory contains derived, paper-facing tables and figures from the completed `rttp_full_grid` run.
No experiment is recomputed here.

## Main Use

- Use `tables/sgd_h40_main_table.csv` for the main scaled SGD result.
- Use `tables/frozen_h20_main_table.csv` and `tables/sgd_frozen_h40_diagnostic.csv` for the evolving-dynamics claim.
- Use `tables/adam_h20_component_table.csv` and `tables/adam_validated_horizons.csv` for the adaptive optimizer section.
- Use `tables/adam_rms_layerwise_resnet_h20.csv` for the ResNet RMS-memory nuance.

## Claim Summary

| claim | evidence | table |
| --- | --- | --- |
| SGD scales across datasets and architectures | All four SGD blocks validate through H=40 with endpoint error < 0.04 and cosine > 0.999. | sgd_h40_main_table.csv |
| Chronological tangent propagation matters | At H=20, frozen endpoint errors are consistently larger; at H=40, ResNet-18 frozen SGD diverges/overflows. | frozen_h20_main_table.csv; sgd_frozen_h40_diagnostic.csv |
| Adam supports the state-space response principle | Adam m is strong on SmallCNN and meaningful on ResNet-18; RMS channels show component-dependent local regimes. | adam_h20_component_table.csv; adam_validated_horizons.csv |
| Adam RMS response is heterogeneous | ResNet-18 later blocks validate better than stem/BN directions. | adam_rms_layerwise_resnet_h20.csv |

## Figures

| figure | description | path_png | path_pdf |
| --- | --- | --- | --- |
| fig_sgd_timevarying_endpoint_error | SGD time-varying response | reference_results/full_grid/reanalysis/figures/fig_sgd_timevarying_endpoint_error.png | reference_results/full_grid/reanalysis/figures/fig_sgd_timevarying_endpoint_error.pdf |
| fig_sgd_timevarying_endpoint_cosine | SGD time-varying response | reference_results/full_grid/reanalysis/figures/fig_sgd_timevarying_endpoint_cosine.png | reference_results/full_grid/reanalysis/figures/fig_sgd_timevarying_endpoint_cosine.pdf |
| fig_sgd_frozen_vs_timevarying_endpoint_error | SGD frozen versus time-varying | reference_results/full_grid/reanalysis/figures/fig_sgd_frozen_vs_timevarying_endpoint_error.png | reference_results/full_grid/reanalysis/figures/fig_sgd_frozen_vs_timevarying_endpoint_error.pdf |
| fig_sgd_frozen_vs_timevarying_h20_bar | SGD H=20 frozen comparison | reference_results/full_grid/reanalysis/figures/fig_sgd_frozen_vs_timevarying_h20_bar.png | reference_results/full_grid/reanalysis/figures/fig_sgd_frozen_vs_timevarying_h20_bar.pdf |
| fig_adam_h20_endpoint_error_by_component | Adam H=20 response by memory component | reference_results/full_grid/reanalysis/figures/fig_adam_h20_endpoint_error_by_component.png | reference_results/full_grid/reanalysis/figures/fig_adam_h20_endpoint_error_by_component.pdf |
| fig_adam_validated_horizon_heatmap | Adam strong-regime maximum horizon | reference_results/full_grid/reanalysis/figures/fig_adam_validated_horizon_heatmap.png | reference_results/full_grid/reanalysis/figures/fig_adam_validated_horizon_heatmap.pdf |
| fig_adam_resnet_rms_layerwise_h20 | Adam ResNet-18 RMS layerwise response at H=20 | reference_results/full_grid/reanalysis/figures/fig_adam_resnet_rms_layerwise_h20.png | reference_results/full_grid/reanalysis/figures/fig_adam_resnet_rms_layerwise_h20.pdf |

## Caveat

The original quality report marks `finite_key_metrics=False` because the frozen-boundary ResNet-18 heavy-ball baseline overflows at `H=40`. All time-varying RTTP rows are finite. Treat frozen overflow as a diagnostic result, not as a failure of the proposed response model.