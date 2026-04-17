# Modeling Workflow

Start here if you want the current modeling layout without tracing imports.

## Package Layout

- `official/`
  - Durable modeling entrypoints that define the canonical component-model outputs
- `experiments/`
  - Structured sweeps, tuning runs, and exploratory model families that are useful but not part of the official reporting contract
- `common/`
  - Narrow shared helpers reused by multiple active entrypoints

## Official Entrypoints

1. `official/component_single_text_calibrated.py`
   - Purpose: fit the locked official single-label component model
   - Reads: `odi_component_single_label_cases` and `odi_component_text_sidecar`
   - Writes: `component_single_label_official_manifest.json` plus official holdout, class, confusion, and calibration artifacts

2. `official/component_multi_routing.py`
   - Purpose: fit the locked official multi-label routing model
   - Reads: `odi_component_multilabel_cases`
   - Writes: `component_multilabel_official_manifest.json` plus official split, metrics, and label-metrics artifacts

3. `../reporting/update_component_readme.py`
   - Purpose: validate the official manifests and publish the README summary block
   - Reads: the two official manifests only
   - Writes: generated README benchmark block plus official benchmark summary artifacts

## Experiment Entrypoints

1. `experiments/component_single_structured_tuning.py`
   - Focused Optuna search for the structured single-label benchmark branch

2. `experiments/component_single_structured_baseline.py`
   - Structured-only single-label benchmark and baselines

3. `experiments/component_feature_wave1.py`
   - Structured feature-family sweeps beyond the locked official contract

4. `experiments/component_text_wave2.py`
   - Text and fusion family comparisons used during exploration

## Shared Helpers

- `common/core.py`
  - Split policy, feature manifests, scoring helpers, and model-stage feature derivation for lean case tables
- `common/multilabel.py`
  - Shared CatBoost multi-label helpers
- `common/text_fusion.py`
  - Shared text/fusion helpers used by the experiment and official single-label flows

## Reading Order

1. Read the official entrypoint you plan to run
2. Check its input/output artifact names near the top of the file
3. Open the matching `common/` helper only if you need implementation detail
4. Use `experiments/` only when you intentionally want exploratory or cloud-heavy work
