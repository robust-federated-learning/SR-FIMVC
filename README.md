# SR-FIMVC Handwritten Experiment

This directory is a clean, self-contained Handwritten-only experiment package.
It contains the current extended implementation with structured Z updates,
residual-weighted completion, consensus alignment, and direct Laplacian-guided
completion.

## Requirements

- MATLAB
- Statistics and Machine Learning Toolbox

The default projected Z solver does not require Optimization Toolbox.

## Run

1. Open this directory in MATLAB.
2. Open `run_Handwritten.m`.
3. Leave `FORMAL_MODE = false` for the quick validation run.
4. Click Run.
5. After validation succeeds, set `FORMAL_MODE = true` for the formal study.

Quick mode uses one run, three rounds, and 30 anchors. Formal mode uses ten
runs, 200 rounds, and 300 anchors. Both modes evaluate PER values 0, 0.1, 0.3,
and 0.5 without using ground-truth labels for model selection.

Results and checkpoints are written to the `results` directory. The summary is
saved in both MAT and CSV formats.

## Entry Point

Run only `run_Handwritten.m`. The other MATLAB files are algorithm dependencies
and do not need to be executed directly.
