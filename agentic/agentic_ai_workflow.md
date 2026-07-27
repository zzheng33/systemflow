# Agentic AI Workflow for Scientific Resource Modeling

## Purpose

This workflow turns notebook-based scientific performance modeling into a human-in-the-loop agentic AI pipeline.

The goal is not to replace the scientific assumptions or the researcher. The goal is to automate the repetitive and error-prone parts of the process:

- planning benchmark matrices
- running scientific benchmark experiments
- parsing logs and power traces
- validating data quality
- fitting latency, power, and energy models
- exporting coefficient tables
- integrating coefficients into SystemFlow or another modeling layer
- generating a report for scientific review

The human researcher remains responsible for approving experiment plans, deciding which runs are valid, approving model assumptions, and accepting final coefficients.

## Scientific Context

This workflow is application-agnostic. It should support many scientific applications, including X-ray scattering, high-energy physics, microscopy, simulation workflows, scientific AI inference pipelines, and iterative numerical solvers.

The current XRS work is one application profile. In that profile, the experiment models X-ray scattering reconstruction workloads, especially ptychography. It compares two reconstruction backends:

- **PtychoPINN**: a learned neural-network inference backend.
- **Pty-chi**: a family of iterative ptychography reconstruction algorithms such as `pie`, `dm`, `lsqml`, `bh`, and `ad_ptycho`.

The workflow estimates system-level resource metrics:

- I/O latency
- inference or reconstruction latency
- GPU power
- energy
- images per joule
- GFLOPs per joule
- SystemFlow-compatible reconstruction performance

The output coefficient tables are consumed by:

- `systemflow/xrs_models.py`
- `systemflow/xrs.py`
- `XRS/performance.ipynb`

## Proposed Agents

The recommended workflow uses five agents.

```text
Experiment Planner Agent
    -> human approval

Experiment Runner Agent
    -> raw logs and power CSVs

Data & Validation Agent
    -> human approval

Modeling Agent
    -> human approval

SystemFlow Integration & Report Agent
    -> human approval
```

## 1. Experiment Planner Agent

### Responsibility

The Experiment Planner Agent converts a research question into a concrete benchmark plan.

Example research question:

```text
Compare PtychoPINN and Pty-chi on A100, H200, and MI300X for ptychography reconstruction.
```

The agent produces a structured experiment matrix.

Example output:

```yaml
backends:
  - ptychopinn
  - ptychi

gpus:
  - A100
  - H200
  - MI300X

datasets:
  - R1000
  - R2000
  - R4000
  - R8000
  - R12000
  - R16000
  - R20000
  - R26000

resolutions:
  - 64

batch_sizes:
  ptychopinn:
    - 1024
  ptychi:
    - 1000

ptychi_algorithms:
  - pie
  - dm
  - lsqml
  - bh
  - ad_ptycho

ptychi_epochs:
  - 1
  - 2
  - 4
  - 6

metrics:
  - io_latency
  - inference_latency
  - reconstruction_latency
  - power
  - energy
  - images_per_joule
  - gflops_per_joule
```

### Inputs

- Research question
- Target backend list
- GPU or accelerator list
- Dataset sizes
- Resolution
- Batch sizes
- Algorithm list
- Time or resource budget

### Outputs

- `experiment_plan.yaml`
- estimated number of runs
- expected folder layout
- expected output files
- dry-run command list

### Human Approval Gate

The human researcher approves the plan before any benchmark is run.

The approval should answer:

- Is the experiment matrix scientifically meaningful?
- Are the GPU resources acceptable?
- Are the chosen algorithms and batch sizes fair?
- Should any datasets or accelerators be added or removed?

## 2. Experiment Runner Agent

### Responsibility

The Experiment Runner Agent executes the approved experiment plan.

It should support both local execution and HPC execution. In early versions, it can operate in dry-run mode and only generate scripts.

### PtychoPINN Expected Output

```text
modeling_exp/
  A100/
    R1000/
      bs1024.log
      bs1024_power.csv
    R2000/
      bs1024.log
      bs1024_power.csv
```

### Pty-chi Expected Output

```text
pty-chi/modeling_exp/
  A100/
    R1000/
      dm/
        e1_bs1000.log
        e1_bs1000_power.csv
        e2_bs1000.log
        e2_bs1000_power.csv
```

### Inputs

- approved `experiment_plan.yaml`
- benchmark command templates
- GPU/HPC configuration
- power monitoring configuration

### Outputs

- benchmark logs
- power CSV files
- run manifest
- failed run list
- skipped run list

### Safety Rules

- Default to dry-run mode.
- Never delete old results automatically.
- Skip existing complete runs unless explicitly told to rerun.
- Write failed runs to `failed_runs.csv`.
- Attach a unique run ID to logs and power traces.
- Keep benchmark stdout, stderr, command, start time, end time, and exit code.

### Human Approval Gate

The main approval happens before running. A second approval may be required before rerunning failed experiments, because reruns consume GPU time.

## 3. Data & Validation Agent

### Responsibility

The Data & Validation Agent parses raw logs and power CSVs into structured tables, then checks data quality.

This agent is responsible for making the raw benchmark data auditable before modeling.

### PtychoPINN Parsing Targets

From `bs1024.log`, extract:

- raw images `R`
- valid or potential central scan points `V`
- grouped NN samples `S`
- batch size `B`
- group coordinates time
- write image tensor time
- model load time
- data load time
- inference time
- assembly time

From `bs1024_power.csv`, extract:

- average power
- min power
- max power
- power duration
- integrated energy
- power samples

### Pty-chi Parsing Targets

From `e*_bs*.log`, extract:

- raw images `R`
- image resolution `N`
- algorithm
- epochs
- batch size
- I/O load time
- setup time
- task setup time
- reconstruction runtime
- save time
- total time

From matching power CSV files, extract phase-window power:

- I/O power
- setup power
- reconstruction power

### Validation Checks

The agent should flag:

- missing logs
- missing power CSVs
- failed or incomplete runs
- NaN timing values
- impossible negative durations
- unexpected dataset sizes
- unexpected batch sizes
- abnormal `S/R` grouped-sample ratio
- power duration mismatch against log timing
- outliers in latency or power
- unsupported GPU or algorithm labels

### Outputs

- `modeling_runs_extracted.csv`
- `validation_summary.md`
- `missing_runs.csv`
- `failed_runs.csv`
- `suspicious_runs.csv`

### Human Approval Gate

The human researcher decides which suspicious runs to keep, drop, or rerun.

The Modeling Agent should not fit final coefficients until this approval is complete.

## 4. Modeling Agent

### Responsibility

The Modeling Agent fits resource models from the validated benchmark data.

It should reproduce the logic currently found in:

- `XRS/modeling_analysis.ipynb` for PtychoPINN
- `XRS/modeling_analysis2.ipynb` for Pty-chi

### PtychoPINN Models

#### I/O Latency

```text
T_IO(D,G) ~= io_intercept(G) + io_slope(G) * S(D)
```

Resolution-aware write tensor form:

```text
T_write(D,N,G) ~= beta_pixel(G) * S(D) * C * N^2
```

where:

- `S(D)` is grouped NN samples
- `C = 4` grouped input channels/images
- `N` is diffraction image resolution

#### Inference Latency

Batch model:

```text
T_infer(D,B,G) ~= A_batch(G,B) + K(G,B) * ceil(S(D) / B)
```

FLOPs model:

```text
T_infer(D,N,B,G) ~= A_flops(G,B,N) + FLOPs(D,N) / P_eff(G,B,N)
```

with:

```text
FLOPs(D,N) = S(D) * F_sample(N)
```

Current baseline assumptions:

```text
F_sample(64)  ~= 1.670 GFLOPs/sample
F_sample(128) ~= 3.774 GFLOPs/sample
F_sample(256) ~= 12.002 GFLOPs/sample
```

#### Power and Energy

```text
P_IO(G) = median I/O power
P_infer(D,B,G) ~= P0(G) + P1(G) * ceil(S(D) / B)
E_total = T_IO * P_IO + T_infer * P_infer
```

### Pty-chi Models

#### I/O Latency

```text
T_IO(D,G,N) ~= A_IO(G) + beta_IO(G) * R(D) * N^2
```

#### Reconstruction FLOPs

```text
FFT2_flops(N) = 10 * N^2 * log2(N)
F_image_epoch(A,N) = q_fft(A) * FFT2_flops(N) + q_elem(A) * N^2
F_recon(D,N,E,A) = R(D) * E * F_image_epoch(A,N)
```

#### Reconstruction Latency

```text
T_recon(D,N,E,A,G) ~= A_recon(A,G) + c(A,G) * F_recon(D,N,E,A)
```

Epoch `1` should be treated as warm-up and excluded from reconstruction fitting unless the human researcher changes this assumption.

#### Power and Energy

```text
P_IO(G,A) = median I/O power
P_recon(D,E,B,G,A) ~= P0(G,A) + P1(G,A) * E * ceil(R(D) / B)
E_total = T_IO * P_IO + T_recon * P_recon
images_per_joule = R(D) / E_total
```

### Outputs

For PtychoPINN:

- `modeling_io_coefficients.csv`
- `modeling_inference_batch_coefficients.csv`
- `modeling_inference_flops_coefficients.csv`
- `modeling_power_model_coefficients.csv`
- `modeling_efficiency_estimates.csv`

For Pty-chi:

- `ptychi_flops_assumptions.csv`
- `ptychi_io_coefficients.csv`
- `ptychi_io_algorithm_coefficients.csv`
- `ptychi_reconstruction_flops_coefficients.csv`
- `ptychi_reconstruction_batch_coefficients.csv`
- `ptychi_power_coefficients.csv`

Shared outputs:

- residual plots
- fit quality report
- model assumptions summary

### Human Approval Gate

The human researcher approves:

- whether the selected model forms are acceptable
- whether warm-up runs are excluded
- whether outliers should be excluded
- whether extrapolation from `N=64` to larger `N` is acceptable
- whether the final coefficients are allowed to enter SystemFlow

## 5. SystemFlow Integration & Report Agent

### Responsibility

The SystemFlow Integration & Report Agent moves approved coefficient tables into the SystemFlow package, validates the model-backed mutations, and produces a final report.

### Integration Targets

PtychoPINN coefficients:

```text
systemflow/xrs_model_data/ptychopinn/
  modeling_io_coefficients.csv
  modeling_inference_flops_coefficients.csv
  modeling_power_model_coefficients.csv
```

Pty-chi coefficients:

```text
systemflow/xrs_model_data/ptychi/
  ptychi_flops_assumptions.csv
  ptychi_io_algorithm_coefficients.csv
  ptychi_reconstruction_flops_coefficients.csv
  ptychi_power_coefficients.csv
```

### Validation Tasks

The agent should run checks equivalent to:

```python
from systemflow.xrs_models import PtyChiResourceModel, PtychoPINNResourceModel

ptychi = PtyChiResourceModel.from_bundled_data()
ptychopinn = PtychoPINNResourceModel.from_bundled_data()

ptychi.predict(num_images=1000, resolution=(64, 64))
ptychopinn.predict(num_images=1000, resolution=(64, 64))
```

It should also validate SystemFlow mutations:

```python
from systemflow.xrs import PhaseReconstruction2D_model, PhaseReconstruction3D_model
```

Expected validation:

- coefficient files load successfully
- supported GPUs are available
- supported Pty-chi algorithms are available
- prediction outputs have positive latency
- prediction outputs have positive energy
- SystemFlow graph execution succeeds

### Report Contents

The final report should include:

- experiment matrix
- data completeness summary
- model formulas
- coefficient tables
- fit quality metrics
- residual plots
- PtychoPINN vs Pty-chi comparison
- SystemFlow validation results
- limitations
- recommended next experiments

### Human Approval Gate

The human researcher approves the final report and whether the coefficient update should be committed or published.

## Human-in-the-Loop Summary

The workflow should include four approval gates.

| Stage | Human Decision |
|---|---|
| After planning | Approve experiment matrix before GPU time is used |
| After validation | Decide which runs are valid, suspicious, or failed |
| After modeling | Approve model assumptions and coefficient tables |
| After integration/reporting | Approve final SystemFlow update and report |

This makes the system:

- automated
- reproducible
- auditable
- scientifically controlled

## Minimal Viable Implementation

The first version should be script-driven rather than fully autonomous.

Suggested folder:

```text
XRS/agentic_workflow/
  agentic_ai_workflow.md
  experiment_plan.yaml
  scripts/
    plan_experiments.py
    run_experiments.py
    parse_logs.py
    validate_runs.py
    fit_ptychopinn.py
    fit_ptychi.py
    validate_systemflow.py
    generate_report.py
  reports/
```

The AI agent should orchestrate these deterministic scripts, inspect outputs, flag problems, and ask for approval at gates. The scientific logic should remain in version-controlled code, not hidden inside prompts.

## One-Sentence Description

This is a five-agent, human-in-the-loop workflow that turns raw XRS benchmark logs and power traces into validated SystemFlow resource models for comparing PtychoPINN and Pty-chi reconstruction performance.
