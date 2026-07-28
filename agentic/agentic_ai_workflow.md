# Agentic AI Workflow for Scientific Application Resource Modeling

## Purpose

This workflow builds system-level performance and energy models for a scientific application as a function of its inputs.

The primary objective is not to compare applications or backends. Given one scientific application, the workflow should determine:

- which application inputs affect the workload
- how those inputs affect FLOPs, data volume, memory use, and execution phases
- which benchmark points are needed to characterize the application
- how latency, power, and energy should be measured
- how measured data should be validated
- how parameterized resource models should be fitted
- how the resulting models should be integrated into SystemFlow

The target model has the general form:

```text
(application inputs, execution configuration, hardware)
    -> (latency, power, energy, throughput, memory)
```

For example:

```text
(number of images, resolution, batch size, iterations, GPU)
    -> (end-to-end latency, total energy)
```

The workflow automates application analysis, experiment design, benchmark execution, data validation, model fitting, and SystemFlow integration. Human researchers remain responsible for confirming application semantics, approving measurement boundaries, accepting assumptions, authorizing expensive runs, and approving final models.

## Core Modeling Principles

### Model one application across an input space

The central object is a scientific application with a parameterized workload. Different algorithms or backends may be represented as application inputs when appropriate, but backend comparison is not the organizing principle.

### Separate theoretical work from measured resource use

The workflow distinguishes:

- **algorithmic FLOPs**: useful operations implied by the algorithm
- **executed FLOPs**: operations actually executed, including padding or other implementation effects
- **achieved FLOP/s**: hardware-specific performance measured at runtime
- **data movement**: input, output, host-device, and memory traffic
- **latency and energy**: system-level quantities measured over an explicitly defined boundary

FLOPs alone are not sufficient to predict performance or energy. I/O volume, memory behavior, initialization, communication, and hardware utilization may be equally important.

### Make the system boundary explicit

Every plan must define what is included in the measurement. A system-level measurement may include:

```text
application startup
    -> input loading
    -> preprocessing
    -> host-to-device transfer
    -> core computation
    -> device-to-host transfer
    -> postprocessing
    -> output writing
```

End-to-end and phase-level measurements should be kept distinct.

### Keep scientific logic auditable

LLMs can help inspect source code, interpret configuration, identify likely inputs, and draft formulas. Deterministic code should validate schemas, evaluate formulas, expand experiment matrices, compute run counts, and enforce approval rules. Every inferred input or workload equation should include evidence, assumptions, and a confidence level.

## Five-Agent Workflow

```text
Application Characterization & Planning Agent
    -> human approval of inputs, workload model, and experiment plan

Benchmark Runner Agent
    -> raw logs, power traces, profiler outputs, and run manifest

Data & Validation Agent
    -> human approval of usable measurements

Resource Modeling Agent
    -> human approval of model form and fitted coefficients

SystemFlow Integration & Report Agent
    -> human approval of the integrated model and final report
```

The workflow may be iterative. If the Resource Modeling Agent finds poor coverage or large uncertainty in part of the input space, it can request additional benchmark points from the first agent.

## 1. Application Characterization & Planning Agent

### Responsibility

The first agent analyzes a scientific application before any expensive benchmarking begins. It discovers the application's meaningful inputs, traces how those inputs affect computation and data movement, constructs a symbolic workload model, and proposes a benchmark plan for calibrating a system-level performance and energy model.

This agent combines two closely related activities:

1. **Application characterization**: understand what work the application does.
2. **Experiment planning**: choose measurements that identify how resource use changes across the intended input space.

It produces a characterization and experiment contract, not benchmark commands and not fitted hardware coefficients.

### Inputs

- application source repository or package
- application entry point, CLI, API, notebook, or job script
- example invocation and representative input
- configuration files
- target input domain
- target hardware or system configuration
- desired predictions, such as latency or energy
- measurement and resource constraints
- optional publications or algorithm descriptions

### Step 1: Discover Application Inputs

The agent should inspect:

- command-line arguments
- configuration objects and files
- public APIs
- dataset loaders
- tensor or array shapes
- model and solver configuration
- loop bounds and convergence conditions
- example scripts, notebooks, and documentation

It should classify discovered parameters:

| Class | Examples |
|---|---|
| Scientific input | number of images, events, particles, grid size |
| Problem shape | resolution, dimensions, channels |
| Algorithm parameter | iterations, epochs, tolerance, solver |
| Execution parameter | batch size, worker count, precision |
| Hardware parameter | GPU, CPU count, memory |
| Reproducibility parameter | random seed |
| Operational parameter | output directory, logging verbosity |

Not every configuration field belongs in the resource model. The agent should explain why each selected variable is or is not a modeling input.

Each discovered input should contain provenance:

```yaml
- name: resolution
  symbol: N
  type: integer
  units: pixels
  role: problem_shape
  affects: [input_bytes, flops, memory]
  evidence:
    file: application/config.py
    line: 42
  confidence: high
```

### Step 2: Identify Execution Phases

The agent should identify important system-level phases and their boundaries:

- initialization and model loading
- input I/O
- preprocessing
- host-device transfer
- core compute
- communication or synchronization
- device-host transfer
- postprocessing
- output I/O

It should state which phases are included in end-to-end latency, end-to-end energy, and phase-level measurements. If phase boundaries are not observable, the agent should recommend instrumentation points rather than silently inventing them.

### Step 3: Construct a Parameterized Workload Model

The workload model should express theoretical work as functions of application inputs. It may include:

- number of samples and batches
- tensor or array shapes
- input and output bytes
- algorithmic and executed FLOPs
- estimated peak memory
- communication volume
- operation counts by phase

Example for batched inference:

```text
K(R,B) = ceil(S(R) / B)
F_useful(R,N) = S(R) * F_sample(N)
F_executed(R,N,B) = K(R,B) * B * F_sample(N)   # if padded batches execute
```

Example for an iterative FFT-based application:

```text
FFT2_flops(N) = 10 * N^2 * log2(N)
F_image_iteration(N) = q_fft * FFT2_flops(N) + q_elementwise * N^2
F_total(R,N,I) = R * I * F_image_iteration(N)
```

The formulas must declare assumptions. When a quantity cannot be derived statically, the agent should mark it for profiler or benchmark validation:

```yaml
total_flops:
  expression: "R * I * (q_fft * 10 * N**2 * log2(N))"
  method: analytical
  assumptions:
    - square two-dimensional input
    - fixed iteration count
  evidence:
    - file: application/reconstruction.py
      operation: fft2
  confidence: medium
  validation_required: true
```

Evidence may come from source-level operator and shape analysis, framework FLOP counters, profiler output, algorithm documentation, or publications. Estimates based only on analogy or LLM inference must be labeled low-confidence.

### Step 4: Define the Modeling Objective

The plan must identify dependent and independent variables:

```yaml
modeling_objective:
  predict:
    - end_to_end_latency_s
    - total_energy_j
    - throughput_samples_per_s
  as_function_of:
    - number_of_images
    - resolution
    - batch_size
    - iterations
    - hardware_id
  intended_prediction_domain:
    number_of_images: [1000, 16000]
    resolution: [64, 256]
```

The intended prediction domain matters because a fitted model should not be silently used far outside the measured range.

### Step 5: Design the Benchmark Matrix

The agent selects benchmark points that provide enough information to calibrate workload and resource models without requiring an unnecessarily large Cartesian product.

It should consider:

- linear, logarithmic, or domain-specific sampling
- boundary points
- pilot versus full experiments
- one-factor and interaction coverage
- repetitions and random seeds
- warm-up runs
- invalid or unsupported input combinations
- resource budgets and maximum run count
- expected runtime and storage

Example:

```yaml
sampling:
  number_of_images:
    strategy: logarithmic
    values: [1000, 2000, 4000, 8000, 16000]
  resolution:
    strategy: categorical
    values: [64, 128, 256]
  batch_size:
    strategy: selected
    values: [256, 512, 1024]
  iterations:
    strategy: selected
    values: [1, 2, 4]
```

The agent must calculate the resulting run count deterministically and may recommend a smaller pilot matrix when the full design exceeds the budget.

### Step 6: Define the Measurement Protocol

The plan should specify:

- warm-up policy and measured repetitions
- timer and synchronization requirements
- power measurement target: accelerator, CPU, or whole node
- power sampling interval
- clock alignment between logs and power traces
- profiler requirements
- system metadata to record
- expected output artifacts

For short runs, the agent should verify that the power sampling frequency is sufficient for meaningful energy integration.

### Validation and Feasibility Checks

The agent should flag:

- missing or ambiguous modeling inputs
- unbounded or runtime-dependent loop counts
- insufficient sampling of a proposed model variable
- a requested energy model without a power measurement method
- input points that exceed device memory
- unsupported input or hardware combinations
- runs too short for the selected power sampling interval
- experiments that vary too many coupled parameters without enough coverage
- plans that exceed run-count, wall-time, or storage budgets
- inconsistent system boundaries across runs
- formulas without evidence or assumptions
- extrapolation beyond the benchmark domain

Validation severity should be explicit:

```text
ERROR   prevents approval or execution
WARNING requires human review
INFO    records assumptions or recommendations
```

### Outputs

- `application_profile.yaml`
- `workload_model.yaml`
- `experiment_plan.yaml`
- `dry_run_matrix.csv`
- `application_analysis_report.md`
- `approval_summary.md`

`application_profile.yaml` records inputs and phases. `workload_model.yaml` records symbolic formulas and evidence. `experiment_plan.yaml` records the measurement contract. `dry_run_matrix.csv` contains one row per proposed run.

### Human Approval Gate

Before benchmark execution, the researcher approves:

- whether the application inputs were interpreted correctly
- which inputs belong in the performance and energy model
- the system and measurement boundaries
- workload formulas and assumptions
- the intended prediction domain
- benchmark sample points and repetitions
- target hardware and measurement tools
- estimated experiment cost

The approval state should be stored in the plan. Any material plan change after approval should invalidate the approval and require review again.

## 2. Benchmark Runner Agent

### Responsibility

The Benchmark Runner Agent executes the approved run matrix and collects raw measurements without reinterpreting the research objective. It should support local and HPC execution. Early implementations should default to dry-run mode and generate commands or job scripts for review.

### Inputs

- approved `experiment_plan.yaml`
- `dry_run_matrix.csv`
- application entry point and environment specification
- instrumentation configuration
- hardware or scheduler configuration

### Outputs

- stdout and stderr logs
- phase timing logs
- accelerator, CPU, or node power traces
- optional profiler outputs
- environment and hardware metadata
- run manifest
- failed and skipped run lists

Every run should record its deterministic run ID, exact inputs, exact invocation, source revision, software environment, hardware identity, timestamps, exit code, and output paths.

### Safety Rules

- Do not execute a plan that is not approved.
- Default to dry-run mode.
- Never delete previous results automatically.
- Skip complete existing runs unless rerun is explicitly approved.
- Do not invent or modify benchmark points.
- Keep stdout, stderr, commands, timestamps, and exit codes.
- Record failures without stopping unrelated runs when safe.

### Human Approval Gate

Approval is required before consuming compute resources. Additional approval may be required for rerunning failed or suspicious measurements.

## 3. Data & Validation Agent

### Responsibility

The Data & Validation Agent converts logs, power traces, profiler results, and manifests into an auditable table aligned with the planned application inputs.

### Parsing Targets

For each run, extract:

- all planned input and configuration values
- phase and end-to-end latency
- average, minimum, maximum, and time-series power
- integrated energy
- profiler operation counts and FLOPs when available
- peak memory
- I/O and transfer volume
- run status and measurement completeness

### Validation Checks

The agent should flag:

- missing, failed, incomplete, or duplicate runs
- observed parameters that differ from the plan
- NaN or negative durations
- clock misalignment between logs and power traces
- insufficient power samples
- power duration inconsistent with application timing
- outliers across repetitions
- impossible power, energy, FLOP, or memory values
- disagreement between analytical and profiler FLOPs
- unexpected hardware or software changes
- measurements outside the approved input domain

### Outputs

- `modeling_runs_extracted.csv`
- `validation_summary.md`
- `missing_runs.csv`
- `failed_runs.csv`
- `suspicious_runs.csv`
- `approved_modeling_dataset.csv`

### Human Approval Gate

The researcher decides which suspicious runs to retain, exclude, or rerun. Final model fitting must use the explicitly approved dataset.

## 4. Resource Modeling Agent

### Responsibility

The Resource Modeling Agent fits parameterized system-level models using the approved benchmark dataset and workload equations produced by the first agent.

Possible model structure:

```text
T_total(x,h) = T_startup(h) + T_io(x,h) + T_compute(x,h) + T_output(x,h)
T_compute(x,h) = intercept(h) + F(x) / P_effective(x,h)
E_total(x,h) = integral(P_system(t), t)
```

Here `x` is the application input vector, `h` is the hardware configuration, and `F(x)` is the parameterized workload model.

The agent should not assume that FLOPs alone explain runtime. It should consider input/output bytes, batch count, memory footprint, communication, and execution phases where supported by measurements.

### Responsibilities

- select candidate model forms consistent with application characterization
- fit latency, power, energy, throughput, and memory models as requested
- quantify residuals and uncertainty
- compare analytical FLOPs with profiler measurements
- detect under-sampled regions and parameter interactions
- validate models on held-out benchmark points
- define the supported prediction domain
- recommend additional experiments where fit quality is insufficient

### Outputs

- fitted coefficient tables
- machine-readable model definitions
- fit-quality metrics
- residual and uncertainty plots
- model assumptions summary
- supported-domain metadata
- recommendations for additional benchmark points

### Feedback Loop

If the data cannot identify the proposed model, or error is concentrated in part of the input domain, the agent should request a targeted plan extension:

```text
Resource Modeling Agent
    -> requested additional input points
    -> Application Characterization & Planning Agent
    -> human approval
    -> Benchmark Runner Agent
```

### Human Approval Gate

The researcher approves model form, included measurements, workload assumptions, treatment of warm-up and outliers, uncertainty, supported range, and coefficients allowed to enter SystemFlow.

## 5. SystemFlow Integration & Report Agent

### Responsibility

The SystemFlow Integration & Report Agent packages an approved application resource model for use by SystemFlow, validates predictions through SystemFlow mutations and graphs, and produces a final scientific report.

### Integration Requirements

The integrated model should:

- accept approved application inputs
- evaluate derived workload quantities such as FLOPs and bytes
- select hardware-specific fitted coefficients
- return positive, unit-consistent latency, energy, power, and throughput
- reject or warn about inputs outside the supported domain
- preserve coefficient provenance and model version

For XRS applications, existing integration patterns include:

```python
from systemflow.xrs_models import PtyChiResourceModel, PtychoPINNResourceModel
from systemflow.xrs import PhaseReconstruction2D_model, PhaseReconstruction3D_model
```

These are examples of model consumers, not a requirement that the workflow compare PtychoPINN with Pty-chi.

### Validation Tasks

- load coefficient and model-definition files
- reproduce selected benchmark predictions
- validate units and input mappings
- verify supported hardware and input domains
- check positive latency and energy
- execute representative SystemFlow graphs
- compare SystemFlow outputs with held-out measurements
- ensure out-of-domain inputs produce a warning or error

### Report Contents

- application inputs and intended prediction domain
- system and measurement boundaries
- execution phase decomposition
- analytical FLOP, data-volume, and memory formulas
- benchmark matrix and completeness
- measurement methodology
- fitted performance and energy models
- coefficient tables
- residuals, uncertainty, and validation results
- SystemFlow integration results
- limitations and recommended follow-up experiments

### Human Approval Gate

The researcher approves the final report, model package, and whether the SystemFlow update should be committed or published.

## Human-in-the-Loop Summary

| Stage | Human decision |
|---|---|
| After characterization and planning | Confirm inputs, formulas, boundaries, sampling, and cost |
| Before execution or reruns | Approve compute and measurement resource use |
| After validation | Decide which measurements are scientifically usable |
| After modeling | Approve model form, coefficients, uncertainty, and supported domain |
| After integration | Approve the SystemFlow model and final report |

## Minimal Viable Implementation

The first version should be script-driven rather than fully autonomous:

```text
agentic/
  agentic_ai_workflow.md
  agents/
    characterization/
      agent_design.md
      analysis_request_schema.yaml
      application_characterization_schema.yaml
      prompts/
      runner.py
      tools.py
      cli.py
    planner/
      experiment_plan_schema.yaml
      expand_matrix.py
      validate_plan.py
      render_approval_summary.py
  runner/
    generate_run_scripts.py
    run_experiments.py
  validation/
    parse_measurements.py
    validate_runs.py
  modeling/
    fit_resource_models.py
  integration/
    validate_systemflow.py
    generate_report.py
```

The AI agents should orchestrate deterministic tools, inspect outputs, identify uncertainties, and request approval at defined gates. Scientific logic and workload equations should remain in version-controlled files, not hidden inside prompts.

## First Implementation Milestone

The first milestone should implement only the first agent for one selected scientific application:

1. discover and classify application inputs
2. identify major execution phases
3. construct a parameterized FLOP and data-volume model
4. attach evidence, assumptions, and confidence to each formula
5. define the target prediction domain
6. propose a pilot benchmark matrix
7. validate the matrix and calculate its run count
8. generate an application analysis report for human approval

## One-Sentence Description

This is a five-agent, human-in-the-loop workflow that analyzes a scientific application, characterizes how its inputs determine computational work, measures system-level resource use, fits performance and energy models, and integrates the approved models into SystemFlow.
