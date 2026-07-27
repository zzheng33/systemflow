# Experiment Planner Agent Design

## Goal

The Experiment Planner Agent converts a scientific question into a structured experiment plan that can be reviewed by a human and executed later by an Experiment Runner Agent.

The planner should be generic. It must not assume that every experiment is XRS, PtychoPINN, Pty-chi, or even GPU-based. Those are application profiles that can plug into the same planning interface.

## Core Principle

The planner produces a **contract**, not a script.

The contract is `experiment_plan.yaml`. It describes:

- what scientific application is being studied
- what workloads will run
- what datasets or problem sizes will be used
- what hardware targets will be used
- what parameters will be swept
- what metrics will be collected
- what outputs are expected
- what human approvals are required

The Runner Agent can later turn this contract into concrete commands, job scripts, or HPC submissions.

## Planner Inputs

The planner should accept:

- user research question
- application profile
- workload or backend definitions
- datasets or problem sizes
- hardware targets
- software environment
- parameter sweeps
- metric requirements
- execution constraints
- resource budget
- output format requirements

Example user request:

```text
Plan a pilot experiment comparing a neural reconstruction backend and an iterative solver on A100 and H200 using three dataset sizes.
```

The planner should turn that into a structured matrix and an approval summary.

## Planner Outputs

The planner should write or update:

- `experiment_plan.yaml`
- `approval_summary.md`
- `dry_run_matrix.csv`

The first implementation can generate only `experiment_plan.yaml` and `approval_summary.md`. The dry-run matrix can be added after the schema stabilizes.

## Generic Planning Model

The planner should organize every experiment around these concepts:

| Concept | Meaning |
|---|---|
| `application` | Scientific domain or application profile |
| `study` | High-level research objective |
| `workloads` | Algorithms, models, solvers, or pipeline variants to benchmark |
| `datasets` | Input data, synthetic cases, problem sizes, or event samples |
| `hardware` | GPUs, CPUs, accelerators, nodes, memory limits |
| `sweeps` | Parameters varied across runs |
| `metrics` | Quantities to collect or fit |
| `execution` | Local/HPC mode, dry-run behavior, skip/rerun policy |
| `outputs` | Expected logs, power traces, manifests, coefficient files |
| `approval` | Human approval status and notes |

## Application Profiles

Application profiles describe domain-specific vocabulary and defaults.

Examples:

- `xrs_ptychography`
- `hep_trigger_pipeline`
- `microscopy_segmentation`
- `climate_simulation`
- `generic_scientific_ai_inference`
- `generic_iterative_solver`

An application profile can define:

- allowed workloads
- default datasets
- default metrics
- expected log patterns
- expected output folder layout
- domain-specific fairness checks
- domain-specific warning rules

For XRS, PtychoPINN and Pty-chi are workloads under the `xrs_ptychography` profile. They should not be baked into the generic planner.

## Planner Agent Responsibilities

### 1. Interpret the Research Question

The planner extracts:

- scientific goal
- application domain
- workloads to compare
- hardware targets
- data sizes
- required metrics
- constraints

If required fields are missing, the planner should use conservative defaults from the selected application profile and clearly mark assumptions.

### 2. Build an Experiment Matrix

The planner expands the experiment into run combinations.

Example dimensions:

- workloads
- datasets
- hardware targets
- batch sizes
- resolutions
- algorithm settings
- random seeds
- repetitions

The output should include an estimated run count.

### 3. Check Feasibility

The planner should flag:

- very large run counts
- missing hardware targets
- unsupported workload/hardware combinations
- missing required metrics
- comparisons that may be scientifically unfair
- experiments that cannot collect required power or timing data

### 4. Generate an Approval Summary

The approval summary should be short and direct:

- total run count
- run count by workload
- estimated outputs
- important assumptions
- warnings
- human decisions needed

### 5. Produce a Runner-Ready Plan

The plan should be machine-readable and stable enough that the Runner Agent can consume it without needing to reinterpret the original user request.

## Human-in-the-Loop Gate

The Experiment Runner Agent must not run expensive experiments until the planner output is approved.

Approval state should be stored in the plan:

```yaml
approval:
  status: pending
  approved_by: null
  approved_at: null
  notes: null
```

Valid statuses:

- `pending`
- `approved`
- `rejected`
- `needs_revision`

## Minimal Planner Folder

Recommended files:

```text
agentic_workflow/planner/
  planner_design.md
  experiment_plan_schema.yaml
  example_experiment_plan.yaml
```

Later implementation files:

```text
agentic_workflow/planner/
  plan_experiments.py
  validate_plan.py
  render_approval_summary.py
```

## Design Decision

The planner should be LLM-assisted but schema-constrained.

The LLM can help translate natural language into a draft plan. Deterministic validation code should check that the plan is complete, consistent, and safe to run.
