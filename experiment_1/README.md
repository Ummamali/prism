# Experiment 1 — Does visual grounding decay survive control for prefix redundancy?

**The one-sentence version:** when a vision-language model (a model that can look at
an image and answer questions about it — "VLM" for short) writes out a step-by-step
answer, people have observed that later reasoning steps seem to depend less on the
actual image than earlier steps do (as if the model "stops looking" partway through).
This experiment checks whether that's a real effect, or just an illusion caused by
the model's own earlier sentences already containing enough information that it
doesn't need to re-check the image anymore.

Pilot implementation of the marginal/total dependence decomposition ("decomposition"
= breaking one effect down into the separate pieces that add up to it) from
`docs/research_proposal_prism.pdf` §5.1–§5.2, run across three visual-QA benchmarks
using `Qwen/Qwen2-VL-2B-Instruct` (a small, Kaggle/Colab-friendly vision-language
model). Goal: compute β_M ("beta-M", the marginal/controlled-direct slope), β_T
("beta-T", the total slope), and Δβ ("delta-beta") = β_T − β_M, and see whether
reported grounding decay (β_M < 0, i.e. dependence trending downward across steps)
survives once prefix redundancy is controlled for (β_T ≈ 0 vs β_T < 0).

If that last sentence is dense, skip to **["The four conditions, in plain
language"](#the-four-conditions-in-plain-language)** below before the technical
table — it walks through the same idea slowly.

## Codebase map

```
experiment_1/
├── config.yaml              All the settings: which model, which benchmark,
│                             how many examples, output paths. Start here.
├── requirements.txt          Python packages to install.
│
├── lib/                       Shared building blocks. Nothing here is meant to
│  │                           be run directly — the scripts below import from it.
│  ├── config.py               Loads config.yaml; DATASET_PRESETS (the three
│  │                           benchmarks' Hugging Face dataset ids/splits/columns).
│  ├── io_utils.py             Read/write the .jsonl files passed between stages.
│  ├── model.py                Loads the vision-language model onto a GPU/CPU.
│  ├── scoring.py              THE ACTUAL EXPERIMENT MATH: image ablation, splitting
│  │                           an answer into steps, generating trajectories, and
│  │                           teacher-forced scoring (all defined below).
│  └── checkpoint.py           Zips a benchmark's results directory for crash-safety.
│
├── data_loading.py    STAGE 1  Downloads a benchmark, samples ~100 examples,
│                                caches them + their images locally.
├── inference.py        STAGE 2  Runs the four-condition scoring harness over the
│                                cached examples. This is the slow part.
├── decomposition.py    STAGE 3  Turns per-example scores into the headline
│                                β_M / β_T / Δβ numbers and CSVs.
│
├── run_multi_shard.py           Kaggle helper: one GPU process working through
│                                several benchmark "shards" without reloading the
│                                model between them.
├── run_benchmark.py              Kaggle helper: runs ONE benchmark split across
│                                both GPUs, then decomposition + checkpoint.
├── checkpoint_now.py             Kaggle helper: manually trigger a results backup
│                                (zip) at any time, e.g. right before stopping early.
│
├── common.py                     A thin compatibility shim — old code imported
│                                everything from here before it was split into
│                                lib/. Safe to ignore; new code should use lib/*.
│
├── data/                          Generated: cached benchmark samples + images.
├── results_{mathvista,hallusionbench,chartqa}/   Generated: per-benchmark scores.
└── checkpoints/                   Generated: timestamped .zip backups.
```

`common.py` exists only so that anything still doing `from common import X` (for
example, a monitoring snippet already pasted into a running Kaggle notebook cell)
keeps working after the reorganization — it just re-exports names from `lib/`.

## How to run it

**Local / single-GPU / small smoke test:**

```bash
pip install -r requirements.txt -r experiment_1/requirements.txt
python experiment_1/data_loading.py --dataset mathvista
python experiment_1/inference.py --dataset mathvista --limit 3   # smoke test: only 3 examples
python experiment_1/inference.py --dataset mathvista             # full run
python experiment_1/decomposition.py --dataset mathvista
```

Swap `mathvista` for `hallusionbench` or `chartqa` to run a different benchmark —
see ["The three benchmarks"](#the-three-benchmarks-and-why-these-three) below.
Swap models by editing `model.name` in `config.yaml`.

**Kaggle dual-T4 GPU setup** (all three benchmarks, both GPUs, automated,
crash-safe): see ["Running on Kaggle"](#running-on-kaggle-dual-gpu) further down.

## What's implemented, following the proposal's §5.1 table exactly

For each step `s` of a trajectory (a "trajectory" here just means one full
step-by-step answer — a sequence of steps `y = (y_1, ..., y_S)`) generated with the
image present, four teacher-forced conditions are scored. ("Teacher-forced scoring"
means: instead of letting the model freely generate text, we feed it a specific
piece of text and ask "how probable did you think each of these words were?" — see
the plain-language section below for more.)

| Condition | Image | Prefix (the text already "said" before this step) | Log-likelihood |
|---|---|---|---|
| C1 | I (present) | y_<s (the real preceding steps) | ℓ₁ |
| C2 | Ĩ (ablated/blanked out) | y_<s | ℓ₂ |
| C3 | I (present) | scrub(y_<s) (a stand-in prefix with no real visual info) | ℓ₃ |
| C4 | Ĩ (ablated) | scrub(y_<s) | ℓ₄ |

("Log-likelihood", ℓ, is a number that goes up when the model found the actual
words more predictable/expected, and down when it found them surprising. It's
measured in **nats/token** — "nats" is just the log-probability unit you get from
natural logarithms, the same idea as "bits" but base-*e* instead of base-2; higher
nats/token = the model was less surprised, i.e. it thought this text was more
likely. "/token" means it's averaged per word-piece, so short and long steps are
comparable.)

```
M_s = ℓ₁ - ℓ₂           controlled direct effect ("marginal" — what most prior work measures)
T_s = ℓ₁ - ℓ₄           total visual dependence
C_s = T_s - M_s = ℓ₂ - ℓ₄   prefix-carried / indirect dependence
S_s = ℓ₃ - ℓ₄           visual sufficiency
R_s = C_s / max(T_s, ε)     redundancy ratio, clipped to [0,1]
```

`β_M`, `β_T` are slopes ("slope" = how much a value trends up or down, here as you
move from the first step of an answer to the last) of `M_s`, `T_s` on normalised
step index `ŝ = s/S` (so step 1 of a 4-step answer and step 3 of a 12-step answer
can both be compared as "25% of the way through") per proposal §5.2.
`Δβ = β_T - β_M`.

### The four conditions, in plain language

Imagine the model has just written steps 1 and 2 of its answer, and we want to
check: how much is step 3 actually informed by the picture, versus just following
logically from what it already wrote?

- **C1 (real image, real prefix):** the normal case — show the model the real
  picture and what it actually said in steps 1–2, then check how "expected" step 3
  was. This is the baseline everything else compares against.
- **C2 (blanked image, real prefix):** take away the picture, but keep the real
  steps 1–2. If step 3 is *still* just as expected, that means steps 1–2 alone
  already gave the model everything it needed — the image wasn't doing much work
  for this particular step, at least not directly.
- **C3 (real image, fake prefix):** give the model the real picture back, but
  swap steps 1–2 for unrelated placeholder text that carries no real visual
  information (see "scrub" below). If step 3 is still expected, the image alone is
  *sufficient* to predict it, without needing the earlier steps at all.
- **C4 (blanked image, fake prefix):** take away both the picture and the real
  prefix. This is the "worst case" — how expected is step 3 with essentially no
  real information at all? It's the baseline for the other three.

The whole point of running all four (not just C1 vs C2) is this: if you only
compare C1 to C2, and see that removing the image barely changes anything, you
might conclude "the model isn't using the image for this step." But that's
ambiguous — maybe the model isn't using the image *directly* anymore because its
own earlier sentences already smuggled the visual information forward (i.e. the
image's influence got "carried" into the prefix). C3/C4 exist specifically to
separate those two explanations: `M_s` (from C1/C2) mixes both effects together,
while `T_s` (from C1/C4) isolates the model's *total* dependence on the image, with
no help at all from a real prefix. `C_s = T_s - M_s` is then just "whatever part
of the total effect wasn't captured by M_s" — the part that's flowing through the
prefix rather than the image directly. `R_s` turns that into a 0–1 fraction: R_s
close to 1 means most of the apparent visual dependence is actually redundant with
what the prefix already says; R_s close to 0 means the image really is still doing
independent work at that step.

## What "scrub" means, concretely

"scrub(y_<s)" = a prefix that looks like a normal partial answer but carries no
real visual information tied to the actual image. This pilot builds it by
generating a **second, independent** answer to the same question with the image
blanked out from the very start (called the "blind trajectory", z), and using its
first few steps as the stand-in prefix. Concretely: `scrub(y_<s)` = the first
`min(s-1, len(z))` steps of z (`lib/scoring.py::cumulative_text`,
`inference.py::scrub_prefix`).

## The three benchmarks, and why these three

Each benchmark tests a different flavor of "does the model actually need to keep
looking at the picture":

- **MathVista** (`AI4Math/MathVista`, split `testmini`) — math problems that
  reference a diagram, chart, or geometric figure. Chosen as the "sanity anchor":
  it's the benchmark the original research proposal used, so re-running it under
  the current, cleaned-up pipeline gives a baseline to compare the other two
  against.
- **HallusionBench** (`lmms-lab/HallusionBench`, split `image`) — questions
  specifically designed to tempt a model into a plausible-sounding but wrong answer
  if it stops checking the image and just pattern-matches from text alone. Directly
  relevant to this experiment's question, since that's exactly the failure mode
  being measured.
- **ChartQA** (`lmms-lab/ChartQA`, split `test`) — questions about reading values
  off charts. Charts require precise, sustained visual reading (you can't guess an
  exact number from general knowledge the way you sometimes can with photos),
  making this a stress test for whether grounding decay is worse when the task
  genuinely can't be solved from memory/reasoning alone.

A fourth benchmark, **Visual-CoT**, was investigated and dropped for this batch:
its Hugging Face loader throws a schema-mismatch error across its source files,
and its images are split across ~139GB of undocumented tar archives rather than
being embedded in the dataset — not worth building a fragile pipeline around for a
100-example pilot.

Which benchmark a given run uses is picked via `data.dataset` in `config.yaml`
(or the `--dataset` flag on any script) — see `lib/config.py`'s `DATASET_PRESETS`
for the exact Hugging Face dataset id / split / column names each one maps to.
Because `data_loading.py` normalizes every benchmark into the same on-disk
pid/question/answer/image_path format, **the benchmark you pick never changes the
scoring math** — `inference.py` and `decomposition.py` are completely benchmark-
agnostic.

## Design decisions made for this pilot (confirmed with the user first)

These are exactly the places the proposal leaves open or flags as risk;
everything below was a deliberate choice, not a silent approximation:

1. **Four-condition harness, not two.** The user's original ask ("run the
   model twice, with/without image") only yields M_s. Computing T_s and
   Δβ requires C3/C4, which need a scrub operator — confirmed with the
   user before implementing.
2. **Scrub operator = blind regeneration** (proposal §5.3's "cleanest
   causally" option) — see ["What 'scrub' means"](#what-scrub-means-concretely)
   above for the plain-language version.
3. **Matched by step index, not token count.** The proposal says "a
   prefix of matched length" without specifying step vs. token matching.
   This pilot matches by step index (`z_<s` for `y_<s`) because it is
   simplest to implement correctly; if `z` is shorter than `s-1` steps,
   the scrub prefix is capped at all of `z`. Revisit if step counts
   between the real and blind trajectories diverge a lot in practice —
   `n_steps` for both trajectories is logged per example so this is
   auditable from `results_{dataset}/per_example/*.json`.
4. **Image ablation = mean-pixel fill** ("ablation" here means replacing
   the real image with a featureless stand-in — specifically, a flat image
   painted with the image's own average color), the first (simplest) of
   the proposal's five listed operators (§5.4). The proposal's headline
   operator, region-blur, needs a question-relevant-region localizer and
   is out of scope for this pilot. Ablation preserves image size, so the
   number of "vision tokens" (the internal placeholder units the model
   uses to represent an image, roughly like word-tokens but for image
   patches) — and therefore prompt structure — is identical across
   image-present/image-ablated conditions.
5. **Step segmentation = sentence-split** on the free-text CoT ("CoT" =
   chain-of-thought, i.e. the model's step-by-step written reasoning;
   the proposal defines "steps" abstractly and doesn't specify how to cut
   raw text into them). `lib/scoring.py::segment_steps`.
6. **Aggregation = per-example OLS slope, then mean + percentile
   bootstrap CI across examples** (`decomposition.py`), not the
   proposal's full mixed-effects model ("mixed-effects model" = a fancier
   regression that accounts for some examples/models being systematically
   different from others, via "random intercepts") with random intercepts
   by item *and* model (§5.2) — this pilot only runs one model, and a
   single random-effects axis adds a `statsmodels` dependency for little
   benefit at ~100 examples. A pooled (unclustered) OLS slope is also
   reported as a fast secondary check. Upgrade to `statsmodels.MixedLM`
   when scaling to the multi-model sweep in Phase 2 of the proposal.
7. **Sample = 100 items per benchmark** (config default), matching the
   proposal's Phase 0 sanity-anchor dataset scale and its own ~50-100
   pilot-scale sample size, run across three benchmarks (see above)
   instead of just MathVista.

## Explicitly out of scope for this pilot

Everything below is real proposal content but belongs to later phases,
not the "does decay survive prefix control" pilot question:
- Scrub-fidelity human audit / Gate G1 (§5.3)
- The necessity × sufficiency taxonomy and confabulation rate (§5.5)
- The policy-independent demand estimator `d_s` and PRISM-Minimal (§5.6–5.7)
- Multi-model sweep and mixed-effects regression (Phase 2)
- Region-blur ablation and the other 4 ablation/4 scrub operator cross-grid
- Visual-CoT (see ["The three benchmarks"](#the-three-benchmarks-and-why-these-three) above for why)

## Output files

Every path below is per-benchmark: `results_{dataset}/` where `{dataset}` is
`mathvista`, `hallusionbench`, or `chartqa`.

- `results_{dataset}/per_example/{pid}.json` — real & blind trajectories, per-step
  ℓ₁..ℓ₄.
- `results_{dataset}/aggregate/per_step_measures.csv` — every step's M_s, T_s, C_s,
  S_s, R_s, s_hat.
- `results_{dataset}/aggregate/per_example_slopes.csv` — per-example β_M, β_T, Δβ,
  mean R_s.
- `results_{dataset}/aggregate/summary.json` — aggregate β_M/β_T/Δβ with bootstrap
  CIs, pooled-OLS secondary check, mean redundancy ratio, and a mechanical
  read of the proposal's §5.2 three-way outcome split (real / artifactual
  / regime-dependent) — treat as a pointer to inspect, not a conclusion.
- `checkpoints/{dataset}_{label}_{timestamp}.zip` — crash-safety backups (see below).

## Running on Kaggle (dual GPU)

This pipeline is set up to run all three benchmarks (~100 examples each) across
Kaggle's two T4 GPUs, automatically, with crash-safe checkpointing so a
disconnected session doesn't lose completed work.

**One-time setup per benchmark** (downloads + samples the data; CPU-only, cheap):

```bash
python experiment_1/data_loading.py --dataset mathvista
python experiment_1/data_loading.py --dataset hallusionbench
python experiment_1/data_loading.py --dataset chartqa
```

**Automated run** — launches one benchmark in the background, split 50/50
across both GPUs in parallel:

```python
import subprocess, sys
orchestrator = subprocess.Popen(
    [sys.executable, "experiment_1/run_benchmark.py", "--dataset", "chartqa"],
    stdout=open("/kaggle/working/orchestrator.log", "w"), stderr=subprocess.STDOUT,
    start_new_session=True,   # survives the notebook cell being interrupted
)
```

`run_benchmark.py` writes its live status to
`experiment_1/checkpoints/run_status.json` (which benchmark is running, each GPU's
progress) for external polling. The script also prints one line every 3 seconds with
the last log line of each GPU. The per-GPU logs are the
per-benchmark, per-GPU log files under `/kaggle/working/`.

**Crash-safety.** `experiment_1/checkpoints/` (which resolves under
`/kaggle/working/` on Kaggle — persistent output storage) fills up with zipped
snapshots automatically:
- every 10 completed examples *or* every 10 minutes within a benchmark
  (`*_partial_*.zip`),
- once each GPU's half of a benchmark finishes (`*_shard_done_*.zip`),
- once a full benchmark (both GPUs) finishes (`*_complete_*.zip`).

You can also trigger one manually at any time — e.g. right before stopping a run
early:

```bash
python experiment_1/checkpoint_now.py                       # all three benchmarks
python experiment_1/checkpoint_now.py --dataset hallusionbench  # just one
```

**Important:** these zips only protect you *within* a live session. Kaggle only
permanently attaches `/kaggle/working`'s contents to your account when you
explicitly commit a "Save Version" (or the session ends normally) — if the session
is killed before that, anything not committed is lost. Commit periodically, or
download the checkpoint zips, if a run matters.
