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
`docs/research_proposal_prism.pdf` §5.1–§5.2, run across five visual-QA benchmarks
(MathVista, HallusionBench, ChartQA, MMMU, RealWorldQA) using
`Qwen/Qwen3-VL-2B-Thinking` (a small vision-language "thinking" model that fits on
Kaggle's T4 GPUs). Goal: compute β_M ("beta-M", the marginal/controlled-direct
slope), β_T ("beta-T", the total slope), and Δβ ("delta-beta") = β_T − β_M, and see
whether reported grounding decay (β_M < 0, i.e. dependence trending downward across
steps) survives once prefix redundancy is controlled for (β_T ≈ 0 vs β_T < 0).

If that last sentence is dense, skip to **["The four conditions, in plain
language"](#the-four-conditions-in-plain-language)** below before the technical
table — it walks through the same idea slowly.

## Pipeline at a glance

```
STAGE 1  data_loading.py   download a benchmark, sample N examples, cache locally
STAGE 2  inference.py      per example: generate a real trajectory (image present)
                           and a blind trajectory (image blanked), then teacher-force
                           score every real step under four (image, prefix) conditions
STAGE 3  decomposition.py  turn per-step scores into M_s/T_s/C_s/S_s/R_s, per-example
                           slopes, and aggregate β_M / β_T / Δβ with bootstrap CIs
(extra)  plot_results.py   four descriptive PNGs comparing benchmarks
```

`run_benchmark.py` is the one-command wrapper used in practice: it launches STAGE 2
on both GPUs at once (each takes half the examples), then runs STAGE 3 and zips the
results.

## Codebase map

```
experiment_1/
├── config.yaml              All the settings: model, sampling, budget forcing,
│                             benchmark, sample size, output paths. Start here.
├── requirements.txt          Extra Python packages to install.
│
├── lib/                       Shared building blocks. Nothing here is meant to
│  │                           be run directly — the scripts below import from it.
│  ├── config.py               Loads config.yaml; DATASET_PRESETS (each benchmark's
│  │                           Hugging Face dataset id / split / column names).
│  ├── io_utils.py             Read/write the .jsonl files passed between stages.
│  ├── model.py                Loads the vision-language model onto a GPU/CPU.
│  ├── scoring.py              THE ACTUAL EXPERIMENT MATH: image ablation, step
│  │                           segmentation, trajectory generation (with budget
│  │                           forcing) and teacher-forced scoring.
│  ├── progress.py             Per-GPU "which example/stage is in flight" files.
│  └── checkpoint.py           Zips a benchmark's results + context for crash-safety.
│
├── data_loading.py    STAGE 1
├── inference.py       STAGE 2  The slow part.
├── decomposition.py   STAGE 3
├── plot_results.py             Figures from STAGE 3's outputs.
│
├── run_benchmark.py            Kaggle launcher: ONE benchmark, both GPUs, then
│                               decomposition + checkpoint. The main entry point.
├── run_multi_shard.py          Kaggle helper: one GPU process working through
│                               several benchmark "shards" without reloading the model.
├── checkpoint_now.py           Manually trigger a results backup (zip).
│
├── common.py                   Compatibility shim: old code imported everything
│                               from here before it was split into lib/. New code
│                               should use lib/*.
│
├── data/                       Generated: cached benchmark samples + images.
└── plots/                      Generated: PNGs from plot_results.py.
```

Results and checkpoints are generated under `/kaggle/working/` on Kaggle (see
["Output files"](#output-files)).

## How to run it

**Kaggle (the intended setup — dual T4, Save & Run).** One-time data caching per
benchmark (CPU-only, cheap), then one cell per benchmark:

```bash
!python experiment_1/data_loading.py --dataset mathvista --n-samples 100
!python experiment_1/run_benchmark.py --dataset mathvista --n-samples 100
```

`run_benchmark.py` options:

| Flag | Meaning |
|---|---|
| `--dataset` | Required. One of `mathvista`, `hallusionbench`, `chartqa`, `mmmu`, `realworldqa`. |
| `--n-samples N` | How many examples to run; overrides `data.n_samples`. Examples `[0:N/2]` go to `cuda:0`, `[N/2:N]` to `cuda:1`. Must not exceed what `data_loading.py` cached. |
| `--max-new-tokens N` | Cap on generated tokens per trajectory; overrides `model.max_new_tokens` (1536 by default). Use a small value for a smoke test. |
| `--config PATH` | Alternate config file (used by the decomposition step; see note below). |

Smoke test example:

```bash
!python experiment_1/run_benchmark.py --dataset mathvista --n-samples 10 --max-new-tokens 128
```

Note: `run_benchmark.py` does not forward `--config` to the two `inference.py`
processes, so a custom config only affects the decomposition step; edit
`config.yaml` (or use the CLI flags above) to change inference settings.

**What you see in the cell output.** While the GPUs work, `run_benchmark.py` prints
one line every 3 seconds with each GPU's latest log line (tqdm progress bar, warnings,
checkpoint notices), e.g.:

```
mathvista: cuda0: Scoring mathvista:  40%|████ | 10/25 | cuda1: Scoring mathvista:  36%|███▌ | 9/25
```

Full logs are in `/kaggle/working/gpu0_{benchmark}.log`, `gpu1_{benchmark}.log` and
`decomposition_{benchmark}.log`. Live status is also written to
`/kaggle/working/checkpoints/run_status.json`.

**Local / single-GPU:**

```bash
pip install -r requirements.txt -r experiment_1/requirements.txt
python experiment_1/data_loading.py --dataset mathvista
python experiment_1/inference.py --dataset mathvista --limit 3   # smoke test: 3 examples
python experiment_1/inference.py --dataset mathvista             # full run
python experiment_1/decomposition.py --dataset mathvista
python experiment_1/plot_results.py --datasets mathvista hallusionbench
```

`inference.py` also accepts `--device`, `--start`/`--end` (example slice) and
`--max-new-tokens`. It skips examples whose result file already exists, so an
interrupted run resumes where it stopped.

## What's implemented, following the proposal's §5.1 table exactly

For each step `s` of a trajectory (a "trajectory" here just means one full
step-by-step answer — a sequence of steps `y = (y_1, ..., y_S)`) generated with the
image present, four teacher-forced conditions are scored. ("Teacher-forced scoring"
means: instead of letting the model freely generate text, we feed it a specific
piece of text and ask "how probable did you think each of these words were?")

| Condition | Image | Prefix (the text already "said" before this step) | Log-likelihood |
|---|---|---|---|
| C1 | I (present) | y_<s (the real preceding steps) | ℓ₁ |
| C2 | Ĩ (ablated/blanked out) | y_<s | ℓ₂ |
| C3 | I (present) | scrub(y_<s) (a stand-in prefix with no real visual info) | ℓ₃ |
| C4 | Ĩ (ablated) | scrub(y_<s) | ℓ₄ |

("Log-likelihood", ℓ, is a number that goes up when the model found the actual
words more predictable/expected, and down when it found them surprising. It's
measured in **nats/token** — the log-probability unit from natural logarithms, like
"bits" but base-*e*; higher = the model was less surprised. "/token" means it's
averaged per word-piece, so short and long steps are comparable.)

```
M_s = ℓ₁ - ℓ₂           controlled direct effect ("marginal" — what most prior work measures)
T_s = ℓ₁ - ℓ₄           total visual dependence
C_s = T_s - M_s = ℓ₂ - ℓ₄   prefix-carried / indirect dependence
S_s = ℓ₃ - ℓ₄           visual sufficiency
R_s = C_s / max(T_s, ε)     redundancy ratio, clipped to [0,1]
```

`β_M`, `β_T` are slopes ("slope" = how much a value trends up or down as you move
from the first step of an answer to the last) of `M_s`, `T_s` on normalised step
index `ŝ = s/S` (so step 1 of a 4-step answer and step 3 of a 12-step answer can both
be compared as "25% of the way through") per proposal §5.2. `Δβ = β_T - β_M`.

### The four conditions, in plain language

Imagine the model has just written steps 1 and 2 of its answer, and we want to
check: how much is step 3 actually informed by the picture, versus just following
logically from what it already wrote?

- **C1 (real image, real prefix):** the normal case — show the model the real
  picture and what it actually said in steps 1–2, then check how "expected" step 3
  was. This is the baseline everything else compares against.
- **C2 (blanked image, real prefix):** take away the picture, but keep the real
  steps 1–2. If step 3 is *still* just as expected, steps 1–2 alone already gave the
  model everything it needed — the image wasn't doing much work for this step, at
  least not directly.
- **C3 (real image, fake prefix):** give the model the real picture back, but swap
  steps 1–2 for placeholder text that carries no real visual information (see
  "scrub" below). If step 3 is still expected, the image alone is *sufficient* to
  predict it, without needing the earlier steps.
- **C4 (blanked image, fake prefix):** take away both the picture and the real
  prefix. This is the "worst case" baseline for the other three.

The whole point of running all four (not just C1 vs C2) is this: if you only compare
C1 to C2 and see that removing the image barely changes anything, you might conclude
"the model isn't using the image for this step." But that's ambiguous — maybe the
model isn't using the image *directly* anymore because its own earlier sentences
already smuggled the visual information forward (the image's influence got
"carried" into the prefix). C3/C4 separate those two explanations: `M_s` (from
C1/C2) mixes both effects together, while `T_s` (from C1/C4) isolates the model's
*total* dependence on the image with no help from a real prefix. `C_s = T_s - M_s`
is the part flowing through the prefix rather than the image directly. `R_s` turns
that into a 0–1 fraction: close to 1 means most of the apparent visual dependence is
redundant with what the prefix already says; close to 0 means the image is still
doing independent work at that step.

## What "scrub" means, concretely

"scrub(y_<s)" = a prefix that looks like a normal partial answer but carries no real
visual information tied to the actual image. This pilot builds it by generating a
**second, independent** answer to the same question with the image blanked out from
the very start (the "blind trajectory", z), and using its first few steps as the
stand-in prefix: `scrub(y_<s)` = the first `min(s-1, len(z))` steps of z
(`lib/scoring.py::cumulative_text`, `inference.py::scrub_prefix`).

## How trajectories are generated

- **Model:** `Qwen/Qwen3-VL-2B-Thinking`, fp16 (T4 GPUs have no bf16 support). A
  "Thinking" model writes its reasoning inside a `<think>` block, then a final answer.
  The measured trajectory is the **whole** output, reasoning and final answer
  together, with only the `</think>` marker dropped
  (`lib/scoring.py::extract_trajectory_text`).
- **Prompt:** the question plus "Solve this step by step, showing your reasoning for
  each step, then give your final answer." The identical prompt is used for
  generation and for scoring, so ℓ₁..ℓ₄ are computed under the exact prompt that
  produced the text.
- **Prompt construction:** a partial trajectory is appended to the chat-template
  prompt as a raw string, never as a second assistant message. Thinking-model chat
  templates re-wrap assistant content with a synthetic closing `</think>`, which
  would silently change the context being scored.
- **Sampling, not greedy:** temperature 0.6, top-p 0.95, top-k 20 (Qwen's
  recommended thinking-mode settings; greedy decoding caused endless repetition
  loops), plus `no_repeat_ngram_size: 4`. Each generation is seeded from
  `generation_seed` + example id + `real`/`blind`, so re-runs reproduce the same
  trajectories without every example sharing one random draw.
- **Token cap and budget forcing:** `model.max_new_tokens` (1536; override with
  `--max-new-tokens`) bounds each trajectory. Blind (no-image) generations were
  observed to ramble through an entire 8192-token budget without converging, so
  generation is two-stage ("budget forcing", s1-style): stage 1 generates
  `stage1_fraction` (75%) of the cap; if the model hasn't reached a final answer
  (a "Final Answer" phrase or `\boxed{}`), an interrupt ("I need to stop reasoning
  now and give my final answer: **Final Answer:**") is spliced in and stage 2 uses
  the remaining budget to force a conclusion. If the model finishes on its own in
  stage 1, nothing is forced.
- **Steps:** the trajectory is split into steps by sentence boundaries / newlines;
  steps under 4 characters are dropped (`lib/scoring.py::segment_steps`).
- **Ablation:** the blanked image Ĩ is a flat image filled with the real image's
  mean color (`mean_pixel_fill`), same size as the original so the number of vision
  tokens, and therefore prompt structure, is identical across conditions.

## The five benchmarks, and why these

Each tests a different flavor of "does the model actually need to keep looking at
the picture":

- **MathVista** (`AI4Math/MathVista`, `testmini`) — math problems referencing a
  diagram, chart, or figure. The "sanity anchor": the benchmark the original
  proposal used, so it gives a baseline to compare the others against.
- **HallusionBench** (`lmms-lab/HallusionBench`, `image`) — questions designed to
  tempt a model into a plausible but wrong answer if it stops checking the image and
  pattern-matches from text. Directly targets the failure mode being measured.
- **ChartQA** (`lmms-lab/ChartQA`, `test`) — reading values off charts. Requires
  precise, sustained visual reading (an exact number can't be guessed from general
  knowledge), a stress test for whether decay is worse when the task can't be
  solved from memory alone.
- **MMMU** (`lmms-lab/MMMU`, `validation`) — college-level multi-discipline
  questions. The `test` split has no public answers and `dev` is tiny (150), so
  `validation` is used. Questions can carry up to 7 images; the first non-empty
  image field is used.
- **RealWorldQA** (`lmms-lab/RealWorldQA`, `test`) — questions about real-world
  photographs, where general knowledge can often substitute for looking.

A sixth candidate, **Visual-CoT**, was dropped: its Hugging Face loader throws a
schema-mismatch error and its images are split across ~139GB of undocumented tar
archives — not worth a fragile pipeline for a pilot.

Which benchmark a run uses is set by `data.dataset` in `config.yaml` or the
`--dataset` flag; `lib/config.py::DATASET_PRESETS` holds each one's Hugging Face id,
split and column names. `data_loading.py` normalizes every benchmark into the same
pid/question/answer/image_path format, so **the benchmark never changes the scoring
math** — `inference.py` and `decomposition.py` are benchmark-agnostic. Benchmarks
without a natural unique id use the row index as the pid.

## Design decisions made for this pilot

Each is a deliberate choice at a point the proposal leaves open or flags as risk:

1. **Four-condition harness, not two.** A plain with/without-image comparison only
   yields M_s. T_s and Δβ need C3/C4, which need a scrub operator.
2. **Scrub operator = blind regeneration** (proposal §5.3's "cleanest causally"
   option).
3. **Matched by step index, not token count.** The proposal says "a prefix of
   matched length" without specifying. Step index is simplest to implement
   correctly; if z has fewer than s-1 steps the scrub prefix is all of z. Revisit if
   real/blind step counts diverge a lot — `n_steps` for both trajectories is stored
   per example in `per_example/*.json`, so this is auditable.
4. **Image ablation = mean-pixel fill**, the simplest of the proposal's five
   operators (§5.4). The headline operator, region-blur, needs a question-relevant
   region localizer and is out of scope. (`gaussian_noise` is also implemented.)
5. **Step segmentation = sentence split** of free-text chain-of-thought (the
   proposal defines steps abstractly).
6. **Aggregation = per-example OLS slope, then mean + percentile bootstrap CI
   across examples** (2000 resamples), not the proposal's full mixed-effects model
   with random intercepts by item and model (§5.2): this pilot runs one model and a
   single random-effects axis adds a `statsmodels` dependency for little benefit at
   ~100 examples. A pooled (unclustered) OLS slope is reported as a secondary
   check. Upgrade to `statsmodels.MixedLM` for the multi-model sweep.
7. **Sample = 100 items per benchmark** by default (`data.n_samples`), sampled with
   a fixed seed so re-caching gives the same rows.
8. **Thinking model + sampling + budget forcing** (see "How trajectories are
   generated"): chosen after greedy decoding and an 8192-token cap produced runaway
   repetition and non-terminating blind trajectories.

## Explicitly out of scope for this pilot

Real proposal content that belongs to later phases:
- Scrub-fidelity human audit / Gate G1 (§5.3)
- The necessity × sufficiency taxonomy and confabulation rate (§5.5)
- The policy-independent demand estimator `d_s` and PRISM-Minimal (§5.6–5.7)
- Multi-model sweep and mixed-effects regression (Phase 2)
- Region-blur ablation and the rest of the ablation × scrub operator grid
- Visual-CoT

## Output files

Paths are per-benchmark (`{dataset}` is e.g. `mathvista`). With the default config
they live under `/kaggle/working/`:

- `results/results_{dataset}/per_example/{pid}.json` — real & blind trajectories and
  steps, `n_steps`, and per-step ℓ₁..ℓ₄.
- `results/results_{dataset}/aggregate/per_step_measures.csv` — every step's M_s,
  T_s, C_s, S_s, R_s, ŝ.
- `results/results_{dataset}/aggregate/per_example_slopes.csv` — per-example β_M,
  β_T, Δβ, mean R_s.
- `results/results_{dataset}/aggregate/summary.json` — aggregate β_M/β_T/Δβ with
  bootstrap CIs, the pooled-OLS check, mean redundancy ratio, and a mechanical read
  of the proposal's §5.2 three-way outcome (real decay / redundancy artifact /
  mixed) — treat it as a pointer to inspect, not a conclusion.
- `checkpoints/{dataset}_{label}_{timestamp}.zip` — crash-safety backups.
- `checkpoints/run_status.json` and `checkpoints/progress_{device}.json` — live run
  status.
- `experiment_1/plots/*.png` — from `plot_results.py`: `per_step_trends`, `beta_ci`,
  `beta_scatter`, `redundancy_trend`.

## Crash-safety and checkpoints

On Kaggle, `/kaggle/working/checkpoints/` fills with zipped snapshots automatically:
- every 10 completed examples *or* every 10 minutes (`*_partial_*.zip`),
- when each GPU's half finishes (`*_shard_done_*.zip`),
- when the whole benchmark finishes (`*_complete_*.zip`).

Each zip contains the results directory plus the effective `run_config.json`,
`meta.json` (timestamp, git commit and dirty flag, package versions, GPU info), the
cached data sample, `run_status.json`, and the GPU logs — enough to trace results
back to exact code, settings and inputs.

Trigger one manually (e.g. before stopping a run early):

```bash
python experiment_1/checkpoint_now.py                            # all benchmarks
python experiment_1/checkpoint_now.py --dataset hallusionbench   # just one
```

**Important:** these zips only protect you *within* a live session. Kaggle only
permanently attaches `/kaggle/working` to your account when a "Save Version" is
committed (or the session ends normally). Commit periodically, or download the
zips, if a run matters.

## Other launchers

- `inference.py` — one shard on one GPU. Good for local runs and smoke tests.
- `run_multi_shard.py` — one process per GPU working through several
  `dataset:start:end` shards with the model loaded once, e.g.
  `python experiment_1/run_multi_shard.py --device cuda:0 --shard mathvista:0:100 --shard hallusionbench:0:50`.
  Use it when you want to assign work to GPUs by hand across benchmarks.
- `run_benchmark.py` — the turnkey option described above (reloads the model on
  each launch, in exchange for a simple even split and automatic decomposition).
