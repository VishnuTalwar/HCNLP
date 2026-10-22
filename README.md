# Prompt Sensitivity Analysis Across LLM Families

**A Taxonomy of Prompt Sensitivity**

Group 13 · Human-Centred NLP (HCNLP 2026) · Otto-von-Guericke-Universität Magdeburg (OvGU)

**Authors:** Vishnu Talwar, Mudit Khandelwal, Neha Lakshman Konchady

---

## Table of Contents

1. [Overview](#overview)
2. [Research Questions](#research-questions)
3. [Repository Contents](#repository-contents)
4. [Experimental Design](#experimental-design)
   - [Models](#models)
   - [Tasks and Datasets](#tasks-and-datasets)
   - [Perturbation Taxonomy](#perturbation-taxonomy)
   - [Metrics](#metrics)
   - [Statistical Testing](#statistical-testing)
5. [Pipeline Walkthrough](#pipeline-walkthrough)
6. [Setup and Installation](#setup-and-installation)
7. [Running the Experiments](#running-the-experiments)
8. [Output Files](#output-files)
9. [Reproducibility](#reproducibility)
10. [Methodological Notes and Known Limitations](#methodological-notes-and-known-limitations)
11. [Version History](#version-history)
12. [Security: Handling API Tokens](#security-handling-api-tokens)
13. [Authors and Acknowledgements](#authors-and-acknowledgements)

---

## Overview

Large language models can change their answers when a prompt is reworded in ways that should not matter: a typo, a different role preamble, a changed separator, or a moved instruction. This project measures **how sensitive** open-weight LLMs are to such changes, and organizes the perturbations into a **taxonomy** (lexical, structural, contextual, formatting, rewrite).

We evaluate three model families (Llama 3.1, Gemma 2, Qwen 2.5), each in an **instruct** and a **base** variant, on three classification-style tasks. For each task we hold the example body (premise, question, choices) fixed and perturb **only the instruction text**, then compare predictions against an unperturbed baseline.

Predictions are made by **scoring the log-probabilities of the candidate label tokens** (e.g. `A`/`B`/`C`) rather than free-form generation. This avoids parsing errors and gives a per-example confidence margin.

## Research Questions

1. Which kinds of prompt perturbation (lexical, structural, contextual, formatting, rewrite) change model predictions the most?
2. Do different model families (Llama, Gemma, Qwen) show the same sensitivity pattern? (cross-model agreement)
3. Are instruction-tuned models more or less sensitive than their base counterparts?
4. Are observed accuracy changes statistically significant, or within noise?
5. Does lexical or semantic distance between the original and perturbed instruction predict sensitivity?

## Repository Contents

| File | Description |
|---|---|
| `hcnlp-v7 (2).ipynb` | Main notebook containing the full pipeline: setup, data loading, perturbation engine, inference, evaluation, statistics, and analysis. |
| `HCNLP_PPT (1).pdf` | Project presentation (PDF). |
| `HCNLP_PPT_googleSlide_compatible.pptx` | Project presentation (PowerPoint / Google Slides compatible). |
| `README.md` | This file. |

Directories created when the notebook runs (not committed by default, see [Output Files](#output-files)):

```
perturbation_cache/      # precomputed perturbed prompts (built once)
results/                 # per-model results, summaries, CSV exports
  ├── <model_key>/       # baseline + per-perturbation JSON, SUMMARY.json
  ├── paired/            # paired baseline-vs-perturbed CSVs
  ├── all_metrics.csv
  ├── stats.csv
  └── prompt_templates.csv
```

## Experimental Design

### Models

All models are loaded from the Hugging Face Hub in **4-bit NF4 quantization** (bitsandbytes, double quantization, fp16 compute).

| Registry key | Hugging Face ID | Type | Params |
|---|---|---|---|
| `llama3.1-8b-instruct` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Instruct | 8B |
| `gemma2-9b-it` | `google/gemma-2-9b-it` | Instruct | 9B |
| `qwen2.5-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` | Instruct | 7B |
| `llama3.1-8b-base` | `meta-llama/Meta-Llama-3.1-8B` | Base | 8B |
| `gemma2-9b-base` | `google/gemma-2-9b` | Base | 9B |
| `qwen2.5-7b-base` | `Qwen/Qwen2.5-7B` | Base | 7B |

Instruct models are prompted through each tokenizer's native `apply_chat_template`. Base models receive the raw prompt. **Base models are treated as an appendix comparison**, since some perform near chance on these tasks.

> Llama 3.1 and Gemma 2 are gated models. You must accept their licenses on Hugging Face and use an account with access.

### Tasks and Datasets

| Task key | Dataset | Labels | Examples |
|---|---|---|---|
| `mnli` | [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli), `validation_matched` | A = entailment, B = neutral, C = contradiction | 300 (100 per class, label-balanced) |
| `bbh_date` | [BIG-Bench Hard](https://huggingface.co/datasets/lukaemon/bbh), `date_understanding` | Multiple choice (A–F, read per example) | 250 (full set) |
| `arc` | ARC (multiple choice), labels read from each example's `choices` | Per-example (numeric labels remapped to letters) | 300 |

Each prompt is split into a **body** (task content, never perturbed) and an **instruction** (the directive text at the end, the only part perturbations touch). This guarantees that any change in behavior is attributable to the instruction wording or format, not to altered task content.

### Perturbation Taxonomy

Nine perturbations in five categories, plus the unperturbed baseline:

| Category | Perturbation | What it does |
|---|---|---|
| Baseline | `baseline` | Original instruction. |
| Lexical | `lex_rewrite` | Hand-audited lexical variants of instruction wording per task (label mapping preserved). |
| Lexical | `lex_typo` | Random character substitutions (nlpaug `RandomCharAug`, ~5% of characters). |
| Lexical | `lex_lowercase` | Lower-cases the instruction (formality change). |
| Structural | `struct_inst_first` | Moves the instruction before the body. |
| Contextual | `ctx_role` | Prepends a role preamble. |
| Contextual | `ctx_irrelevant` | Adds irrelevant context. |
| Formatting | `fmt_caps` | Capitalization/format emphasis changes. |
| Formatting | `fmt_separators` | Changes separators between prompt segments. |
| Rewrite | `instruction_rewrite` | Full rewrite of the instruction, preserving label semantics (e.g. MNLI keeps entailment / neutral / contradiction). |

All perturbations are computed once and stored in `perturbation_cache/` (see [Reproducibility](#reproducibility)). A perturbation audit file (`results/prompt_templates.csv`) lists the exact instruction template used for each task and perturbation for manual review.

### Metrics

Per model, task, and perturbation:

| Metric | Definition |
|---|---|
| `base_acc` / `pert_acc` | Accuracy on the unperturbed / perturbed prompts. |
| `acc_drop` | `base_acc − pert_acc` (signed; negative means the perturbation *helped*). |
| `flip_rate` | Fraction of examples whose predicted label differs from the baseline prediction. |
| `sensitivity_score` (SS) | `flip_rate × \|acc_drop\|`. Bounded and interpretable. |
| `d_sem` | Cosine distance between `all-MiniLM-L6-v2` embeddings of the original and perturbed **instruction** strings. Reported separately, not folded into SS. |
| `margin` | Top log-prob minus second log-prob for each prediction (confidence). Used to detect confidence drops even when accuracy is unchanged. |
| `direction` | Whether the perturbation lowered or raised accuracy. |
| Collapse flag | Marks runs where a single label accounts for more than 90% of predictions. |

**Interpreting SS:** because it uses `|acc_drop|`, improvements and degradations both contribute. It measures *sensitivity*, not fragility. Always read it alongside `direction` and the signed `acc_drop`; do not rank "most fragile" by SS alone.

`d_sem` is `0` for perturbations that change the structure but not the wording of the instruction (`struct_inst_first`, `fmt_separators`). This is expected, since the instruction string itself is unchanged.

### Statistical Testing

- **Bootstrap confidence intervals** (2000 resamples) on accuracy and on the paired accuracy drop.
- **McNemar's test** on paired baseline-vs-perturbed correctness, with **Holm–Bonferroni** correction across comparisons.
- Each test reports `n_discordant`, effect direction, and an **underpowered** flag (`n_discordant < 20`).
- **Cross-model agreement** between instruct models computed with Spearman (primary) and Pearson correlation of per-perturbation sensitivity.

## Pipeline Walkthrough

The notebook is organized into numbered cells:

| Cell | Purpose |
|---|---|
| 1–2 | Install pinned dependencies and verify versions. |
| 3 | Hugging Face authentication. |
| 4 | Imports, seeds, example counts, quantization config, `MODEL_REGISTRY`, perturbation names/categories. |
| 5 | Model loading and memory cleanup between models. |
| 6 | Inference pipeline: chat-template tokenization, label-token scoring via log-probs. |
| 7 | Load and sample datasets (MNLI, BBH date understanding, ARC). |
| 8 | Prompt templates, split into body / instruction. |
| 9 | Perturbation engine. |
| 10 | Perturbation cache: build once, auto-rebuild on format/name changes. |
| 11 / 11a | Semantic distance and sensitivity score; single-token label verification. |
| 12 | Evaluation loop (stores predictions, log-probs, margins). |
| 13 | Experiment runner, paired CSV export, collapse diagnostics, perturbation audit export. |
| 14–18 | Sanity check, then runs for Llama, Gemma, Qwen (instruct) and the three base models. |
| 19 / 19a | Results summary and flat-file exports for plotting. |
| 20 | Instruct vs. base comparison (with chance-level competence warnings). |
| 21 | Cross-model agreement. |
| 22 | Bootstrap CIs, McNemar test, Holm–Bonferroni. |
| 23 | Qualitative error analysis (flip failures and confidence drops > 0.5 nats). |
| 24 | Chat-template sanity check (expects 0 disagreements). |
| After | Zip export helper and exploratory analysis cells that read the saved CSV/JSON files. |

## Setup and Installation

### Requirements

- Python 3.10+
- A CUDA-capable GPU. 4-bit quantized 7–9B models need roughly 8–12 GB VRAM each, and models are loaded one at a time.
- A Hugging Face account with access to the gated Llama 3.1 and Gemma 2 models.

### Dependencies

Pinned in the first notebook cell:

```
transformers==4.46.3
huggingface_hub==0.26.2
tokenizers==0.20.3
accelerate==1.0.1
datasets==3.1.0
```

Also used (installed in Cell 1): `torch`, `bitsandbytes`, `sentence-transformers`, `nlpaug`, `nltk`, `scipy`, `numpy`, `pandas`, `jinja2`, `tqdm`.

```bash
pip install transformers==4.46.3 huggingface_hub==0.26.2 tokenizers==0.20.3 \
            accelerate==1.0.1 datasets==3.1.0 \
            torch bitsandbytes sentence-transformers nlpaug nltk \
            scipy numpy pandas jinja2 tqdm jupyter
```

### Proxy configuration

The notebook was developed behind a corporate HTTP proxy, and the proxy URL is set in Cells 1 and 3 (`PROXY = "..."`). **If you are not behind that proxy, remove or edit these lines** (and the `--proxy` pip flag and `nltk.set_proxy` call), or downloads will fail.

### Hugging Face token

Do **not** hardcode a token in the notebook. Set it in your environment before launching Jupyter:

```bash
export HF_TOKEN=hf_your_token_here
jupyter notebook "hcnlp-v7 (2).ipynb"
```

If `HF_TOKEN` is unset, Cell 3 prompts for it interactively.

## Running the Experiments

1. Open `hcnlp-v7 (2).ipynb`.
2. Run Cells 1–2, then **restart the kernel** (required after the dependency install).
3. Run Cells 3–13 to set up auth, config, data, perturbations, and the cache. The first run builds `perturbation_cache/`.
4. (Optional) Run Cell 14 as a sanity check on one model.
5. Run Cells 15–17 for the three instruct models, and Cell 18 for the base models. Each model is loaded, verified (`verify_single_token_labels`), evaluated on all tasks and perturbations, then unloaded.
6. Run Cells 19–24 for summaries, comparisons, statistics, and error analysis.
7. Use the final analysis cells to explore `results/*.csv`. These cells only read saved outputs and don't need a GPU.

> Runtime is dominated by inference: 3 tasks × 10 prompt variants × (300 + 250 + 300) examples per model. Results are saved to disk per model, so an interrupted run does not need to repeat finished models.

## Output Files

| Path | Contents |
|---|---|
| `perturbation_cache/` | Cached perturbed prompts with a content fingerprint. |
| `results/<model_key>/<task>_baseline.json` | Baseline predictions, log-probs, margins. |
| `results/<model_key>/<task>_<perturbation>.json` | Per-perturbation predictions. |
| `results/<model_key>/SUMMARY.json` | Aggregated metrics for that model. |
| `results/paired/<model>_<task>_<perturbation>.csv` | Paired baseline vs. perturbed predictions with correctness and margins. |
| `results/all_metrics.csv` | Flat table: model, task, perturbation, accuracies, drop, flip rate, `d_sem`, SS, direction. |
| `results/stats.csv` | Bootstrap CIs, McNemar p-values (raw and adjusted), significance and underpowered flags. |
| `results/prompt_templates.csv` | Perturbation audit: exact instruction templates and semantic distances. |
| `hcnlp_experiment_data.zip` | Optional archive of `results/` and `perturbation_cache/` (Cell: `create_export_zip`). |

Expected sizes as a sanity check: `all_metrics.csv` and `stats.csv` contain 9 perturbations × 3 tasks × 3 instruct models = **81 rows** each.

## Reproducibility

- Global seed `42` is set for `random`, `numpy`, and `torch`.
- Dataset sampling uses an isolated `numpy.random.default_rng(42)`, so other RNG use cannot change the sample.
- MNLI is class-balanced (100 per label) and shuffled so that early-example checks cover all labels.
- Perturbations (including random typos) are generated **once** and cached to disk. After the cache exists, the seed no longer affects results; delete `perturbation_cache/` to regenerate.
- The cache has a format version and fingerprint. It rebuilds automatically if the format or perturbation names change.
- Library versions are pinned (see above). Quantized inference on different GPUs or bitsandbytes builds can still produce small numeric differences.

## Methodological Notes and Known Limitations

- **Log-prob scoring, not generation.** Predictions are the highest-scoring candidate label token. Every candidate label must encode to a single token, which `verify_single_token_labels` checks after each model load.
- **Instruction-only perturbation.** The body is never modified, so results reflect instruction sensitivity only.
- **Sample sizes.** 250–300 examples per task give wide confidence intervals for small effects. Many comparisons are flagged underpowered.
- **Quantization.** All models run in 4-bit, which may slightly change sensitivity relative to full precision.
- **Base models.** Some base models are near chance on these tasks (flagged in the output), so their sensitivity numbers are of limited interpretive value.
- **`d_sem` scope.** It compares instruction strings only, not full prompts.
- **Scope.** Three families at 7–9B and three task types; conclusions may not generalize to larger models, other languages, or generative tasks.

## Version History

**v7**
- Label scoring uses a shared `continuation_token_ids()` helper so the scored token matches what the model generates after the chat template (bare-letter, not space-prefixed).
- Preflight label check rewritten to use the same helper.
- MNLI sample shuffled so early sanity checks are not single-label.

**v6.1**
- `format_and_tokenize` uses `apply_chat_template(tokenize=True)` to avoid the double-BOS problem.
- `semantic_distance` renamed `instruction_semantic_distance`.

**v6**
- Hardcoded chat templates replaced with `tokenizer.apply_chat_template()` (Llama showed disagreements with its native template on 14/50 MNLI, 10/50 BBH, and 19/50 ARC examples).
- Removed `struct_inst_last` (identical to baseline, flip rate 0 everywhere).
- Replaced the `1/(d_sem+eps)` fragility index with `sensitivity_score = flip_rate × |acc_drop|`.
- Replaced nlpaug WordNet synonyms (produced nonsense such as "Solvent" for "Answer") with hand-audited lexical variants.
- Fixed MNLI rewrite so it preserves label names; renamed `para_rewrite` to `instruction_rewrite`.
- Added collapse diagnostics, perturbation audit CSV, and enhanced stats output.
- Base models reframed as an appendix comparison.

## Security: Handling API Tokens

Never commit Hugging Face tokens or other credentials. GitHub secret scanning will block pushes that contain them. Use the `HF_TOKEN` environment variable (or the interactive prompt) instead. If a token was ever committed or shared, **revoke it** at <https://huggingface.co/settings/tokens> and create a new one.

## Authors and Acknowledgements

- **Vishnu Talwar**
- **Mudit Khandelwal**
- **Neha Lakshman Konchady**

Human-Centred NLP, Otto-von-Guericke-Universität Magdeburg. Built on Hugging Face Transformers, Datasets, MultiNLI, BIG-Bench Hard, ARC, Sentence-Transformers, and nlpaug.
