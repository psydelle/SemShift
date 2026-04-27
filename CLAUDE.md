# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
uv sync
uv run python -c "import nltk; nltk.download('wordnet')"
uv run python -m spacy download en_core_web_sm
```

Run any script with `uv run python <script>.py`. The venv is managed by `uv`; do not use `pip install` directly.

## Running the pipeline

```bash
# On EIDF cluster — extract KWICs from Sketch Engine corpus
bash eidf_get_kwics.sh

# Locally or on cluster — compute all delta/LSCD metrics
uv run python compute_deltas.py

# Analysis notebook
uv run jupyter notebook notebooks/delta_analysis.ipynb
```

`compute_deltas.py` runs in ~3 seconds locally on CPU. The `.pt` files it reads are large (verbs.pt ~2.6 GB, nouns.pt ~5.2 GB) and must be present at `output/output_verbs/verbs.pt` and `output/output_nouns/nouns.pt`.

## Architecture

### Pipeline overview

```
Sketch Engine corpus
      ↓  (eidf_get_kwics.sh → build_comprehensive_kwic_dataset.py)
KWICs + raw embeddings stored as .pt files
      ↓  (compute_deltas.py)
output/verb_noun_deltas.csv   ← one row per stimuli item, all LSCD metrics
      ↓  (notebooks/delta_analysis.ipynb)
Statistical analysis + figures
```

### Key files

| File | Role |
|------|------|
| `compute_deltas.py` | Loads `.pt` files, computes all LSCD metrics, writes `output/verb_noun_deltas.csv` |
| `analysis.R` | All statistical models (t-tests, mixed models, OLS). Run with `Rscript analysis.R` |
| `scripts/` | One-off corpus/pipeline scripts (KWIC extraction, embedding, preprocessing). Not needed for analysis. |
| `data/stimuli.csv` | Seed verb-noun pairs used to query Sketch Engine |
| `data/iRT_AJT.csv` | Item-level stimuli with mean RT per item (one row per verb-noun pair) |
| `data/experiment_data_anonymised.csv` | Trial-level psycholinguistic data (~4,457 trials, 230 participants) |
| `output/figures/` | Generated plots from the notebook |

### `.pt` file structure

Each `.pt` file is a `dict[str, dict]` keyed by `"verb noun"` (e.g. `"kill time"`). Each record contains:

- `verb`, `noun`: strings
- `verb_embeddings`: `torch.Tensor` of shape `(n_kwics, 384)` — one SBERT embedding per KWIC sentence
- `noun_embeddings`: same shape for the noun token
- `kwics`, `kwic_words`, `failed_indices`: raw sentences and metadata

`verbs.pt` (~7,884 pairs): verb was the Sketch Engine seed — used for `X[verb]`, `Y[verb]`, `Y[noun]`.
`nouns.pt` (~15,934 pairs): noun was the seed — used only for `X[noun]` (richer noun prototype).

### LSCD metrics in `compute_deltas.py`

All metrics are computed in the same two-pass loop over the `.pt` files:

- **`verb_delta_cos` / `noun_delta_cos`**: cosine similarity between the word's prototype `X[word]` (mean across all pairings) and its item-specific mean `Y[word]`. High = conventional use, low = semantic shift.
- **`verb_apd` / `noun_apd`**: Average Pairwise Distance across the `(n_kwics, 384)` embeddings for a specific pairing. Measures within-pairing meaning variability.
- **`verb_n_clusters` / `noun_n_clusters`**: optimal k from k-means (k ∈ 2–5, selected by silhouette score) over the KWIC embeddings. Effective sense count.
- **`verb_samd`**: Symmetric Average Minimum Distance between the verb's KWIC embeddings in its Collocation pairing vs its Productive pairing. Measures collocation-induced semantic bleaching for the verb. Each verb has exactly one Collocation item and one Productive item (minimal pair design, e.g. *kill time* vs *kill rabbit*), so `verb_samd` is the same value for both rows of a verb.

## Research context

**Project**: Semantic mutability of verbs vs nouns in English verb-noun combinations. Target venue: EMNLP.

**Stimuli design**: ~190 verb-noun pairs. Each verb appears in exactly two items — one Collocation (e.g. *kill time*) and one Productive (e.g. *kill rabbit*). This minimal-pair design enables within-verb comparisons.

**Embedding model**: SBERT `sentence-transformers/all-MiniLM-L6-v2` (384-dim) via Flair `TransformerWordEmbeddings`. Contexts sourced from Sketch Engine (BNC/ukWaC).

**Research questions**: RQ1 — are verbs more mutable than nouns, and does the gap differ by condition? RQ2 — do LSCD metrics predict acceptability judgement RT, and does this interact with condition?

**Psycholinguistic data**: `RT_AJT` is trial-level raw RT. `iRT_AJT` is item-level mean (same value across all participants for one item). For mixed models use `RT_AJT` filtered to `Response_AJT == 'y'` (accepted trials only) with 350ms floor and 3-SD per-participant ceiling trim.

**Known exclusions**: 3 items missing from `.pt` files — *beat odds*, *dent cans*, *launder clothes*.

**Pool size**: `verb_pool_size` and `noun_pool_size` are the number of distinct pairing contexts used to build the prototype. Items with pool < 50 have noisier prototype estimates. Results hold at pool ≥ 50.

## Agent

`.github/agents/embedding-analysis-mentor.agent.md` defines a specialist agent for embedding analysis and linguistic interpretation. Its constraints (no fabricated numbers, evidence-trail for every claim, effect sizes alongside p-values) apply to all analysis work in this repo.

## Toolchain rules

- **R only for statistical modelling** (t-tests, mixed models, OLS). All statistical analysis lives in `analysis.R`. Run it with `Rscript analysis.R` from the project root.
- **Python for everything else**: data wrangling, metric computation, embedding manipulation, visualisation.
- The notebook (`notebooks/delta_analysis.ipynb`) handles data prep and plots; it calls `Rscript analysis.R` via subprocess for model results.
- The user is a linguistics domain expert and a coding novice — explain technical outputs in linguistically meaningful terms and always provide traceable evidence (file → code → result → interpretation).
- ggplot2 in R is preferred for visualizations due to its flexibility and publication-quality output. Avoid using Python plotting libraries for final figures, but they can be used for exploratory data analysis if needed.Plots should embody best practices: clear labels, appropriate scales, error bars where relevant, and a clean aesthetic suitable for academic publication. Coonsistent color choices across figures, descriptive axis labels and clear legends should be prioritised. Consider facets or separate panels for different conditions or word types to enhance readability. Do not include a caption as part of the plot code; captions should be written separately in the paper text.
