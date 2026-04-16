# Sprint M05 — Semantic Axes

Build a **semantic map** of a domain you find interesting. You will pick a
list of terms, design two semantic axes from opposing word sets, and
produce one publication-quality 2D scatterplot that tells a story about
the terms.

> [!IMPORTANT]
> **Fork this repo first.** If you skip this step, your Codespace will open
> on the original repo and you will have no way to save or push your work.
> Fork (top-right) → clone your fork → do all work on your fork.

## Getting oriented

1. Read this README end-to-end.
2. Open `assignment.py` with `uvx marimo edit --sandbox assignment.py`.
   This is a **worked example** — not the deliverable. It walks through
   the full pipeline on the U.S. universities dataset so you can see what
   "done" looks like.
3. Pick a case study (see below), then build your own pipeline from
   scratch in a new notebook / script.

## Pick a case study

Choose **one** of the three provided datasets, or bring your own.

| File | Case study | N | Extra columns |
|---|---|---|---|
| `data/universities.csv` | U.S. higher-education institutions | 157 | `type`, `region` |
| `data/sp500.csv` | S&P 500 (sample) companies | 203 | `sector` |
| `data/chemicals.csv` | Chemical compounds and materials | 179 | `class` |

**Bring your own?** Fine — with two constraints:

- **≥ 100 terms.** Below that, the scatterplot gets too sparse to tell a
  useful story.
- **At least one categorical attribute per term** that you can encode as
  color or shape in the scatterplot.

## Your tasks

### 1. Design two semantic axes

For each axis, choose 3–6 words for the **+ pole** and 3–6 words for the
**− pole**. A good axis:

- Has well-separated poles (cosine distance between pole centroids ≥ 0.3).
- Spreads your dataset out rather than piling points near the midpoint.
- Has an interpretation you can state in one sentence.

The two axes should capture **different, ideally orthogonal** aspects of
your data. Two axes that measure almost the same thing waste half the
plot.

### 2. Produce one scatterplot

Plot every term at `(axis1_score, axis2_score)`. Use **color** and
**marker shape** to encode categorical attributes of the terms (e.g.,
sector, type, region, class). Good scatterplots follow the data-viz
principles you've already learned:

- **Clarity.** All symbols, lines, and text are easily readable. No
  overlapping labels. No default matplotlib-blue soup.
- **Group separability.** Different groups are visually distinct — you
  (and your reader) can tell them apart at a glance.
- **Colorblind-friendly.** Use a palette that survives red-green
  colorblindness (Okabe–Ito, viridis, or similar). Redundantly encode
  groups with shape when possible.
- **Pre-attentive attention.** Color, size, position — use these
  deliberately to pull the reader's eye to the story *you* want them
  to see first.
- **Gestalt.** Proximity, similarity, common fate — let the plot's
  groupings do work for you. Zero lines, quadrant annotations, and
  well-placed text anchors all help.

The scatterplot in `assignment.py` is a starting point, not a ceiling.

### 3. Document your observations

Write a short analysis (2–4 paragraphs) in your notebook or in a
`REPORT.md` that answers:

- What clusters or patterns separate along each axis?
- What is the most **surprising** point or group, and what does it
  tell you about how the embedding model represents your domain?
- What would a **third axis** need to capture? I.e., what variation is
  your 2D projection hiding?

## Deliverable

You have freedom in format. At a minimum, your repo must contain:

- The **code** that produces your figure (marimo, Jupyter, or plain `.py`).
- A **reproducible pipeline** — a `run.sh` (or `Makefile` / Snakemake
  workflow) that regenerates the final figure from scratch. Someone
  should be able to clone your fork and run one command to reproduce
  your output. A starter `run.sh` is included; edit or replace it.
- The **raw data file** you used (CSV in `data/`).
- The **final figure** saved to disk (PNG or PDF).
- **Observations** either inline in the notebook or in `REPORT.md`.

Submit by pushing to your fork and posting the URL to Brightspace.

## Evaluation criteria

| Criterion | What we look for |
|---|---|
| **Atomic git history** | Small, focused commits with meaningful messages. One commit = one logical change. Not `final`, `final2`, `final-real`. |
| **Reproducible pipeline** | `bash run.sh` (or equivalent) regenerates data and figure from scratch on a fresh clone. No manual steps, no orphaned outputs. |
| **Comprehensive documentation** | Code is commented where non-obvious. The notebook / report explains *why* each axis was chosen and *what the figure shows*. A new reader can follow your reasoning. |
| **Visualization quality** | Follows the data-viz principles in Task 2. Symbols and lines are clearly visible. Groups are clearly separated. Colorblind-friendly. Pre-attentive cues and Gestalt grouping are used deliberately. |
| **Task completion** | Two plausible axes, one scatterplot, written observations — all present. |

## FAQ

**Can I work in a team?** Yes. Make a team repo and list members in
`REPORT.md`. Everyone submits the same URL.

**What embedding model should I use?** `all-MiniLM-L6-v2` (what the intro
notebook uses) is the default. You may swap in a larger model if you want,
but document the choice.

**My axis distribution is unimodal and boring.** That's information. Try
different pole words — use named entities instead of abstract concepts,
or pick poles that you think should actually separate the data. Iterate.
