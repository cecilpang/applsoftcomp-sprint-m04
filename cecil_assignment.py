# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.10.0",
#     "sentence-transformers>=2.7.0",
#     "numpy>=1.24",
#     "pandas>=2.0",
#     "matplotlib>=3.7",
#     "scipy>=1.11",
#     "ipython>=8.0",
#     "drawdata==0.5.0",
#     "anywidget>=0.9",
#     "seaborn==0.13.2",
#     "altair==6.0.0",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import altair as alt
    from sentence_transformers import SentenceTransformer
    from drawdata import ScatterWidget

    return SentenceTransformer, mo, np, pd


@app.cell
def _(SentenceTransformer):
    ## Instantiate SentenseTransformer model

    model = SentenceTransformer("all-mpnet-base-v2")  # all-MiniLM-L6-v2 if you want faster but noisier results
    model
    return (model,)


@app.cell
def _(np):
    def make_axis(positive_words, negative_words, embedding_model):
        """Return a unit-length semantic axis from two word sets."""

        # get the embeddings for each pole
        pos_emb = embedding_model.encode(positive_words, normalize_embeddings=True)
        neg_emb = embedding_model.encode(negative_words, normalize_embeddings=True)

        # Compute the pole centroids
        # axis = 0 means "average across the rows, keep the columns (dims) intact"
        # since pos_emb is shape (num_pos_words, embedding_dim), the mean is shape (embedding_dim,)
        pole_pos = pos_emb.mean(axis=0)  # (embedding_dim,)
        pole_neg = neg_emb.mean(axis=0)  # (embedding_dim,)

        # The axis is the difference between the two centroids, normalized to unit length.
        v = pole_pos - pole_neg

        v = v / (np.linalg.norm(v) + 1e-10)  # add small epsilon to prevent division by zero

        return v / (np.linalg.norm(v) + 1e-10)

    return (make_axis,)


@app.function
def score_words(words, axis, embedding_model):
    """Project each word onto the axis. Returns one score per word."""

    emb = embedding_model.encode(list(words), normalize_embeddings=True)

    # Projection to the axis is just a dot product (since the axis is unit-length).
    # @ is matrix multiplication in NumPy. Since `emb` is shape (num_words, embedding_dim) and `axis` is shape (embedding_dim,), the result is shape (num_words,), which is exactly what we want: one score per word.
    proj = emb @ axis

    return proj


@app.cell
def _(pd):
    ## Read in SP 500 companies

    df = pd.read_csv(
        "data/sp500.csv",
        dtype={
            "name": "string",
            "sector": "category",
        },
    )
    print(f"{len(df)} companies across {df['sector'].nunique()} sectors.")
    print(f"sectors: {list(df['sector'].unique())})")
    df.sample(20)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1 — Design two semantic axes



    Our two axes for companies:

    - **Horizontal** — *not innovative* (−) ↔ *innovative* (+)
    - **Vertical** — *low growth* (−) ↔ *high growth* (+)
    """)
    return


@app.cell
def _(make_axis, model):
    axis1_pos = [
        "innovative",
        "high R&D spend",
        "high patent activity",
        "high % revenue from new products",
        "high AI/tech adoption",
    ]

    axis1_neg = [
        "operationally focused with limited innovation",
        "incremental rather than breakthrough-driven",
        "mature and slow to adopt new technologies",
        "primarily execution-driven rather than innovation-led",
        "conservative in product and technology evolution",
    ]
    axis_innovative = make_axis(axis1_pos, axis1_neg, model)
    return (axis_innovative,)


@app.cell
def _(make_axis, model):
    axis2_pos = [
        "rapidly scaling with strong revenue momentum",
        "experiencing accelerated growth and market expansion",
        "high-velocity growth driven by strong demand",
        "scaling quickly with expanding market share",
        "growth-driven with significant upside potential",
    ]
    axis2_neg = [
        "experiencing modest, steady growth",
        "operating in a mature, slow-expansion phase",
        "limited growth with consistent performance",
        "growing at a measured, incremental pace",
        "low-growth but stable and predictable",
    ]
    axis_growth = make_axis(axis2_pos, axis2_neg, model)
    return (axis_growth,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Score every company along both axes
    For each company name we compute its embedding and take the dot product with each axis.
    """)
    return


@app.cell
def _(axis_growth, axis_innovative, df, model):
    x = score_words(df["name"].tolist(), axis_innovative, model)
    y = score_words(df["name"].tolist(), axis_growth, model)
    df_scored = df.assign(x=x, y=y)
    df_scored.head()
    return


if __name__ == "__main__":
    app.run()
