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

    return SentenceTransformer, alt, mo, np, pd


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
    return (df_scored,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2 — Visualize
    """)
    return


@app.cell
def _(mo):
    color_by = mo.ui.dropdown(
        options={
            "sector (categorical)": "sector",
        },
        value="sector (categorical)",
        label="Color by: ",
    )
    return (color_by,)


@app.cell
def _(alt, color_by, df_scored, mo):
    # Okabe–Ito palette — categorical, colorblind-safe.
    SECTOR_COLORS = {
        "Communication Services": "#009E73",
        "Consumer Discretionary": "#0072B2",
        "Consumer Staples": "#D55E00",
        "Energy": "#E69F00",
        "Financials": "#56B4E9",
        'Health Care': "#CC79A7",  # reddish purple
        'Industrials': "#F0E442",  # yellow
        'Information Technology': "#000000",  # black (high contrast)
        'Materials': "#999999",  # medium gray
        'Real Estate': "#8B4513",  # brown (earth tone, distinct from orange)
        'Utilities': "#6A3D9A"   # deep purple
    }

    # GaWC rating ordered from "most globally connected" to "least".
    BUSINESS_ORDER = [
        "Alpha++",
        "Alpha+",
        "Alpha",
        "Alpha-",
        "Beta+",
        "Beta",
        "Beta-",
        "Gamma+",
        "Gamma",
        "Gamma-",
        "High Sufficiency",
        "Sufficiency",
    ]

    if color_by.value == "sector":
        # Categorical → qualitative palette.
        _color = alt.Color(
            "sector:N",
            scale=alt.Scale(
                domain=list(SECTOR_COLORS.keys()),
                range=list(SECTOR_COLORS.values()),
            ),
            legend=alt.Legend(title="Sector"),
        )

    chart = (
        alt.Chart(df_scored)
        .mark_circle(size=90, opacity=0.8, stroke="white", strokeWidth=0.6)
        .encode(
            x=alt.X(
                "x:Q",
                title="← not innovative          innovative →",
                scale=alt.Scale(zero=False, padding=20),
                axis=alt.Axis(grid=False),
            ),
            y=alt.Y(
                "y:Q",
                title="← stable          growing →",
                scale=alt.Scale(zero=False, padding=20),
                axis=alt.Axis(grid=False),
            ),
            color=_color,
            tooltip=[
                alt.Tooltip("name:N", title="Company"),
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("x:Q", title="megacity score", format=".3f"),
                alt.Tooltip("y:Q", title="tropical-warm score", format=".3f"),
            ],
        )
        .properties(
            width=720,
            height=500,
            title="SP 500 companies in a 2D semantic space (hover for details)",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_legend(labelFontSize=11, titleFontSize=12)
        .interactive()  # pan + zoom
    )

    # Stack the dropdown directly above the chart so it is always visible.
    mo.vstack([color_by, chart])
    return


if __name__ == "__main__":
    app.run()
