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
        "forward-thinking innovation",
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
def _(alt, color_by, df_scored, mo, pd):
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

    zero_line_data = pd.DataFrame({"x": [0], "y": [0]})
    vertical_zero_line = (
        alt.Chart(zero_line_data)
        .mark_rule(color="#666666", strokeDash=[4, 4], strokeWidth=1)
        .encode(x="x:Q")
    )
    horizontal_zero_line = (
        alt.Chart(zero_line_data)
        .mark_rule(color="#666666", strokeDash=[4, 4], strokeWidth=1)
        .encode(y="y:Q")
    )

    points = (
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
                alt.Tooltip("x:Q", title="innovative score", format=".3f"),
                alt.Tooltip("y:Q", title="growth score", format=".3f"),
            ],
        )
    )

    chart = (
        (points + vertical_zero_line + horizontal_zero_line)
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
    return (SECTOR_COLORS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Focused sector view

    This chart shows only **Information Technology**, **Financials**, and
    **Health Care** so their positions are easier to compare directly.
    """)
    return


@app.cell
def _(SECTOR_COLORS, alt, df_scored, pd):
    focus_sectors = ["Information Technology", "Financials", "Health Care"]
    df_focus = df_scored[df_scored["sector"].isin(focus_sectors)].copy()
    focus_zero_line_data = pd.DataFrame({"x": [0], "y": [0]})

    focus_color = alt.Color(
        "sector:N",
        scale=alt.Scale(
            domain=focus_sectors,
            range=[SECTOR_COLORS[sector] for sector in focus_sectors],
        ),
        legend=alt.Legend(title="Sector"),
    )

    focus_vertical_zero_line = (
        alt.Chart(focus_zero_line_data)
        .mark_rule(color="#666666", strokeDash=[4, 4], strokeWidth=1)
        .encode(x="x:Q")
    )
    focus_horizontal_zero_line = (
        alt.Chart(focus_zero_line_data)
        .mark_rule(color="#666666", strokeDash=[4, 4], strokeWidth=1)
        .encode(y="y:Q")
    )

    focus_points = (
        alt.Chart(df_focus)
        .mark_circle(size=110, opacity=0.85, stroke="white", strokeWidth=0.7)
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
            color=focus_color,
            tooltip=[
                alt.Tooltip("name:N", title="Company"),
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("x:Q", title="innovative score", format=".3f"),
                alt.Tooltip("y:Q", title="growth score", format=".3f"),
            ],
        )
    )

    focus_chart = (
        (focus_points + focus_vertical_zero_line + focus_horizontal_zero_line)
        .properties(
            width=720,
            height=500,
            title="Focused comparison: Information Technology, Financials, and Health Care",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_legend(labelFontSize=11, titleFontSize=12)
        .interactive()
    )

    focus_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Step 3 — Observations

    1. Overall significant bias towards the positive end of the Y-axis and most of the companies are above 0.0 on the y-axis. That is, vast majority of the companies are growing, regarless of which sector.
    2. For innovation, overall biased towards the positive end but only slighly and much less significant than that of the growth bias. There are still a lot (~40%) sitting on the negative end of x-axis.
    3. In the Information Technology sector, there are slightly more companies on the negative innovative side. This is a surprise. I would expect the opposite.
    4. Both the financials and the health care sectors are opposite to IT: bias towards positive innovation.
    5. IT and financials have some outliers on the positive innovative end. More so than other sectors.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
