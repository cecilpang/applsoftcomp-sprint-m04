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
#     "vl-convert-python>=1.7.0",
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
    from pathlib import Path
    from sentence_transformers import SentenceTransformer
    from drawdata import ScatterWidget

    return Path, SentenceTransformer, alt, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # S&P 500 Semantic Axis Map

    This notebook places S&P 500 companies into a two-dimensional semantic space.
    Each axis is built from two sets of descriptive phrases: one negative pole
    and one positive pole. Company names are embedded with a SentenceTransformer
    model, then projected onto the selected axis vectors.

    The final charts use sector as the categorical attribute. Sector is encoded
    with both color and shape, and the zero lines divide the map into
    interpretable quadrants such as **high risk / slow growth**.
    """)
    return


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


@app.function
def save_chart_png(chart, output_path, scale_factor=2):
    """Save an Altair chart as PNG without stopping notebook execution."""
    try:
        output_path.parent.mkdir(exist_ok=True)
        chart.save(str(output_path), scale_factor=scale_factor)
        print(f"Saved figure: {output_path}")
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"Could not save {output_path}: {exc}")


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
    df.sample(5)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1 — Define Candidate Axes

    The list below defines ten candidate business axes. Each candidate axis has:

    - a short `name`,
    - a `negative_label` and `positive_label` for chart labels,
    - positive-pole phrases in `pos`,
    - negative-pole phrases in `neg`.

    For each axis, the notebook embeds the positive and negative phrase sets,
    averages each pole, subtracts the negative pole from the positive pole, and
    normalizes the result into a unit-length semantic direction.
    """)
    return


@app.cell(hide_code=True)
def _(make_axis, model, np):
    axes = [
        {
            "name": "Innovation",
            "negative_label": "not innovative",
            "positive_label": "innovative",
            "pos": [
                "forward-thinking innovation",
                "high R&D spend",
                "high patent activity",
                "high % revenue from new products",
                "high AI/tech adoption",
            ],
            "neg": [
                "operationally focused with limited innovation",
                "incremental rather than breakthrough-driven",
                "mature and slow to adopt new technologies",
                "primarily execution-driven rather than innovation-led",
                "conservative in product and technology evolution",
            ],
        },
        {
            "name": "Growth",
            "negative_label": "slow growth",
            "positive_label": "high growth",
            "pos": [
                "rapidly scaling with strong revenue momentum",
                "accelerated growth and market expansion",
                "high-velocity growth driven by demand",
                "expanding market share quickly",
                "growth-driven with significant upside potential",
            ],
            "neg": [
                "modest, steady growth",
                "measured, incremental expansion",
                "low-growth but stable and predictable",
                "operating in a mature growth phase",
                "limited growth with consistent performance",
            ],
        },
        {
            "name": "Profitability",
            "negative_label": "low profitability",
            "positive_label": "high profitability",
            "pos": [
                "high margin business model",
                "strong operating leverage",
                "consistently high ROIC",
                "efficient cost structure",
                "strong earnings generation",
            ],
            "neg": [
                "low margin operations",
                "cost-heavy structure",
                "limited operating leverage",
                "weak returns on capital",
                "profitability under pressure",
            ],
        },
        {
            "name": "Risk",
            "negative_label": "high risk",
            "positive_label": "low risk / stable",
            "pos": [
                "strong balance sheet",
                "low leverage and high liquidity",
                "stable cash flows",
                "resilient to market volatility",
                "investment-grade financial profile",
            ],
            "neg": [
                "highly leveraged balance sheet",
                "volatile earnings profile",
                "cash flow uncertainty",
                "sensitive to economic cycles",
                "elevated financial risk",
            ],
        },
        {
            "name": "Operational Efficiency",
            "negative_label": "inefficient",
            "positive_label": "highly efficient",
            "pos": [
                "high asset utilization",
                "strong operational discipline",
                "efficient capital allocation",
                "optimized supply chain",
                "high productivity per employee",
            ],
            "neg": [
                "operational inefficiencies",
                "low asset turnover",
                "suboptimal cost structure",
                "process fragmentation",
                "underutilized resources",
            ],
        },
        {
            "name": "Business Model",
            "negative_label": "asset-heavy / traditional",
            "positive_label": "asset-light / platform",
            "pos": [
                "asset-light and scalable",
                "platform-driven business model",
                "high operating leverage",
                "network effects driven",
                "subscription or recurring revenue model",
            ],
            "neg": [
                "asset-heavy operations",
                "capital-intensive model",
                "linear scaling with costs",
                "transaction-based revenue",
                "limited scalability",
            ],
        },
        {
            "name": "Market Position",
            "negative_label": "weak position",
            "positive_label": "strong position",
            "pos": [
                "market leader with strong share",
                "durable competitive advantage",
                "strong brand and customer loyalty",
                "pricing power",
                "dominant position in core markets",
            ],
            "neg": [
                "limited market share",
                "intense competitive pressure",
                "weak differentiation",
                "price taker in the market",
                "fragile competitive position",
            ],
        },
        {
            "name": "Customer Orientation",
            "negative_label": "product-centric",
            "positive_label": "customer-centric",
            "pos": [
                "deep customer focus",
                "strong customer experience",
                "data-driven customer insights",
                "high retention and loyalty",
                "customer-first product design",
            ],
            "neg": [
                "product-centric approach",
                "limited customer engagement",
                "low visibility into customer needs",
                "high churn or weak retention",
                "reactive to customer feedback",
            ],
        },
        {
            "name": "Technology Maturity",
            "negative_label": "legacy systems",
            "positive_label": "modern / AI-native",
            "pos": [
                "modern, cloud-native architecture",
                "AI-first or data-driven systems",
                "high automation and tooling maturity",
                "integrated data platform",
                "scalable and flexible tech stack",
            ],
            "neg": [
                "legacy technology stack",
                "fragmented systems landscape",
                "manual and process-heavy workflows",
                "limited data integration",
                "slow technology evolution",
            ],
        },
        {
            "name": "Governance",
            "negative_label": "loosely governed",
            "positive_label": "well-governed",
            "pos": [
                "strong governance frameworks",
                "clear policies and controls",
                "high transparency and auditability",
                "well-defined decision rights",
                "robust risk management practices",
            ],
            "neg": [
                "weak governance structures",
                "unclear policies and controls",
                "limited transparency",
                "ad hoc decision-making",
                "inconsistent risk management",
            ],
        },
    ]

    candidate_axis_vectors = [
        make_axis(_axis["pos"], _axis["neg"], model)
        for _axis in axes
    ]
    _axis_distance_rows = []

    for _axis_i in range(len(axes)):
        for _axis_j in range(_axis_i + 1, len(axes)):
            _cosine_similarity = float(
                np.dot(candidate_axis_vectors[_axis_i], candidate_axis_vectors[_axis_j])
            )
            _cosine_distance = 1 - _cosine_similarity
            _axis_distance_rows.append(
                {
                    "axis_i": _axis_i,
                    "axis_j": _axis_j,
                    "axis_i_name": axes[_axis_i]["name"],
                    "axis_j_name": axes[_axis_j]["name"],
                    "cosine_distance": _cosine_distance,
                }
            )

    top_axis_distances = sorted(
        _axis_distance_rows,
        key=lambda _row: _row["cosine_distance"],
        reverse=True,
    )[:10]

    print("Top 10 cosine distances between candidate axes (0-based indices):")
    for _row in top_axis_distances:
        print(
            f"axes {_row['axis_i']} and {_row['axis_j']}: "
            f"{_row['cosine_distance']:.3f} "
            f"({_row['axis_i_name']} vs. {_row['axis_j_name']})"
        )
    return axes, candidate_axis_vectors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compare Candidate Axes

    The output above ranks candidate-axis pairs by cosine distance:

    \[
    \text{cosine distance} = 1 - (\text{axis}_i \cdot \text{axis}_j)
    \]

    Since every axis vector is normalized, larger cosine distances indicate more
    different semantic directions. This helps identify pairs of axes that are
    less redundant before choosing the final two-dimensional view.
    """)
    return


@app.cell
def _(axes, candidate_axis_vectors):
    # Change this tuple to choose the two candidate axes used everywhere below.
    # The first index is the x-axis, and the second index is the y-axis.
    selected_axis_indices = (1, 3)

    selected_axes = {
        "x": axes[selected_axis_indices[0]],
        "y": axes[selected_axis_indices[1]],
    }
    selected_axis_vectors = {
        "x": candidate_axis_vectors[selected_axis_indices[0]],
        "y": candidate_axis_vectors[selected_axis_indices[1]],
    }

    print(
        "Selected axes: "
        f"x={selected_axis_indices[0]} ({selected_axes['x']['name']}), "
        f"y={selected_axis_indices[1]} ({selected_axes['y']['name']})"
    )
    return selected_axes, selected_axis_vectors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Select and Score the Final Axes

    `selected_axis_indices` is the single control point for the final map. The
    first value chooses the x-axis and the second value chooses the y-axis. In
    the current notebook:

    - x-axis: candidate axis `1`, **Growth**,
    - y-axis: candidate axis `3`, **Risk**.

    Changing `selected_axis_indices` automatically updates company scores, axis
    labels, tooltip labels, and quadrant labels.

    The next cell scores each company by embedding its name and taking the dot
    product with the selected x and y axis vectors.
    """)
    return


@app.cell
def _(df, model, selected_axis_vectors):
    x = score_words(df["name"].tolist(), selected_axis_vectors["x"], model)
    y = score_words(df["name"].tolist(), selected_axis_vectors["y"], model)
    df_scored = df.assign(x=x, y=y)
    df_scored.head()
    return (df_scored,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2 — Visualize

    The visualizations encode the selected semantic scores and company sector:

    - **Position** shows the selected semantic-axis scores.
    - **Color** shows sector.
    - **Shape** also shows sector, making the category encoding redundant.
    - **Dashed zero lines** mark the boundary between negative and positive
      sides of each selected axis.
    - **Quadrant labels** combine the x/y pole labels, such as
      **high risk / slow growth** and **low risk / high growth**.

    The first chart shows all sectors. The focused chart below filters to a
    smaller sector subset for easier comparison.
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
def _(Path, alt, color_by, df_scored, pd, selected_axes):
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
    SECTOR_SHAPES = {
        "Communication Services": "circle",
        "Consumer Discretionary": "square",
        "Consumer Staples": "diamond",
        "Energy": "triangle-up",
        "Financials": "triangle-down",
        "Health Care": "cross",
        "Industrials": "triangle-right",
        "Information Technology": "triangle-left",
        "Materials": "stroke",
        "Real Estate": "arrow",
        "Utilities": "wedge",
    }

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
        _shape = alt.Shape(
            "sector:N",
            scale=alt.Scale(
                domain=list(SECTOR_SHAPES.keys()),
                range=list(SECTOR_SHAPES.values()),
            ),
            legend=alt.Legend(title="Sector"),
        )

    _x_axis = selected_axes["x"]
    _y_axis = selected_axes["y"]
    _x_axis_title = (
        f"← {_x_axis['negative_label']}          {_x_axis['positive_label']} →"
    )
    _y_axis_title = (
        f"↓ {_y_axis['negative_label']}          {_y_axis['positive_label']} ↑"
    )
    _x_negative_label = _x_axis["negative_label"].split("/")[0].strip()
    _x_positive_label = _x_axis["positive_label"].split("/")[0].strip()
    _y_negative_label = _y_axis["negative_label"].split("/")[0].strip()
    _y_positive_label = _y_axis["positive_label"].split("/")[0].strip()
    _x_extent = max(
        abs(float(df_scored["x"].min())),
        abs(float(df_scored["x"].max())),
        0.01,
    ) * 1.12
    _y_extent = max(
        abs(float(df_scored["y"].min())),
        abs(float(df_scored["y"].max())),
        0.01,
    ) * 1.12
    _x_scale = alt.Scale(domain=[-_x_extent, _x_extent], zero=False)
    _y_scale = alt.Scale(domain=[-_y_extent, _y_extent], zero=False)

    zero_line_data = pd.DataFrame({"x": [0], "y": [0]})
    _quadrant_label_data = pd.DataFrame(
        [
            {
                "x": -_x_extent * 0.52,
                "y": -_y_extent * 0.78,
                "label": f"{_y_negative_label} / {_x_negative_label}",
            },
            {
                "x": _x_extent * 0.52,
                "y": -_y_extent * 0.78,
                "label": f"{_y_negative_label} / {_x_positive_label}",
            },
            {
                "x": -_x_extent * 0.52,
                "y": _y_extent * 0.78,
                "label": f"{_y_positive_label} / {_x_negative_label}",
            },
            {
                "x": _x_extent * 0.52,
                "y": _y_extent * 0.78,
                "label": f"{_y_positive_label} / {_x_positive_label}",
            },
        ]
    )
    vertical_zero_line = (
        alt.Chart(zero_line_data)
        .mark_rule(color="#666666", strokeDash=[4, 4], strokeWidth=1)
        .encode(
            x=alt.X(
                "x:Q",
                title=_x_axis_title,
                scale=_x_scale,
                axis=alt.Axis(grid=False, title=_x_axis_title, titlePadding=12),
            )
        )
    )
    horizontal_zero_line = (
        alt.Chart(zero_line_data)
        .mark_rule(color="#666666", strokeDash=[4, 4], strokeWidth=1)
        .encode(
            y=alt.Y(
                "y:Q",
                title=_y_axis_title,
                scale=_y_scale,
                axis=alt.Axis(grid=False, title=_y_axis_title, titlePadding=12),
            )
        )
    )

    points = (
        alt.Chart(df_scored)
        .mark_point(
            size=100,
            opacity=0.85,
            filled=True,
            stroke="white",
            strokeWidth=0.7,
        )
        .encode(
            x=alt.X(
                "x:Q",
                title=_x_axis_title,
                scale=_x_scale,
                axis=alt.Axis(grid=False, title=_x_axis_title, titlePadding=12),
            ),
            y=alt.Y(
                "y:Q",
                title=_y_axis_title,
                scale=_y_scale,
                axis=alt.Axis(grid=False, title=_y_axis_title, titlePadding=12),
            ),
            color=_color,
            shape=_shape,
            tooltip=[
                alt.Tooltip("name:N", title="Company"),
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("x:Q", title=f"{_x_axis['name']} score", format=".3f"),
                alt.Tooltip("y:Q", title=f"{_y_axis['name']} score", format=".3f"),
            ],
        )
    )
    _quadrant_labels = (
        alt.Chart(_quadrant_label_data)
        .mark_text(
            align="center",
            baseline="middle",
            color="white",
            fontSize=13,
            fontWeight="bold",
            stroke="white",
            strokeWidth=4,
        )
        .encode(
            x=alt.X(
                "x:Q",
                title=_x_axis_title,
                scale=_x_scale,
                axis=alt.Axis(grid=False, title=_x_axis_title, titlePadding=12),
            ),
            y=alt.Y(
                "y:Q",
                title=_y_axis_title,
                scale=_y_scale,
                axis=alt.Axis(grid=False, title=_y_axis_title, titlePadding=12),
            ),
            text="label:N",
        )
        + alt.Chart(_quadrant_label_data)
        .mark_text(
            align="center",
            baseline="middle",
            color="#111111",
            fontSize=13,
            fontWeight="bold",
        )
        .encode(
            x=alt.X(
                "x:Q",
                title=_x_axis_title,
                scale=_x_scale,
                axis=alt.Axis(grid=False, title=_x_axis_title, titlePadding=12),
            ),
            y=alt.Y(
                "y:Q",
                title=_y_axis_title,
                scale=_y_scale,
                axis=alt.Axis(grid=False, title=_y_axis_title, titlePadding=12),
            ),
            text="label:N",
        )
    )

    chart = (
        (points + vertical_zero_line + horizontal_zero_line + _quadrant_labels)
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

    save_chart_png(chart, Path("figs") / "sp500_all_sectors.png")

    chart
    return SECTOR_COLORS, SECTOR_SHAPES


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Focused sector view

    This chart shows **Information Technology**, **Financials**, and
    **Health Care** so their positions are easier to compare directly.
    """)
    return


@app.cell
def _(Path, SECTOR_COLORS, SECTOR_SHAPES, alt, df_scored, pd, selected_axes):
    focus_sectors = [
        "Information Technology",
        "Financials",
        "Health Care",
    ]
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
    focus_shape = alt.Shape(
        "sector:N",
        scale=alt.Scale(
            domain=focus_sectors,
            range=[SECTOR_SHAPES[sector] for sector in focus_sectors],
        ),
        legend=alt.Legend(title="Sector"),
    )

    _x_axis = selected_axes["x"]
    _y_axis = selected_axes["y"]
    _x_axis_title = (
        f"← {_x_axis['negative_label']}          {_x_axis['positive_label']} →"
    )
    _y_axis_title = (
        f"↓ {_y_axis['negative_label']}          {_y_axis['positive_label']} ↑"
    )
    _x_negative_label = _x_axis["negative_label"].split("/")[0].strip()
    _x_positive_label = _x_axis["positive_label"].split("/")[0].strip()
    _y_negative_label = _y_axis["negative_label"].split("/")[0].strip()
    _y_positive_label = _y_axis["positive_label"].split("/")[0].strip()
    _focus_x_extent = max(
        abs(float(df_focus["x"].min())),
        abs(float(df_focus["x"].max())),
        0.01,
    ) * 1.12
    _focus_y_extent = max(
        abs(float(df_focus["y"].min())),
        abs(float(df_focus["y"].max())),
        0.01,
    ) * 1.12
    _focus_x_scale = alt.Scale(domain=[-_focus_x_extent, _focus_x_extent], zero=False)
    _focus_y_scale = alt.Scale(domain=[-_focus_y_extent, _focus_y_extent], zero=False)
    _focus_quadrant_label_data = pd.DataFrame(
        [
            {
                "x": -_focus_x_extent * 0.52,
                "y": -_focus_y_extent * 0.78,
                "label": f"{_y_negative_label} / {_x_negative_label}",
            },
            {
                "x": _focus_x_extent * 0.52,
                "y": -_focus_y_extent * 0.78,
                "label": f"{_y_negative_label} / {_x_positive_label}",
            },
            {
                "x": -_focus_x_extent * 0.52,
                "y": _focus_y_extent * 0.78,
                "label": f"{_y_positive_label} / {_x_negative_label}",
            },
            {
                "x": _focus_x_extent * 0.52,
                "y": _focus_y_extent * 0.78,
                "label": f"{_y_positive_label} / {_x_positive_label}",
            },
        ]
    )

    focus_vertical_zero_line = (
        alt.Chart(focus_zero_line_data)
        .mark_rule(color="#666666", strokeDash=[4, 4], strokeWidth=1)
        .encode(
            x=alt.X(
                "x:Q",
                title=_x_axis_title,
                scale=_focus_x_scale,
                axis=alt.Axis(grid=False, title=_x_axis_title, titlePadding=12),
            )
        )
    )
    focus_horizontal_zero_line = (
        alt.Chart(focus_zero_line_data)
        .mark_rule(color="#666666", strokeDash=[4, 4], strokeWidth=1)
        .encode(
            y=alt.Y(
                "y:Q",
                title=_y_axis_title,
                scale=_focus_y_scale,
                axis=alt.Axis(grid=False, title=_y_axis_title, titlePadding=12),
            )
        )
    )

    focus_points = (
        alt.Chart(df_focus)
        .mark_point(
            size=120,
            opacity=0.9,
            filled=True,
            stroke="white",
            strokeWidth=0.8,
        )
        .encode(
            x=alt.X(
                "x:Q",
                title=_x_axis_title,
                scale=_focus_x_scale,
                axis=alt.Axis(grid=False, title=_x_axis_title, titlePadding=12),
            ),
            y=alt.Y(
                "y:Q",
                title=_y_axis_title,
                scale=_focus_y_scale,
                axis=alt.Axis(grid=False, title=_y_axis_title, titlePadding=12),
            ),
            color=focus_color,
            shape=focus_shape,
            tooltip=[
                alt.Tooltip("name:N", title="Company"),
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("x:Q", title=f"{_x_axis['name']} score", format=".3f"),
                alt.Tooltip("y:Q", title=f"{_y_axis['name']} score", format=".3f"),
            ],
        )
    )
    _focus_quadrant_labels = (
        alt.Chart(_focus_quadrant_label_data)
        .mark_text(
            align="center",
            baseline="middle",
            color="white",
            fontSize=13,
            fontWeight="bold",
            stroke="white",
            strokeWidth=4,
        )
        .encode(
            x=alt.X(
                "x:Q",
                title=_x_axis_title,
                scale=_focus_x_scale,
                axis=alt.Axis(grid=False, title=_x_axis_title, titlePadding=12),
            ),
            y=alt.Y(
                "y:Q",
                title=_y_axis_title,
                scale=_focus_y_scale,
                axis=alt.Axis(grid=False, title=_y_axis_title, titlePadding=12),
            ),
            text="label:N",
        )
        + alt.Chart(_focus_quadrant_label_data)
        .mark_text(
            align="center",
            baseline="middle",
            color="#111111",
            fontSize=13,
            fontWeight="bold",
        )
        .encode(
            x=alt.X(
                "x:Q",
                title=_x_axis_title,
                scale=_focus_x_scale,
                axis=alt.Axis(grid=False, title=_x_axis_title, titlePadding=12),
            ),
            y=alt.Y(
                "y:Q",
                title=_y_axis_title,
                scale=_focus_y_scale,
                axis=alt.Axis(grid=False, title=_y_axis_title, titlePadding=12),
            ),
            text="label:N",
        )
    )

    focus_chart = (
        (
            focus_points
            + focus_vertical_zero_line
            + focus_horizontal_zero_line
            + _focus_quadrant_labels
        )
        .properties(
            width=720,
            height=500,
            title="Focused comparison: selected S&P 500 sectors",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_legend(labelFontSize=11, titleFontSize=12)
        .interactive()
    )

    save_chart_png(focus_chart, Path("figs") / "sp500_focused_sectors.png")

    focus_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Step 3 — Observations

    From the overall chart:

    1. Almost all of the companies are in the low risk / high growth quadrant, with a small number in the low risk/slow growth quadrant and only 6 in the other two quadrants combined. This could mean SP 500 tends to select low risk/high growth companies to include in the index, or that low risk/high growth leads to the qualities that SP 500 looks for. This makes sense because low risk/high growth is the most desirable kind of companies.

    2. There is a correlation between risk and growth. Lower the risk, higher the growth. The regression line is tilting upwards at approximately 35 degrees. Note that this correlation is among the SP 500 companies. According to the embedding vectors, the concepts of risk and growth themselves are othorganal, with consine distance larger than 1.

    A second chart was created to focus on the three sectors I am most interested in: IT, financials, and health care. These are observations from this chart:

    1. Among the 3 sectors, the risk/growth regression line is steepest for Financials, while IT and health care are similar.

    2. Financials clustered in the lowest risk area, while IT companies have higher risk overall and health care in the middle.

    3. Health care has the narrowest range of risks, indicating that health care companies are similar to each other in risk tolerance. More so than companies in IT and Financials.
    """)
    return


if __name__ == "__main__":
    app.run()
