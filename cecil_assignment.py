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


@app.cell
def _(SentenceTransformer):
    ## Instantiate SentenseTransformer model

    model = SentenceTransformer("all-mpnet-base-v2")  # all-MiniLM-L6-v2 if you want faster but noisier results
    model
    return


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

    return


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


if __name__ == "__main__":
    app.run()

