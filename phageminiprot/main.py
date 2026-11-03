"""PhageMiniProt command-line interface."""

import importlib.resources as resources

import click

from phageminiprot.classify import DEFAULT_THRESHOLD, main as classify_main
from phageminiprot.embed import DEFAULT_TOKS_PER_BATCH
from phageminiprot.train import main as train_main

DEFAULT_PHAGE_MODEL_DIRPATH = str(
    resources.files("phageminiprot") / "models" / "PhageMiniProt_model"
)
DEFAULT_META_MODEL_DIRPATH = str(
    resources.files("phageminiprot") / "models" / "MetaMiniProt_model"
)


@click.group()
@click.version_option(package_name="PhageMiniProt")
def cli():
    """Classify small proteins with ESM-2 embeddings and XGBoost (MiniProt Suite)."""


@cli.command("classify")
@click.argument("input_fasta", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output-filepath",
    default="./predictions.csv",
    show_default=True,
    help="CSV path for predictions.",
)
@click.option(
    "--meta",
    "use_meta",
    is_flag=True,
    help="Use bundled MetaMiniProt (HMP-II) instead of PhageMiniProt.",
)
@click.option(
    "--model-dirpath",
    default=None,
    help="Custom model directory containing model.joblib (overrides --meta).",
)
@click.option(
    "--threshold",
    default=DEFAULT_THRESHOLD,
    show_default=True,
    type=float,
    help="Probability cutoff for model-positive (P >= threshold).",
)
@click.option(
    "--esm-model",
    default="esm2_t6_8M_UR50D",
    show_default=True,
    help="ESM-2 model used for embeddings (must match training).",
)
@click.option(
    "--layer",
    default=-1,
    type=int,
    show_default=True,
    help="ESM layer for embeddings (-1 = final).",
)
@click.option(
    "--batch-size",
    default=DEFAULT_TOKS_PER_BATCH,
    type=int,
    show_default=True,
    help="Approx. tokens per embedding batch (passed to ESM batching).",
)
@click.option(
    "--embedding-output",
    default=None,
    help="Optional path to keep the .npy embeddings; default is a temp file.",
)
@click.option(
    "--keep-embeddings",
    is_flag=True,
    help="Keep the temporary embedding file when --embedding-output is not set.",
)
def classify(
    input_fasta,
    output_filepath,
    use_meta,
    model_dirpath,
    threshold,
    esm_model,
    layer,
    batch_size,
    embedding_output,
    keep_embeddings,
):
    """Classify protein sequences from a FASTA file.

    INPUT_FASTA should be amino-acid FASTA (ideally ≤50 aa). ORF calling is
    upstream of this tool.
    """
    if model_dirpath is None:
        model_dirpath = (
            DEFAULT_META_MODEL_DIRPATH if use_meta else DEFAULT_PHAGE_MODEL_DIRPATH
        )

    classify_main(
        input_fasta=input_fasta,
        model_dirpath=model_dirpath,
        output_filepath=output_filepath,
        esm_model=esm_model,
        layer=layer,
        batch_size=batch_size,
        threshold=threshold,
        embedding_output=embedding_output,
        keep_embeddings=keep_embeddings,
    )


@cli.command("train")
@click.option(
    "--positive-class-filepath",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Positive-class embeddings (.npy).",
)
@click.option(
    "--negative-class-filepath",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Negative-class embeddings (.npy).",
)
@click.option(
    "--model-dirpath",
    required=True,
    help="Directory to write model.joblib.",
)
@click.option("--n-estimators", default=200, type=int, show_default=True)
@click.option("--learning-rate", default=0.05, type=float, show_default=True)
@click.option("--max-depth", default=8, type=int, show_default=True)
@click.option("--n-splits", default=10, type=int, show_default=True)
@click.option("--holdout-fraction", default=0.15, type=float, show_default=True)
def train(
    positive_class_filepath,
    negative_class_filepath,
    model_dirpath,
    n_estimators,
    learning_rate,
    max_depth,
    n_splits,
    holdout_fraction,
):
    """Train an XGBoost classifier (random stratified holdout; not paper split)."""
    train_main(
        positive_class_filepath=positive_class_filepath,
        negative_class_filepath=negative_class_filepath,
        model_dirpath=model_dirpath,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_splits=n_splits,
        holdout_fraction=holdout_fraction,
    )


def main():
    cli()


if __name__ == "__main__":
    main()
