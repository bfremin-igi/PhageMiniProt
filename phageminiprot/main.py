import sys
import click
import importlib.resources as resources

from phageminiprot.classify import main as classify_main
from phageminiprot.train import main as train_main

# Resolve default model dirs relative to the installed package so `pip install .`
# users don't need to run from inside the repo checkout.
DEFAULT_PHAGE_MODEL_DIRPATH = str(resources.files("phageminiprot") / "models" / "PhageMiniProt_model")
DEFAULT_META_MODEL_DIRPATH = str(resources.files("phageminiprot") / "models" / "MetaMiniProt_model")


@click.group()
def cli():
    """PhageMiniProt: A tool for classifying phage proteins using MiniProt embeddings."""
    pass


@cli.command()
@click.argument('input_fasta', type=click.Path(exists=True))
@click.option('--model-dirpath',
              default=DEFAULT_PHAGE_MODEL_DIRPATH,
              show_default=True,
              help='Directory path to the trained model')
@click.option('--output-filepath',
              default='./predictions.csv',
              help='Path to save the output predictions')
@click.option('--esm-model',
              default='esm2_t6_8M_UR50D',
              help='ESM model to use for embedding')
@click.option('--layer',
              default=-1,
              type=int,
              help='Layer to extract embeddings from')
@click.option('--batch-size',
              default=4096,
              type=int,
              help='Batch size for processing')
@click.option('--embed-script',
              default='./embed.py',
              help='Path to the embedding script')
@click.option('--embedding-output',
              default='./temp_embeddings.npy',
              help='Path to save the generated embeddings')
def classify(input_fasta, model_dirpath, output_filepath, esm_model, layer, batch_size, embed_script, embedding_output):
    """
    Classify protein sequences from a FASTA file.

    INPUT_FASTA: Path to the input FASTA file containing protein sequences
    """
    classify_main(input_fasta=input_fasta,
                  model_dirpath=model_dirpath,
                  output_filepath=output_filepath,
                  esm_model=esm_model,
                  layer=layer,
                  batch_size=batch_size,
                  embed_script=embed_script,
                  embedding_output=embedding_output)


@cli.command()
@click.option('--positive-class-filepath', required=True, type=click.Path(exists=True),
              help='Path to positive-class embeddings (.npy).')
@click.option('--negative-class-filepath', required=True, type=click.Path(exists=True),
              help='Path to negative-class embeddings (.npy).')
@click.option('--model-dirpath', required=True,
              help='Directory to save the trained model.')
@click.option('--n-estimators', default=200, type=int, show_default=True,
              help='Number of trees in XGBoost.')
@click.option('--learning-rate', default=0.05, type=float, show_default=True,
              help='Learning rate for XGBoost.')
@click.option('--max-depth', default=8, type=int, show_default=True,
              help='Maximum depth of XGBoost trees.')
@click.option('--n-splits', default=10, type=int, show_default=True,
              help='Number of folds for cross-validation.')
@click.option('--holdout-fraction', default=0.15, type=float, show_default=True,
              help='Fraction of data reserved as an independent holdout set, evaluated after CV.')
def train(positive_class_filepath, negative_class_filepath, model_dirpath,
          n_estimators, learning_rate, max_depth, n_splits, holdout_fraction):
    """
    Train an XGBoost classifier on positive/negative embeddings, with
    cross-validation and an independent holdout evaluation.
    """
    train_main(positive_class_filepath=positive_class_filepath,
               negative_class_filepath=negative_class_filepath,
               model_dirpath=model_dirpath,
               n_estimators=n_estimators,
               learning_rate=learning_rate,
               max_depth=max_depth,
               n_splits=n_splits,
               holdout_fraction=holdout_fraction)


def main():
    """Entry point for the PhageMiniProt CLI."""
    cli()


if __name__ == '__main__':
    main()
