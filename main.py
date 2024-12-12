import os
import click
import sys

@click.group()
def cli():
    """PhageMiniProt: A tool for classifying phage proteins using MiniProt embeddings."""
    pass

@cli.command()
@click.argument('input_fasta', type=click.Path(exists=True))
@click.option('--model-dirpath', 
              default='PhageMiniProt_model', 
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
def classify(input_fasta, model_dirpath, output_filepath, esm_model, layer, batch_size):
    """
    Classify protein sequences from a FASTA file.
    
    INPUT_FASTA: Path to the input FASTA file containing protein sequences
    """
    from .classify import main as classify_main
    
    # Temporarily modify sys.argv to pass arguments to the classify function
    original_argv = sys.argv
    sys.argv = [
        original_argv[0], 
        input_fasta, 
        f'--model-dirpath={model_dirpath}',
        f'--output-filepath={output_filepath}',
        f'--esm-model={esm_model}',
        f'--layer={layer}',
        f'--batch-size={batch_size}'
    ]
    
    try:
        classify_main()
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

@cli.command()
@click.argument('input_fasta', type=click.Path(exists=True))
@click.option('--model-dirpath', 
              default='PhageMiniProt_model', 
              help='Directory to save the trained model')
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
def train(input_fasta, model_dirpath, esm_model, layer, batch_size):
    """
    Train a model on protein sequences from a FASTA file.
    
    INPUT_FASTA: Path to the input FASTA file containing protein sequences
    """
    from .train import main as train_main
    
    # Temporarily modify sys.argv to pass arguments to the train function
    original_argv = sys.argv
    sys.argv = [
        original_argv[0], 
        input_fasta, 
        f'--model-dirpath={model_dirpath}',
        f'--esm-model={esm_model}',
        f'--layer={layer}',
        f'--batch-size={batch_size}'
    ]
    
    try:
        train_main()
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

@cli.command()
@click.argument('input_fasta', type=click.Path(exists=True))
@click.option('--output-filepath', 
              default='./protein_embeddings', 
              help='Path to save the embeddings')
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
def embed(input_fasta, output_filepath, esm_model, layer, batch_size):
    """
    Generate embeddings for protein sequences from a FASTA file.
    
    INPUT_FASTA: Path to the input FASTA file containing protein sequences
    """
    from .embed import main as embed_main
    
    # Temporarily modify sys.argv to pass arguments to the embed function
    original_argv = sys.argv
    sys.argv = [
        original_argv[0], 
        input_fasta, 
        f'--output-filepath={output_filepath}',
        f'--esm-model={esm_model}',
        f'--layer={layer}',
        f'--batch-size={batch_size}'
    ]
    
    try:
        embed_main()
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def main():
    """Entry point for the PhageMiniProt CLI."""
    cli()

if __name__ == '__main__':
    main()
