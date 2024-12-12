import sys
import click
from phageminiprot.classify import main as classify_main

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
    # Pass the click arguments to the classify main function
    classify_main(input_fasta=input_fasta,
                  model_dirpath=model_dirpath,
                  output_filepath=output_filepath,
                  esm_model=esm_model,
                  layer=layer,
                  batch_size=batch_size,
                  embed_script=embed_script,
                  embedding_output=embedding_output)

def main():
    """Entry point for the PhageMiniProt CLI."""
    cli()

if __name__ == '__main__':
    main()

