import os
import pathlib
import click
import numpy as np  # Import first to avoid threading conflicts
import subprocess
from joblib import load
from Bio import SeqIO
import xgboost as xgb

def run_embed_command(embed_command):
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    env["MKL_SERVICE_FORCE_INTEL"] = "1"
    subprocess.run(embed_command, check=True, env=env)

@click.command()
@click.argument("fasta_file", type=click.Path(exists=True))
@click.option("--model-dirpath", type=click.Path(exists=True), required=True, help="Path to the trained model directory.")
@click.option("--output-filepath", type=click.Path(), default="./predictions.csv", help="Output CSV file for predictions.")
@click.option(
    "--embed-script", type=click.Path(exists=True), default="./embed.py", help="Path to the embedding script."
)
@click.option(
    "--embedding-output", type=click.Path(), default="./temp_embeddings.npy", help="Temporary file for embeddings."
)
@click.option("--esm-model", type=str, default="esm2_t6_8M_UR50D", help="ESM model to use for embeddings.")
@click.option("--layer", type=int, default=-1, help="Layer index for embeddings (-1 for final layer).")
@click.option("--batch-size", type=int, default=4096, help="Batch size for embedding computation.")
def classify(
    fasta_file, model_dirpath, output_filepath, embed_script, embedding_output, esm_model, layer, batch_size
):
    """
    Classifies sequences in a FASTA file as 'real' or 'not real' using the trained model.
    """
    # Step 1: Embed the sequences
    print("Generating embeddings...")
    embed_command = [
        "python", embed_script,
        "--output-filepath", str(embedding_output),
        "--model", esm_model,
        "--layer", str(layer),
        fasta_file
    ]
    try:
        run_embed_command(embed_command)
    except subprocess.CalledProcessError as e:
        print(f"Error during embedding generation: {e}")
        return

    # Step 2: Load embeddings and trained model
    print("Loading embeddings and model...")
    embeddings = np.load(embedding_output)
    model = load(pathlib.Path(model_dirpath) / "model.joblib")  # Changed to load model.joblib

    # Step 3: Make predictions
    print("Classifying sequences...")
    predictions = model.predict(embeddings)  # Directly use embeddings here

    # Step 4: Save predictions
    print(f"Saving predictions to {output_filepath}...")
    with open(output_filepath, "w") as f:
        f.write("Sequence_ID,Prediction\n")
        for record, pred in zip(SeqIO.parse(fasta_file, "fasta"), predictions):
            label = "Real" if pred >= 0.5 else "Not Real"
            f.write(f"{record.id},{label}\n")

    print("Classification completed.")

if __name__ == "__main__":
    classify()

