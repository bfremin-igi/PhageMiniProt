import os
import sys
import subprocess
import numpy as np
from Bio import SeqIO
import pathlib
import joblib  # Assuming the model is a sklearn model

# Default values for optional arguments
DEFAULT_MODEL_DIRPATH = 'PhageMiniProt_model'
DEFAULT_OUTPUT_FILEPATH = './predictions.csv'
DEFAULT_ESM_MODEL = 'esm2_t6_8M_UR50D'
DEFAULT_LAYER = -1
DEFAULT_BATCH_SIZE = 4096
DEFAULT_EMBED_SCRIPT = './embed.py'
DEFAULT_EMBEDDING_OUTPUT = './temp_embeddings.npy'


def run_embed_command(embed_command):
    """Run the embedding script to generate embeddings."""
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    env["MKL_SERVICE_FORCE_INTEL"] = "1"
    subprocess.run(embed_command, check=True, env=env)


def main(input_fasta, model_dirpath, output_filepath, esm_model, layer, batch_size, embed_script, embedding_output):
    """
    Classify protein sequences based on their embeddings.

    Args:
        input_fasta (str): Path to the input FASTA file containing protein sequences
        model_dirpath (str): Path to the trained model directory
        output_filepath (str): Path to save the classification output
        esm_model (str): ESM model to use for embedding
        layer (int): Layer to extract embeddings from
        batch_size (int): Batch size for processing
        embed_script (str): Path to the embedding script
        embedding_output (str): Path to save the generated embeddings
    """
    # Validate input file
    if not os.path.exists(input_fasta):
        print(f"Error: The input file {input_fasta} does not exist.")
        return

    # Step 1: Embed the sequences
    classify_script_dir = os.path.dirname(os.path.realpath(__file__))
    embed_script_path = os.path.join(classify_script_dir, embed_script)
    print("Generating embeddings...")
    embed_command = [
        # Use the interpreter this process is running under (sys.executable),
        # not whatever "python" happens to resolve to on PATH. This ensures
        # the embedding subprocess runs in the same env/venv as the CLI itself.
        sys.executable, embed_script_path,
        "--output-filepath", str(embedding_output),
        "--model", esm_model,
        "--layer", str(layer),
        input_fasta
    ]

    try:
        run_embed_command(embed_command)
    except subprocess.CalledProcessError as e:
        print(f"Error during embedding generation: {e}")
        return

    # Step 2: Load embeddings and trained model
    print("Loading embeddings and model...")
    embeddings = np.load(embedding_output)
    model = joblib.load(pathlib.Path(model_dirpath) / "model.joblib")  # Load the model (ensure it's in joblib format)

    # Step 3: Make predictions (labels + probabilities)
    print("Classifying sequences...")
    predictions = model.predict(embeddings)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(embeddings)[:, 1]
    else:
        # Fallback for models without predict_proba: use predict() as a 0/1 proxy
        probabilities = predictions.astype(float)

    # Step 4: Save predictions
    print(f"Saving predictions to {output_filepath}...")
    with open(output_filepath, "w") as f:
        f.write("Sequence_ID,Prediction,Probability\n")
        for record, pred, prob in zip(SeqIO.parse(input_fasta, "fasta"), predictions, probabilities):
            label = "Real" if pred >= 0.5 else "Not Real"
            f.write(f"{record.id},{label},{prob:.6f}\n")

    print("Classification completed.")


if __name__ == "__main__":
    main()
