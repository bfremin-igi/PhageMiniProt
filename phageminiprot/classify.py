import sys
import os
import subprocess
import numpy as np
from Bio import SeqIO
import pathlib
import joblib # Assuming the model is a sklearn model

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

def main():
    # Ensure correct number of arguments (at least one argument should be passed)
    if len(sys.argv) < 2:
        print("Usage: PhageMiniProt classify <input_sequences.faa>")
        sys.exit(1)

    # Input FASTA file (first argument)
    input_file = sys.argv[1]

    # Set the default values for the optional arguments
    model_dirpath = DEFAULT_MODEL_DIRPATH
    output_filepath = DEFAULT_OUTPUT_FILEPATH
    esm_model = DEFAULT_ESM_MODEL
    layer = DEFAULT_LAYER
    batch_size = DEFAULT_BATCH_SIZE
    embed_script = DEFAULT_EMBED_SCRIPT
    embedding_output = DEFAULT_EMBEDDING_OUTPUT

    # Process any additional options passed via command line
    for i in range(2, len(sys.argv), 2):
        if sys.argv[i] == '--model-dirpath':
            model_dirpath = sys.argv[i+1]
        elif sys.argv[i] == '--output-filepath':
            output_filepath = sys.argv[i+1]
        elif sys.argv[i] == '--esm-model':
            esm_model = sys.argv[i+1]
        elif sys.argv[i] == '--layer':
            layer = int(sys.argv[i+1])  # Convert to integer
        elif sys.argv[i] == '--batch-size':
            batch_size = int(sys.argv[i+1])  # Convert to integer
        elif sys.argv[i] == '--embed-script':
            embed_script = sys.argv[i+1]
        elif sys.argv[i] == '--embedding-output':
            embedding_output = sys.argv[i+1]

    # Validate input file
    if not os.path.exists(input_file):
        print(f"Error: The input file {input_file} does not exist.")
        sys.exit(1)

    # Step 1: Embed the sequences
    classify_script_dir = os.path.dirname(os.path.realpath(__file__))
    embed_script_path = os.path.join(classify_script_dir, 'embed.py')
    print("Generating embeddings...")
    embed_command = [
        "python", embed_script_path,
        "--output-filepath", str(embedding_output),
        "--model", esm_model,
        "--layer", str(layer),
        input_file
    ]
    
    try:
        run_embed_command(embed_command)
    except subprocess.CalledProcessError as e:
        print(f"Error during embedding generation: {e}")
        return

    # Step 2: Load embeddings and trained model
    print("Loading embeddings and model...")
    embeddings = np.load(embedding_output)
    model = joblib.load(pathlib.Path(model_dirpath) / "model.joblib")  # Load the model (ensure it's joblib format)

    # Step 3: Make predictions
    print("Classifying sequences...")
    predictions = model.predict(embeddings)  # Directly use embeddings here

    # Step 4: Save predictions
    print(f"Saving predictions to {output_filepath}...")
    with open(output_filepath, "w") as f:
        f.write("Sequence_ID,Prediction\n")
        for record, pred in zip(SeqIO.parse(input_file, "fasta"), predictions):
            label = "Real" if pred >= 0.5 else "Not Real"
            f.write(f"{record.id},{label}\n")

    print("Classification completed.")

if __name__ == "__main__":
    main()
