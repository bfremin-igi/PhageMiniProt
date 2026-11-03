"""Classify protein FASTA sequences with a bundled (or custom) XGBoost model."""

import os
import pathlib
import sys
import tempfile

import joblib
import numpy as np
from Bio import SeqIO

from phageminiprot.embed import DEFAULT_TOKS_PER_BATCH, embed_fasta

DEFAULT_OUTPUT_FILEPATH = "./predictions.csv"
DEFAULT_ESM_MODEL = "esm2_t6_8M_UR50D"
DEFAULT_LAYER = -1
DEFAULT_THRESHOLD = 0.5
POSITIVE_LABEL = "model-positive"
NEGATIVE_LABEL = "model-negative"


def main(
    input_fasta,
    model_dirpath,
    output_filepath=DEFAULT_OUTPUT_FILEPATH,
    esm_model=DEFAULT_ESM_MODEL,
    layer=DEFAULT_LAYER,
    batch_size=DEFAULT_TOKS_PER_BATCH,
    threshold=DEFAULT_THRESHOLD,
    embedding_output=None,
    keep_embeddings=False,
):
    """
    Embed sequences, score with the trained model, write CSV predictions.

    Labels follow the manuscript: model-positive / model-negative.
    A sequence is model-positive when P(positive) >= threshold.
    """
    input_fasta = str(input_fasta)
    if not os.path.exists(input_fasta):
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    model_path = pathlib.Path(model_dirpath) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find a trained model at {model_path}.\n"
            "Pass --model-dirpath to a directory containing model.joblib, "
            "or reinstall the package so bundled models are present."
        )

    output_filepath = pathlib.Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    own_temp = False
    if embedding_output is None:
        # Keep temp embeddings beside the output CSV (not CWD / system /tmp).
        fd, embedding_output = tempfile.mkstemp(
            prefix="phageminiprot_",
            suffix=".npy",
            dir=str(output_filepath.parent),
        )
        os.close(fd)
        own_temp = True
    else:
        embedding_output = str(embedding_output)
        pathlib.Path(embedding_output).parent.mkdir(parents=True, exist_ok=True)

    try:
        print("Generating embeddings...", flush=True)
        embed_fasta(
            fasta_file=input_fasta,
            output_filepath=embedding_output,
            model_name=esm_model,
            layer=layer,
            toks_per_batch=batch_size,
        )

        print("Loading embeddings and model...", flush=True)
        embeddings = np.load(embedding_output)
        model = joblib.load(model_path)

        print("Classifying sequences...", flush=True)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(embeddings)[:, 1]
        else:
            probabilities = model.predict(embeddings).astype(float)

        records = list(SeqIO.parse(input_fasta, "fasta"))
        if len(records) != len(probabilities):
            raise RuntimeError(
                f"FASTA has {len(records)} sequences but model scored "
                f"{len(probabilities)} embeddings."
            )

        print(f"Saving predictions to {output_filepath}...", flush=True)
        with open(output_filepath, "w") as f:
            f.write("Sequence_ID,Prediction,Probability\n")
            for record, prob in zip(records, probabilities):
                label = POSITIVE_LABEL if float(prob) >= threshold else NEGATIVE_LABEL
                f.write(f"{record.id},{label},{float(prob):.6f}\n")

        print("Classification completed.", flush=True)
    finally:
        if own_temp and not keep_embeddings and os.path.exists(embedding_output):
            try:
                os.remove(embedding_output)
            except OSError:
                pass


if __name__ == "__main__":
    print("Use: PhageMiniProt classify <fasta> [-o predictions.csv]", file=sys.stderr)
    sys.exit(2)
