import pathlib
import click
import esm
import numpy as np
import torch
import tqdm
from Bio import SeqIO

MODEL_OPTIONS = {
    "esm2_t48_15B_UR50D": 5120,
    "esm2_t36_3B_UR50D": 2560,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t6_8M_UR50D": 320,
}

@click.command()
@click.option("--output-filepath", type=click.Path(), required=True, help="Path to save embeddings (.npy).")
@click.argument("fasta_file", type=click.Path(exists=True))
@click.option(
    "--model",
    type=click.Choice(MODEL_OPTIONS.keys()),
    default="esm2_t6_8M_UR50D",
    help="Select the ESM model variant.",
)
@click.option("--layer", type=int, default=-1, help="Layer index for embeddings (-1 for final).")
def generate_embeddings(output_filepath, fasta_file, model, layer):
    """Generates embeddings for sequences in a FASTA file."""
    output_filepath = pathlib.Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model, alphabet = esm.pretrained.load_model_and_alphabet(model)
    model.eval().to(device)

    dataset = esm.FastaBatchedDataset.from_file(fasta_file)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(1022),
        batch_sampler=dataset.get_batch_indices(toks_per_batch=4096, extra_toks_per_seq=1),
    )

    if layer == -1:
        layer = model.num_layers

    sequence_ids, mean_embeddings = [], []
    with torch.no_grad():
        for headers, sequences, tokens in tqdm.tqdm(dataloader, total=len(dataloader)):
            tokens = tokens.to(device)
            outputs = model(tokens, repr_layers=[layer], return_contacts=False)
            embeddings = outputs["representations"][layer]

            for i, (header, sequence) in enumerate(zip(headers, sequences)):
                valid_length = min(len(sequence), 1022)
                mean_embedding = embeddings[i, 1:valid_length + 1].mean(dim=0).cpu().numpy()
                sequence_ids.append(header.split()[0])
                mean_embeddings.append(mean_embedding)

    mean_embeddings = np.stack(mean_embeddings)
    fasta_ids = [record.id for record in SeqIO.parse(fasta_file, "fasta")]
    id_to_index = {seq_id: i for i, seq_id in enumerate(sequence_ids)}
    reordered_embeddings = mean_embeddings[[id_to_index[fid] for fid in fasta_ids]]
    np.save(output_filepath, reordered_embeddings)

if __name__ == "__main__":
    generate_embeddings()

