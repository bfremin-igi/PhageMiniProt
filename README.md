# PhageMiniProt

Classify small proteins (typically ≤50 aa) with **ESM-2** mean embeddings and an **XGBoost** classifier.

Bundled models:

| Flag | Model | Training domain |
|------|--------|-----------------|
| *(default)* | **PhageMiniProt** | phage / prophage small proteins |
| `--meta` | **MetaMiniProt** | HMP-II microbiome small proteins |

Paper labels: **model-positive** / **model-negative** (CSV `Prediction` column). A sequence is model-positive when `Probability >= --threshold` (default `0.5`).

This tool takes **protein FASTA**. ORF finding / gene calling is upstream.

## Install

```bash
git clone https://github.com/bfremin-igi/PhageMiniProt.git
cd PhageMiniProt
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install .
```

GPU is optional. CPU works; first run downloads the ESM-2 weights via `fair-esm`.

## Quick start

```bash
# PhageMiniProt (default)
PhageMiniProt classify examples/tiny.faa -o preds_phage.csv

# MetaMiniProt
PhageMiniProt classify examples/tiny.faa -o preds_meta.csv --meta

# Custom probability cutoff
PhageMiniProt classify examples/tiny.faa -o preds.csv --threshold 0.7
```

Example output:

```text
Sequence_ID,Prediction,Probability
test_pos_like,model-positive,0.912345
test_short,model-negative,0.203411
```

`examples/tiny.faa` is a smoke-test FASTA (3 short sequences). Expect mixed labels; it is not a gold-standard set.

## Options (classify)

| Option | Meaning |
|--------|---------|
| `-o / --output-filepath` | CSV path (default `./predictions.csv`) |
| `--meta` | Use bundled MetaMiniProt |
| `--model-dirpath DIR` | Custom directory containing `model.joblib` |
| `--threshold FLOAT` | Positive cutoff on probability (default `0.5`) |
| `--batch-size INT` | Approx. tokens per ESM batch (default `4096`) |
| `--embedding-output PATH` | Keep embeddings `.npy` (otherwise a temp file is used) |

```bash
PhageMiniProt classify --help
```

## Train (optional)

Retrain from precomputed embedding matrices (`.npy`):

```bash
PhageMiniProt train \
  --positive-class-filepath pos.npy \
  --negative-class-filepath neg.npy \
  --model-dirpath ./my_model
```

**Note:** the CLI uses a **random stratified** holdout. Published PhageMiniProt metrics used a **cluster-aware** 90:10 split. The models shipped under `phageminiprot/models/` are the paper models.

## Requirements

Python ≥ 3.8. Main dependencies: `torch`, `fair-esm`, `xgboost`, `scikit-learn`, `biopython`, `click`, `numpy`, `joblib`, `tqdm`.

Pinned versions (optional): `pip install -r requirements.txt` then `pip install .`.

## Citation

If you use this tool, please cite the MiniProt / PhageMiniProt manuscript associated with this repository.

## License

MIT — see [LICENSE](LICENSE).
