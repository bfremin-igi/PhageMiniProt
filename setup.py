from setuptools import setup, find_packages

# Read the README file for a long description (if you have one)
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="PhageMiniProt",
    version="0.2.0",  # Bumped: adds `train` CLI command, holdout eval, bundled model data
    author="Brayon Fremin",
    author_email="bfremin@berkeley.edu",
    description="A tool for classifying phage proteins using MiniProt embeddings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bfremin-igi/PhageMiniProt",
    packages=find_packages(),  # Automatically find package directories
    include_package_data=True,
    package_data={
        # Model directories now live inside the package (phageminiprot/models/...)
        # so they ship with `pip install .` instead of only existing in the repo.
        "phageminiprot": [
            "models/PhageMiniProt_model/*",
            "models/MetaMiniProt_model/*",
        ],
    },
    entry_points={
        "console_scripts": [
            "PhageMiniProt=phageminiprot.main:main",  # Main entry point
        ],
    },
    install_requires=[
        "pathlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",  # If using PyTorch in embed.py
        "tqdm",  # Progress bar
        "click",  # Command-line interface
        "joblib",  # For model serialization
        "biopython",  # For sequence I/O
        "xgboost",  # For XGBoost classifier
        "fair-esm",  # Likely for embedding model
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
