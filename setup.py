from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="PhageMiniProt",
    version="0.3.0",
    author="Brayon Fremin",
    author_email="bfremin@berkeley.edu",
    description="Classify small proteins with ESM-2 embeddings and XGBoost (MiniProt Suite).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bfremin-igi/PhageMiniProt",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    package_data={
        "phageminiprot": [
            "models/PhageMiniProt_model/model.joblib",
            "models/MetaMiniProt_model/model.joblib",
        ],
    },
    # Model dirs have no __init__.py; silence setuptools "package data" warnings
    # by not treating them as importable packages.

    entry_points={
        "console_scripts": [
            "PhageMiniProt=phageminiprot.main:main",
        ],
    },
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "torch>=2.0",
        "tqdm>=4.64",
        "click>=8.0",
        "joblib>=1.2",
        "biopython>=1.80",
        "xgboost>=1.7,<3",
        "fair-esm>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
)
