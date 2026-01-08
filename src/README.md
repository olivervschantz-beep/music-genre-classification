# Source Code Documentation

This directory contains the implementation of the music genre classification pipeline.

## Module Overview

### Core Logic (`elec_c5220_project/`)
* **`data.py`**: Handles loading of the GTZAN dataset, splitting into train/test sets, and managing audio file paths.
* **`sigproc.py`** *(Signal Processing)*: Contains functions for feature extraction (MFCCs, Spectral Centroid, etc.).
* **`model.py`**: Defines the machine learning model architectures (Logistic Regression, Random Forest, and KNN implementations).
* **`losses.py`**: Metric calculations and loss functions.
* **`trainer.py`**: The main training loop that fits the models to the training data and validates performance.
* **`utils.py`**: Helper functions for saving/loading models, setting random seeds, and plotting results.

### Usage
To train the model, the main entry point is run via the trainer module:
```bash
python -m src.elec_c5220_project.trainer
