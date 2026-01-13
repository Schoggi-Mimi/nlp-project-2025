# NLP Project — Tasks 1–3

This repository contains our solution for the NLP & Text Mining project (University of Bern, AI in Medicine).
It includes three Jupyter notebooks (Tasks 1–3), source code helpers, and the presentation slides.

## Repository structure
- `notebooks/`
  - `01_explore_dataset.ipynb` — Task 1 (data exploration + dataset understanding)
  - `02_ner_training.ipynb` — Task 2 (custom NER training + evaluation)
  - `03_prediction_model.ipynb` — Task 3 (baseline + encoder + decoder transformer experiments)
- `src/` — helper code used by notebooks
- `data/` — small helper files / outputs (main dataset is downloaded from Hugging Face)
- `models/` — saved models (if present; large files should not be committed)
- `slides/` — in-class presentation slides (16 Dec)

## Data access
We do not bundle the full dataset in the repository.
The main dataset is downloaded directly from Hugging Face inside the notebook, e.g.:

```python
import pandas as pd
df = pd.read_parquet(
    "hf://datasets/argilla/medical-domain/data/train-00000-of-00001-67e4e7207342a623.parquet"
)
```

## Environment setup

```bash
# create env (conda)
conda create --name nlp python=3.11 -y
conda activate nlp

# core packages
conda install -c conda-forge -y \
  numpy pandas scikit-learn matplotlib tqdm \
  pytorch torchvision torchaudio \
  transformers datasets \
  spacy nltk langdetect jupyter jupyterlab ipykernel

# additional HF evaluation + training utilities used in Task 3
pip install -U evaluate accelerate peft trl
```