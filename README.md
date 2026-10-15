# Exploratory GoEmotions Analysis of Video Game Reviews

Exploratory Python code developed during the research process associated with:

Cho, H., Lee, W.-C., Thach, H., & Hirt, J. (2025). *Evaluating video game moods and their separability based on user-generated reviews*. *Journal of Documentation, 81*(2), 545–565. https://doi.org/10.1108/JD-10-2024-0247

**This experiment was not part of the published article's reported analytical methods or findings.** It is shared as an example of exploratory research practice.

## What this repository contains

- `src/goemotions.py`: A lightly cleaned version of the original Google Colab script. It still uses the same model, text cleaning, tokenization, stopword removal, stemming, per-review classification, and emotion-frequency counting.
- `data/example_reviews.csv`: 150 **synthetic**, invented video game reviews for demonstrating the code. They are not user-submitted Steam reviews and must not be treated as empirical findings.
- `requirements.txt`: Python dependencies.

## Run

From the repository's top-level directory, install dependencies and run:

```bash
python -m pip install -r requirements.txt
python src/goemotions.py
```

The Hugging Face model will be downloaded the first time it runs. The program prints a frequency table of predicted emotions. A CPU-only run may take time. It does **not** publish or download the original Steam dataset.

## Data and collaboration

The original dataset is not included. The repository intentionally distributes only synthetic example reviews. This exploratory analysis was conducted in the context of collaborative research; the repository does not claim that this was the methodology used in the published paper.
