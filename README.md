# Watch Brand & Price Prototype

Identifies a watch's brand from a photo (Phase 1), then estimates its current
market value from the brand plus listing attributes (Phase 2).

## Setup

Use Python 3.10+.

1. Create and activate a virtual environment:

   **Windows (PowerShell)**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (For an NVIDIA GPU, install the CUDA build of torch instead — see the note in `requirements.txt`.)

3. Create the data/output folders:

   **Windows:** `mkdir data\images artifacts`
   **macOS / Linux:** `mkdir -p data/images artifacts`

4. Verify the install:
   ```bash
   python check_env.py
   ```

## Project structure
```
.
├── train_watch_classifier.py     # Phase 1: watch brand classifier (transfer learning)
├── train_price_model.py    # Phase 2: XGBoost price estimator + baselines
├── e2eprediction.py              # Phase 1 + Phase 2 end to end
├── check_env.py            # sanity-check the environment
├── requirements.txt
├── data/
│   ├── images/             # one subfolder per brand
│   │   ├── Rolex/          #   e.g. data/images/Rolex/img001.jpg
│   │   └── Omega/
│   └── watch_prices.csv    # columns: brand,model_family,year,condition,material,size_mm,box_papers,price
└── artifacts/              # created by training (saved models + class list)
```

## Run order
```bash
python train_classifier.py     # writes artifacts/watch_classifier.pt + classes.json
python train_price_model.py    # writes artifacts/price_model.joblib
python predict.py data/images/Rolex/img001.jpg
```

## Notes
- Training the image model on CPU works for a small prototype but is slow; use
  Google Colab's free GPU to scale up.
- Start with 3–4 brands and a couple hundred images each — the first goal is to
  prove the pipeline runs end to end, not to hit high accuracy.
```