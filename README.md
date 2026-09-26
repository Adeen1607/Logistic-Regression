# Credit Card Fraud Detection

An end-to-end classification case study for detecting rare fraudulent card transactions. The project focuses on the parts of fraud modelling that accuracy alone can hide: class imbalance, false-negative cost, decision thresholds, leakage-safe preprocessing, and clear evaluation.

## Business problem

A useful fraud model must identify suspicious transactions while controlling the number of legitimate transactions sent for review. Because fraud is rare, a model can appear accurate while missing most fraud cases. This project therefore prioritizes precision, recall, F1 score, precision-recall AUC, ROC AUC, and confusion-matrix counts.

## Data source

The analysis uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data), originally released through a collaboration between Worldline and the Machine Learning Group of Université Libre de Bruxelles.

The dataset is not redistributed in this repository. Download `creditcard.csv` from the source and place it in `data/raw/`.

## Dataset

- 284,807 anonymized European card transactions
- 492 fraud observations
- `V1`–`V28`: principal-component features
- `Time` and `Amount`: non-transformed attributes
- `Class`: binary target, where `1` indicates fraud

## Modelling workflow

1. Validate the expected schema and binary target.
2. Create stratified train, validation, and test partitions.
3. Fit preprocessing only on training data.
4. Scale `Time` and `Amount` while retaining anonymized features.
5. Train a class-weighted logistic-regression baseline.
6. Select a decision threshold on validation data.
7. Lock the threshold and evaluate once on the held-out test set.
8. Export metrics, predictions, and diagnostic charts.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/train.py \
  --data data/raw/creditcard.csv \
  --output artifacts \
  --minimum-recall 0.80
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Outputs

- `metrics.json`: validation threshold and held-out test metrics
- `test_predictions.csv`: labels, probabilities, and locked predictions
- `precision_recall_curve.png`
- `roc_curve.png`
- `confusion_matrix.png`

## Tools demonstrated

Python, pandas, scikit-learn, stratified sampling, logistic regression, class weighting, threshold selection, precision-recall analysis, ROC analysis, and reproducible experiment design.

## Responsible use

This is an educational benchmark on historical, anonymized data. A production fraud system would also require probability calibration, drift monitoring, cost-sensitive validation, latency testing, explainability controls, human-review workflows, and periodic model governance.
