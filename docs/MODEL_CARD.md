# Model card

## Intended use

The model provides a reproducible baseline for ranking anonymized card transactions by fraud risk. It is designed for portfolio demonstration and offline experimentation, not live transaction decisions.

## Model

- estimator: class-weighted logistic regression;
- preprocessing: standardization of `Time` and `Amount`;
- retained inputs: anonymized components `V1` through `V28`;
- split strategy: stratified 60% train, 20% validation, and 20% test;
- threshold policy: maximize validation precision while satisfying a configurable minimum recall;
- default minimum validation recall: 0.80.

## Why threshold selection matters

The default probability threshold of 0.50 does not encode the operational cost of missed fraud or unnecessary reviews. This project chooses the threshold on validation data, then evaluates that locked threshold once on the test partition. The process avoids tuning against test results.

## Evaluation

The exported report includes:

- average precision;
- ROC AUC;
- precision, recall, and F1 score at the selected threshold;
- true negatives, false positives, false negatives, and true positives;
- precision-recall and ROC curves;
- a held-out confusion matrix.

Accuracy is deliberately excluded from the primary metric set because the severe class imbalance makes it easy to misinterpret.

## Limitations and risks

- Transactions were collected during two days in September 2013.
- Most predictors are anonymized principal components, limiting interpretability.
- The historical fraud rate may not represent another institution, geography, or period.
- The model does not incorporate review capacity, monetary loss, customer friction, or delayed labels.
- Fraud tactics and payment behaviour change over time.
- Class weighting can improve detection while increasing false-positive alerts.

## Production requirements

Before deployment, the workflow would need temporal validation, probability calibration, business-cost simulation, subgroup and stability analysis, feature and prediction drift monitoring, secure feature handling, alert-volume limits, documented human review, and formal approval.
