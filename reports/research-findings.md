Historical saved coursework results, **not a fresh rerun** and **not results of src/train.py**. Logistic regression reported 99.8402% accuracy but only 9.1837% fraud recall. KNN reported 48.9796% recall and 58.8957% F1; SVM reported 95.4545% precision but 21.4286% recall. KNN captured more positives in this recorded experiment, but preprocessing leakage limits any generalization claim.

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Logistic regression | 0.818182 | 0.091837 | 0.165138 |
| KNN | 0.738462 | 0.489796 | 0.588957 |
| SVM | 0.954545 | 0.214286 | 0.350000 |
