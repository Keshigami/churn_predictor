# Results

`models/metrics.json`, this machine, full data (training takes 3 s so there is no quick mode):

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression (champion) | 0.740 | 0.581 | 0.806 |
| Random Forest | 0.730 | 0.542 | 0.787 |

ROC-AUC 0.81 means the model ranks a random churner above a random non-churner 81% of the time. The original brief
wanted accuracy > 0.80 and F1 > 0.70; the synthetic churn rule is noisy by construction (probability drawn from four
indicators), so that target is not reachable on this data and the README says so instead of quoting it.
