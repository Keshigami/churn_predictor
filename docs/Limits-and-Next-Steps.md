# Limits and next steps

- The data is synthetic; nothing here says how the model would do on a real telecom book.
- No calibration or threshold tuning: 0.5 is used as the cut, a retention team would pick a cut by call capacity.
- Feature set is the generator's 11 columns; real data would add usage, support tickets and payment history.
- Next: XGBoost with the same pipeline, SHAP for per-customer reasons, and this model is already served by
  `model_monitoring_api` (port 8115), which is the "deploy as an API with monitoring" step from the original plan.
