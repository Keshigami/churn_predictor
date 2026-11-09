# Getting started

## What you need
- [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just). On a Mac: `brew install uv just`.
- Python 3.11 is installed by uv.
- No API key is needed anywhere in this project.

## Three commands
```bash
git clone https://github.com/Keshigami/churn_predictor.git && cd churn_predictor
just setup
just up           # http://127.0.0.1:8111/  (just down to stop)
```
Everything the app needs is in the repo (synthetic sample data where a dataset is used).

## Commands
`just setup`, `just train` where a model is trained, `just up` / `just down`, `just test`, `just demo`.

## What you see
- Run: `just churn_predictor-setup && just churn_predictor-up`, open `http://127.0.0.1:8111/`.
- Two-minute walkthrough: (1) read the comparison table and say why LR is champion here; (2) look at the importance
  chart (tenure, charges, contract); (3) press "Use sample customer", then "Predict" for a red high-risk card;
  (4) press "Use safe customer", "Predict", green card; flip the contract to Month-to-month and watch it move.
- Video clip: `demo/clips/portfolio__churn_predictor.webm` (recorded by `just demo-video`).
- Sample inputs: the two buttons. Sample customer (3 months, $104, fiber, monthly) scores about 90% churn; safe
  customer (60 months, $30, DSL, two-year) scores under 10%.

## Try this
Try it: Use sample customer, Predict.
