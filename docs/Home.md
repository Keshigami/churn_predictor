# Telecom Churn Predictor

Scores a telecom customer's churn risk with a Logistic Regression and a Random Forest trained once on synthetic data, in a Streamlit dashboard. Portfolio, ML Engineer, by Maximus Friedrich Baluyot.

**Watch first:** [the 90-second film](https://github.com/Keshigami/churn_predictor/blob/main/demo/film.mp4).

## Pages
- [Getting started](Getting-Started.md): run it in three commands, what you see, the API.
- [How it was built](How-It-Was-Built.md): from the brief to a running service, step by step.
- [Results](Results.md): the numbers and what they mean.
- [Limits and next steps](Limits-and-Next-Steps.md): what it cannot do yet.
- [The film](The-Film.md): what the four chapters show.

## The brief
Telecom operators lose revenue when customers switch providers, and the retention team cannot call everyone. This app
predicts which customers are likely to churn so outreach goes to the riskiest first. "Solved" means a scored list with a
model whose numbers are visible and reproducible, and a form to score one customer on demand.

## The one-line answer
0.806 · ROC-AUC · logistic regression · 200 held out

## One honest limit
Synthetic data: no real customers scored yet.
