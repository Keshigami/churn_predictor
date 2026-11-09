from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_renders(trained_models):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("Model comparison" in s.value for s in at.subheader)
    assert len(at.dataframe) >= 1
    assert len(at.metric) == 3


def test_form_scores_customer(trained_models):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.slider(key="input").set_value(3)
    at.selectbox(key="cat_Contract").select("Month-to-month")
    at.selectbox(key="cat_InternetService").select("Fiber optic")
    at.button(key="run").click().run()
    assert not at.exception
    texts = [e.value for e in at.error] + [s.value for s in at.success]
    assert any("risk of churn" in t for t in texts), texts
    assert at.json


def test_sample_button_then_predict(trained_models):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.button(key="sample").click().run()
    assert at.slider(key="input").value == 3
    at.button(key="run").click().run()
    assert not at.exception
    assert at.error, "the sample customer is high risk"
