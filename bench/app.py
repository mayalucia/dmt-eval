"""DMT-Eval Bench Service — live validation at bench.mayalucia.dev.

A Streamlit app that runs DMT evaluations and renders structured
scientific reports (LabReports) in real time.
"""
import sys
import tempfile
from pathlib import Path

# Add dmt-eval src to path (when running locally)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st

from dmt.evaluate import evaluate, WEATHER, DRUG_EFFICACY, Scenario
from dmt.scenario.weather import (
    generate_observations as weather_obs,
    PersistenceModel,
    ClimatologyModel,
    NoisyRegressionModel,
)
from dmt.scenario.drug_efficacy import (
    generate_observations as drug_obs,
    LinearModel,
    SigmoidModel,
    CalibratedModel,
)
from brainscore_cache import brainscore_report


# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DMT-Eval Bench",
    page_icon="🔬",
    layout="wide",
)

st.title("DMT-Eval — Live Validation Service")
st.caption(
    "Data, Models, Tests: a universal validation framework. "
    "Select a domain and run a structured scientific evaluation."
)

# ── Sidebar ─────────────────────────────────────────────────────────────────

domain = st.sidebar.radio(
    "Domain",
    ["Weather Prediction", "Drug Efficacy", "Brain-Score (NeuroAI)"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**DMT-Eval** produces structured scientific reports — "
    "abstract, methods, results, discussion — from "
    "(model, data) pairs.\n\n"
    "[Source on GitHub](https://github.com/mayalucia/dmt-eval) · "
    "[mayalucia.dev](https://mayalucia.dev)"
)

# ── Code snippets for reveal ────────────────────────────────────────────────

WEATHER_CODE = '''
from dmt.evaluate import evaluate, WEATHER
from dmt.scenario.weather import (
    generate_observations,
    PersistenceModel, ClimatologyModel, NoisyRegressionModel,
)

observations = generate_observations(n_days=365)
models = [PersistenceModel(), ClimatologyModel(), NoisyRegressionModel()]

report_path = evaluate(
    models=models,
    observations=observations,
    scenario=WEATHER,
    reference_model=models[0],
    title="Weather Prediction Evaluation",
)
'''.strip()

DRUG_CODE = '''
from dmt.evaluate import evaluate, DRUG_EFFICACY
from dmt.scenario.drug_efficacy import (
    generate_observations,
    LinearModel, SigmoidModel, CalibratedModel,
)

observations = generate_observations()
models = [LinearModel(), SigmoidModel(), CalibratedModel()]

report_path = evaluate(
    models=models,
    observations=observations,
    scenario=DRUG_EFFICACY,
    reference_model=models[0],
    title="Drug Efficacy Evaluation",
)
'''.strip()

BRAINSCORE_CODE = '''
# Brain-Score results are pre-computed (GPU required for live scoring).
# The same DMT report pipeline renders them as a structured LabReport.

from dmt.document.renderer import render_markdown
from collections import OrderedDict
import pandas as pd

results = pd.DataFrame([
    {"model": "AlexNet", "benchmark": "MajajHong2015.IT-pls",
     "raw_score": 0.48, "ceiling": 0.817, "normalized_score": 0.588},
    {"model": "AlexNet", "benchmark": "MajajHong2015.V4-pls",
     "raw_score": 0.55, "ceiling": 0.892, "normalized_score": 0.616},
])

sections = OrderedDict(
    results={"name": "Results", "narrative": "...", "data": results},
    # ... abstract, methods, discussion sections
)
report_path = render_markdown("Brain-Score Evaluation", sections, "./report")
'''.strip()


# ── Evaluation logic ────────────────────────────────────────────────────────

def run_weather():
    observations = weather_obs(n_days=365)
    models = [PersistenceModel(), ClimatologyModel(), NoisyRegressionModel()]
    with tempfile.TemporaryDirectory() as tmp:
        report_path = evaluate(
            models=models,
            observations=observations,
            scenario=WEATHER,
            reference_model=models[0],
            output_dir=tmp,
            title="Weather Prediction Evaluation",
        )
        return report_path.read_text()


def run_drug():
    observations = drug_obs()
    models = [LinearModel(), SigmoidModel(), CalibratedModel()]
    with tempfile.TemporaryDirectory() as tmp:
        report_path = evaluate(
            models=models,
            observations=observations,
            scenario=DRUG_EFFICACY,
            reference_model=models[0],
            output_dir=tmp,
            title="Drug Efficacy Evaluation",
        )
        return report_path.read_text()


def run_brainscore():
    with tempfile.TemporaryDirectory() as tmp:
        return brainscore_report(output_dir=tmp)


# ── Main area ───────────────────────────────────────────────────────────────

if domain == "Weather Prediction":
    st.header("Weather Prediction")
    st.markdown(
        "Synthetic AR(1) temperature data for 5 European cities. "
        "Three models: Persistence (baseline), Climatology, NoisyRegression."
    )
    with st.expander("Show code"):
        st.code(WEATHER_CODE, language="python")

    if st.button("Evaluate", key="weather"):
        with st.spinner("Running evaluation..."):
            report = run_weather()
        st.markdown(report)
        st.download_button(
            "Download report (.md)",
            data=report,
            file_name="dmt_weather_report.md",
            mime="text/markdown",
        )

elif domain == "Drug Efficacy":
    st.header("Drug Efficacy")
    st.markdown(
        "Synthetic Hill-equation dose-response for 3 compounds. "
        "Three models: Linear, Sigmoid (miscalibrated), Calibrated."
    )
    with st.expander("Show code"):
        st.code(DRUG_CODE, language="python")

    if st.button("Evaluate", key="drug"):
        with st.spinner("Running evaluation..."):
            report = run_drug()
        st.markdown(report)
        st.download_button(
            "Download report (.md)",
            data=report,
            file_name="dmt_drug_efficacy_report.md",
            mime="text/markdown",
        )

elif domain == "Brain-Score (NeuroAI)":
    st.header("Brain-Score — NeuroAI Model Evaluation")
    st.markdown(
        "Pre-computed Brain-Score results for AlexNet on macaque visual cortex "
        "benchmarks (MajajHong2015). Rendered through the same DMT LabReport pipeline."
    )
    with st.expander("Show code"):
        st.code(BRAINSCORE_CODE, language="python")

    if st.button("Evaluate", key="brainscore"):
        with st.spinner("Generating report..."):
            report = run_brainscore()
        st.markdown(report)
        st.download_button(
            "Download report (.md)",
            data=report,
            file_name="dmt_brainscore_report.md",
            mime="text/markdown",
        )
