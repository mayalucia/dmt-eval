"""Pre-computed Brain-Score results for the bench service.

These values come from the Brain-Score tutorials (codev/08-12).
Live scoring requires GPU and brainscore_vision; this module provides
the same data through the DMT report pipeline without dependencies.
"""
import pandas as pd
from collections import OrderedDict
from dmt.document.renderer import render_markdown

# From codev/08-brainscore-installation.org and codev/11-brainscore-benchmarks.org
BRAINSCORE_RESULTS = pd.DataFrame([
    {"model": "AlexNet", "benchmark": "MajajHong2015.IT-pls",
     "raw_score": 0.48, "ceiling": 0.817, "normalized_score": 0.588},
    {"model": "AlexNet", "benchmark": "MajajHong2015.V4-pls",
     "raw_score": 0.55, "ceiling": 0.892, "normalized_score": 0.616},
])


def brainscore_report(output_dir="./dmt_report") -> str:
    """Generate a LabReport from cached Brain-Score results."""
    sections = OrderedDict()
    sections["abstract"] = {
        "name": "Abstract",
        "narrative": (
            "We evaluate neural network models against primate neural recordings "
            "using the Brain-Score platform. Models are scored by comparing their "
            "internal representations to neural population responses in macaque "
            "visual cortex (V4, IT) using partial least squares regression, "
            "normalized by the data's split-half reliability ceiling."
        ),
    }
    sections["methods"] = {
        "name": "Methods",
        "narrative": (
            "**Platform**: Brain-Score (brain-score.org)\n\n"
            "**Metric**: PLS regression between model activations and neural "
            "recordings, ceiling-normalized by explained variance.\n\n"
            "**Data**: MajajHong2015 — 2560 presentations, 168 IT neurons, "
            "88 V4 neurons. Public subset.\n\n"
            "**Adapter**: DMT brain-score-dmt layer provides interface enforcement "
            "at registration time via @implements decorator and validated "
            "PluginRegistry."
        ),
    }
    sections["results"] = {
        "name": "Results",
        "narrative": "### Brain-Score Evaluation Results",
        "data": BRAINSCORE_RESULTS,
    }
    sections["discussion"] = {
        "name": "Discussion",
        "narrative": (
            "AlexNet achieves moderate brain-similarity scores (IT: 0.588, "
            "V4: 0.616 after ceiling normalization). These scores are consistent "
            "with published Brain-Score leaderboard values. The DMT adapter layer "
            "validates plugin compliance at registration time — before any "
            "benchmark runs — eliminating the late-failure mode where a model "
            "is discovered to lack required methods only at scoring time.\n\n"
            "**Note**: These results are pre-computed from Brain-Score v2. "
            "Live scoring requires GPU and the brainscore_vision package. "
            "The DMT framework wrapping demonstrated here is domain-agnostic — "
            "the same Scenario/evaluate/LabReport pattern applies to weather "
            "prediction, drug efficacy, and any domain with models and reference data."
        ),
    }
    report_path = render_markdown(
        "Brain-Score Model Evaluation (DMT LabReport)", sections, output_dir
    )
    return report_path.read_text()
