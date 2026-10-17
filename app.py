from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from maligpx_engine import (
    ACCEPTED_MODEL_SEED,
    ACCEPTED_SCORE_SHA256,
    EXPECTED_ELIGIBLE_PATIENTS,
    EXPECTED_PRIMARY_CELLS,
    EXPECTED_REPRESENTED_PATIENTS,
    human_analysis,
    load_and_verify_scores,
    run_fast_validation,
    self_test,
)

DATA_PATH = Path(__file__).with_name("G6_STEELE_PRIMARY_MALIGNANT_PROGRAM_SCORES.tsv")

st.set_page_config(
    page_title="MaligPx Live Validation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: #ffffff; color: #171717; }
    [data-testid="stHeader"] { background: #ffffff; }
    .block-container { max-width: 1080px; padding-top: 2rem; padding-bottom: 3rem; }
    html, body, [class*="css"] { font-family: Arial, Helvetica, sans-serif; }

    h1, h2, h3 { color: #171717; font-weight: 600; letter-spacing: 0; }
    h1 { font-size: 2rem !important; margin-bottom: .15rem !important; }
    h2 { font-size: 1.3rem !important; margin-top: 1.6rem !important; }
    p, li, label { color: #333333; }

    .clinical-subtitle { color: #5f5f5f; font-size: .95rem; margin-bottom: 1rem; }
    .clinical-rule { border-top: 1px solid #d9d9d9; margin: 1rem 0 1.2rem; }
    .result-box {
      background: #fafafa; border: 1px solid #d5d5d5; border-left: 3px solid #7a0019;
      border-radius: 2px; padding: .85rem .95rem; margin-bottom: .75rem;
    }
    .result-heading { font-weight: 600; color: #171717; margin-bottom: .3rem; }
    .result-copy { color: #333333; line-height: 1.5; }
    .result-note { color: #666666; font-size: .88rem; margin-top: .45rem; }

    div[data-testid="stMetric"] {
      background: #ffffff; border: 1px solid #dddddd; border-radius: 3px; padding: .65rem .8rem;
    }
    div[data-testid="stMetricLabel"] { color: #666666; font-size: .82rem; }
    div[data-testid="stMetricValue"] { color: #171717; font-size: 1.2rem; font-weight: 600; }

    .stButton > button {
      width: 100%; background: #7a0019; color: #ffffff; border: 1px solid #7a0019;
      font-weight: 600; border-radius: 3px; padding: .62rem 1rem;
    }
    .stButton > button:hover { background: #620014; color: #ffffff; border-color: #620014; }
    div[data-testid="stExpander"] {
      background: #ffffff; border: 1px solid #dddddd; border-radius: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def verified_data():
    df = load_and_verify_scores(DATA_PATH)
    return df

def fig_patient_delta(result):
    p = result.patients.sort_values("M2_minus_M1")
    vals = p["M2_minus_M1"].to_numpy(float)
    labels = p["patient_key"].astype(str).str.replace("PDAC_TISSUE_", "T", regex=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    y = np.arange(len(vals))
    ax.axvline(0, color="#888888", lw=0.8)
    colors = np.where(vals >= 0, "#333333", "#7a0019")
    ax.hlines(y, 0, vals, color=colors, lw=1.4)
    ax.scatter(vals, y, c=colors, s=34, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color="#dddddd")
    ax.set_xlabel("Held-out mean Δ log copula density (M2 − M1)")
    ax.set_title("Patient-specific versus shared dependence", loc="left", fontsize=11)
    ax.tick_params(colors="#333333")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#bdbdbd")
    fig.tight_layout()
    return fig

def fig_pooling(result):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axhline(0, color="#aaaaaa", lw=0.8)
    ax.scatter([0], [result.median_patient_rho], s=95, color="#eeeeee", zorder=3)
    ax.scatter([1], [result.pooled_rho], s=95, color="#960018", zorder=3)
    ax.plot(
        [0, 1],
        [result.median_patient_rho, result.pooled_rho],
        color="#888888",
        lw=1.0,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Median patient ρ", "Pooled-cell ρ"])
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Pooling effect", loc="left", fontsize=11)
    ax.tick_params(colors="#333333")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#bdbdbd")
    fig.tight_layout()
    return fig

def patient_scatter(df, patient):
    sub = df.loc[df["patient_key"].astype(str) == patient].copy()
    rho = sub[["Classical_raw", "Basal_raw"]].corr(method="spearman").iloc[0, 1]
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.scatter(
        sub["Classical_raw"],
        sub["Basal_raw"],
        s=9,
        alpha=0.35,
        color="#4a4a4a",
        linewidths=0,
    )
    ax.set_xlabel("Classical-like score")
    ax.set_ylabel("Basal-like score")
    ax.set_title(f"{patient.replace('PDAC_TISSUE_', 'T')} · n={len(sub):,} · ρ={rho:+.3f}",
                 loc="left", fontsize=11)
    ax.tick_params(colors="#333333")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#bdbdbd")
    fig.tight_layout()
    return fig, float(rho), len(sub)

st.title("MaligPx Live Validation")
st.markdown(
    '<div class="clinical-subtitle">Independent Steele-cohort validation of patient-specific Classical–Basal dependence.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="clinical-rule"></div>', unsafe_allow_html=True)

try:
    self_test()
    df = verified_data()
except Exception as exc:
    st.error(f"Startup integrity check failed: {exc}")
    st.stop()

with st.expander("Why Steele is used", expanded=False):
    st.write(
        """
        **Peng was the discovery cohort. Steele is the independent replication cohort.**
        Running the live test on Steele asks a harder question than simply rerunning the
        dataset in which the pattern was first established: does patient-specific
        Classical–Basal dependence still improve held-out modeling in independent data?

        **G6** identifies the frozen Phase-6 Steele scoring output used by this validation.
        The table contains the accepted primary-malignant cell population and the
        Classical-like and Basal-like scores required by the copula test. The web app
        verifies its exact SHA-256 before any result is calculated, so a modified or
        substituted table is rejected.
        """
    )

with st.expander("How the seed works", expanded=False):
    st.write(
        f"""
        The seed is the starting state of the pseudorandom number generator. It does **not**
        change the cells, the score values, the model equations, patient eligibility,
        calibration fraction, or decision rules. It determines which cells enter the
        calibration versus held-out evaluation split on each repeat, and it also determines
        the bootstrap/sign-flip resampling stream.

        **Seed {ACCEPTED_MODEL_SEED}** is the frozen study seed. Entering a different integer
        gives a fresh judge-selected split. The same seed produces the same split sequence
        when the pinned numerical environment is used, making the run reproducible rather
        than arbitrary. A fresh seed is useful as a small live robustness challenge:
        the judge can change the random split without changing the scientific method.
        """
    )

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=4_294_967_295,
        value=ACCEPTED_MODEL_SEED,
        step=1,
        help="Use the frozen seed to reproduce the study split stream, or enter another integer for a fresh split.",
    )
with right:
    st.markdown("<div style='height:1.78rem'></div>", unsafe_allow_html=True)
    run_clicked = st.button("Run validation", type="primary")

a, b, c, d = st.columns(4)
a.metric("Primary malignant cells", f"{EXPECTED_PRIMARY_CELLS:,}")
b.metric("Represented patients", EXPECTED_REPRESENTED_PATIENTS)
c.metric("Eligible patients", EXPECTED_ELIGIBLE_PATIENTS)
d.metric("Integrity check", "PASS")

st.caption(f"Accepted input SHA-256: `{ACCEPTED_SCORE_SHA256}`")

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.elapsed = None

if run_clicked:
    with st.spinner("Running validation..."):
        t0 = time.perf_counter()
        try:
            st.session_state.result = run_fast_validation(DATA_PATH, int(seed))
            st.session_state.elapsed = time.perf_counter() - t0
        except Exception as exc:
            st.error(f"Run failed: {exc}")
            st.stop()

result = st.session_state.result
if result is None:
    st.info("Enter a seed and select Run validation.")
    st.stop()

analysis = human_analysis(result)

st.markdown("## Results")
m1, m2, m3, m4 = st.columns(4)
m1.metric("M2 − M1", f"{result.mean_delta:+.5f}")
m2.metric("95% patient CI", f"{result.ci_low:+.4f} to {result.ci_high:+.4f}")
m3.metric("Patients favoring M2", f"{result.positive_patients}/{EXPECTED_ELIGIBLE_PATIENTS}")
m4.metric("Sign-flip p", f"{result.signflip_p:.4g}")

supported = result.mean_delta > 0 and result.ci_low > 0 and result.signflip_p < 0.05
verdict = "Supported in this run" if supported else "Not supported in this run"

st.markdown(
    f'<div class="result-box"><div class="result-heading">Interpretation: {verdict}</div>'
    f'<div class="result-copy">{analysis["model"]}</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="result-box"><div class="result-heading">Pooling analysis</div>'
    f'<div class="result-copy">{analysis["pooling"]}</div>'
    f'<div class="result-note">{analysis["support"]}</div></div>',
    unsafe_allow_html=True,
)

if int(result.seed) == ACCEPTED_MODEL_SEED:
    check_text = (
        "The frozen-seed live effect agrees with the accepted Phase-6 effect within "
        "the predeclared fast-demo tolerance."
        if result.accepted_seed_check == "PASS"
        else "The frozen-seed fast-demo effect does not match the accepted effect within tolerance. Check the pinned numerical environment before public deployment."
    )
    st.info(f"Accepted-seed check: **{result.accepted_seed_check}**. {check_text}")
else:
    st.info(
        "Fresh judge seed: this is a new calibration/evaluation split, not the frozen study run."
    )

st.caption(
    f"Run time on this server: {st.session_state.elapsed:.3f} s. "
    "Endpoint: held-out joint Classical-like/Basal-like dependence fit. "
    "This is not survival, treatment-response, progression, or clinical prediction."
)

tab1, tab2, tab3 = st.tabs(["Figures", "Patient data", "Downloads"])

with tab1:
    c1, c2 = st.columns([1.2, 0.8], gap="large")
    with c1:
        fig = fig_patient_delta(result)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with c2:
        fig = fig_pooling(result)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with tab2:
    eligible_patients = sorted(result.associations["patient_key"].astype(str).tolist())
    chosen = st.selectbox(
        "Patient",
        eligible_patients,
        format_func=lambda x: x.replace("PDAC_TISSUE_", "T"),
    )
    fig, rho, n = patient_scatter(df, chosen)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.write(
        f"{chosen.replace('PDAC_TISSUE_', 'T')}: {n:,} eligible malignant cells; "
        f"within-patient Classical–Basal Spearman ρ = {rho:+.3f}."
    )

with tab3:
    assoc = result.associations.merge(
        result.patients[["patient_key", "M2_minus_M1", "M2_minus_M1_sd_across_repeats"]],
        on="patient_key",
        how="left",
    )
    assoc["favors"] = np.where(assoc["M2_minus_M1"] > 0, "M2 patient-specific", "M1 shared")
    st.dataframe(
        assoc.rename(columns={
            "patient_key": "Patient",
            "n_cells": "Cells",
            "spearman_rho": "Spearman rho",
            "M2_minus_M1": "M2 − M1",
            "M2_minus_M1_sd_across_repeats": "Split SD",
            "favors": "Held-out direction",
        }),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = assoc.to_csv(index=False).encode("utf-8")
    summary = {
        "seed": int(result.seed),
        "input_sha256": result.source_sha256,
        "primary_cells": EXPECTED_PRIMARY_CELLS,
        "represented_patients": EXPECTED_REPRESENTED_PATIENTS,
        "eligible_patients": EXPECTED_ELIGIBLE_PATIENTS,
        "calibration_fraction": 0.50,
        "repeats_per_patient": 10,
        "M2_minus_M1": result.mean_delta,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "signflip_p": result.signflip_p,
        "positive_patients": result.positive_patients,
        "negative_patients": result.negative_patients,
        "median_patient_rho": result.median_patient_rho,
        "pooled_rho": result.pooled_rho,
        "accepted_seed_check": result.accepted_seed_check,
        "claim_boundary": (
            "Held-out joint Classical-like/Basal-like dependence fit; "
            "not survival, treatment response, progression, or clinical prediction."
        ),
    }
    json_bytes = json.dumps(summary, indent=2).encode("utf-8")

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download patient-level CSV",
        data=csv_bytes,
        file_name=f"maligpx_patient_results_seed{result.seed}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d2.download_button(
        "Download run summary JSON",
        data=json_bytes,
        file_name=f"maligpx_run_summary_seed{result.seed}.json",
        mime="application/json",
        use_container_width=True,
    )

with st.expander("Methods used in this live validation"):
    st.write(
        """
        1. Verify the frozen Steele score table by SHA-256 and schema.
        2. Keep patients with at least 30 cells and variable Classical/Basal scores.
        3. For each patient, estimate a shared leave-one-patient-out Gaussian-copula dependence (M1).
        4. Use 50% of that patient's cells to calibrate a patient-specific dependence and shrink it toward M1 (M2).
        5. Score M1 and M2 on the held-out 50% of cells.
        6. Repeat the split 10 times per patient.
        7. Average within patient, then perform patient-level bootstrap and sign-flip inference.
        8. Separately compare the median within-patient Spearman correlation with the pooled-cell correlation.
        """
    )


