"""Enzyme Function Classifier — Web Demonstration Application.

Single-page Streamlit application for classifying protein sequences into
Enzyme Commission top-level classes (EC 1–6) using the trained stacking ensemble
and exact training-consistent preprocessing pipeline.
"""

from typing import Dict
import pandas as pd
import streamlit as st

from src.inference import (
    EC_METADATA,
    EXAMPLE_SEQUENCES,
    clean_and_validate_sequence,
    load_inference_artifacts,
    predict_sequence,
)

# Page configuration
st.set_page_config(
    page_title="Enzyme Function Classifier | EC 1–6",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Biotechnology / Scientific CSS Theme
CUSTOM_CSS = """
<style>
    /* Global scientific typography and margins */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    code, pre, textarea {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main banner styling */
    .hero-container {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 50%, #203a43 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        color: #ffffff;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #f8fafc;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 500;
        color: #38bdf8;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }
    .hero-description {
        font-size: 0.95rem;
        color: #cbd5e1;
        line-height: 1.5;
        max-width: 900px;
        margin-bottom: 0;
    }

    /* Metric card */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* Prediction callout */
    .pred-callout {
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin: 1.5rem 0;
        background: linear-gradient(135deg, #f0fdf4 0%, #e6fcf5 100%);
        border: 2px solid #059669;
        box-shadow: 0 4px 15px rgba(5, 150, 105, 0.08);
    }
    .pred-badge {
        display: inline-block;
        background-color: #059669;
        color: #ffffff;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .pred-heading {
        font-size: 1.75rem;
        font-weight: 700;
        color: #064e3b;
        margin: 0.2rem 0;
    }
    .pred-detail {
        font-size: 0.95rem;
        color: #047857;
        margin: 0;
    }

    /* Small badge tag */
    .tag {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        background-color: #e2e8f0;
        color: #334155;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained enzyme models and feature preprocessors...")
def get_artifacts():
    """Load and cache serialized model artifacts once for all inference requests."""
    return load_inference_artifacts()


artifacts = get_artifacts()

# Sidebar: Model and Project Information
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")
    selected_model = st.selectbox(
        "Classifier Architecture",
        [
            "Stacking Ensemble",
            "Random Forest",
            "LightGBM",
            "Soft Voting Ensemble",
            "Calibrated Linear SVM",
        ],
        index=0,
        help="Stacking Ensemble achieved the highest test accuracy (79.77%) and test AUPRC (0.9016).",
    )

    st.markdown("---")
    st.markdown("### 📊 Benchmark Highlights")
    st.markdown(
        """
        - **Dataset**: SwissProt-EC ($212,273$ deduplicated sequences)
        - **Split**: 70% Train / 15% Val / 15% Held-Out Test
        - **Test Accuracy**: **79.77%** (Stacking)
        - **Test Macro F1**: **79.45%** (Random Forest)
        - **Test Macro AUPRC**: **0.9016** (Stacking)
        - **Feature Space**: 500 features (AAC 20D, DPC 400D, SVD 128D)
        """
    )

    st.markdown("---")
    st.markdown("### 📚 EC Class Reference")
    for ec_num, meta in EC_METADATA.items():
        st.markdown(f"**{meta['code']}** — *{meta['name']}*")

    st.markdown("---")
    st.caption("🔬 Built for professional portfolio demonstration | Deterministic seed 42 | No data leakage")


# Hero Header
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">Enzyme Function Classifier</h1>
        <div class="hero-subtitle">Protein Sequence &rarr; EC Class (1–6)</div>
        <p class="hero-description">
            Predict the primary catalytic Enzyme Commission (EC) top-level functional class from raw protein primary
            amino-acid sequences. Utilizes trained multi-scale biological representations (AAC, Dipeptide, Tripeptide SVD)
            and a calibrated ensemble architecture evaluated on the held-out SwissProt-EC benchmark.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Example sequence selection helper
col_ex1, col_ex2 = st.columns([3, 1])

with col_ex1:
    example_options = ["-- Select an Example Sequence from Test Partition --"] + list(EXAMPLE_SEQUENCES.keys())
    selected_example_key = st.selectbox(
        "Load a verified test sample sequence:",
        options=example_options,
        index=0,
        help="Quickly populate the input with an authentic SwissProt-EC test partition sequence.",
    )

with col_ex2:
    st.write("")
    st.write("")
    clear_button = st.button("Clear Input", use_container_width=True)

# Determine default sequence
default_sequence = ""
if selected_example_key and selected_example_key != example_options[0]:
    default_sequence = EXAMPLE_SEQUENCES[selected_example_key]["sequence"]

if clear_button:
    default_sequence = ""

# Sequence Input Area
st.markdown("#### 1. Paste Protein Amino-Acid Sequence")
user_input = st.text_area(
    "Protein amino-acid sequence (Plain text or FASTA format):",
    value=default_sequence,
    height=160,
    placeholder=">sp|example|ENZYME_TEST Sample enzyme sequence\nMKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE...",
    help="Supports FASTA headers (lines starting with >), multi-line sequences, and whitespace.",
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    predict_clicked = st.button("🔮 Predict Enzyme Function", type="primary", use_container_width=True)

with col_info:
    st.caption("📌 Note: Predicts top-level EC classes (1–6). Sequences are validated for standard 20 amino-acids.")

# Prediction Execution
if predict_clicked:
    if not user_input.strip():
        st.warning("⚠️ Please paste a protein amino-acid sequence or select an example above before predicting.")
    else:
        # 1. Clean and validate sequence
        is_valid, clean_seq, err_msg = clean_and_validate_sequence(user_input)

        if not is_valid:
            st.error(f"❌ **Sequence Validation Error**: {err_msg}")
        else:
            with st.spinner(f"Computing 500-dimensional biological features and predicting with {selected_model}..."):
                # Run exact training-consistent pipeline
                results = predict_sequence(clean_seq, model_name=selected_model, artifacts=artifacts)

            if not results["success"]:
                st.error(f"Prediction failed: {results.get('error', 'Unknown error')}")
            else:
                pred_ec = results["predicted_ec"]
                ec_meta = EC_METADATA[pred_ec]
                conf_pct = results["confidence"] * 100

                st.markdown("---")
                st.markdown("#### 2. Prediction Results")

                # Prominent Result Callout Card
                st.markdown(
                    f"""
                    <div class="pred-callout">
                        <div class="pred-badge">Predicted Primary Functional Class</div>
                        <div class="pred-heading">{ec_meta['title']}</div>
                        <p class="pred-detail"><strong>Catalyzed Mechanism:</strong> {ec_meta['reaction']}</p>
                        <p class="pred-detail" style="margin-top: 0.35rem;">
                            <strong>Model Confidence Score:</strong> {conf_pct:.2f}% |
                            <strong>Model Architecture:</strong> {selected_model} |
                            <strong>Sequence Length:</strong> {results['sequence_length']} aa
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    "ℹ️ *Scientific note: This model predicts the high-level Enzyme Commission class (EC 1–6). "
                    "Reported confidence reflects the calibrated posterior probability distribution from the meta-learner.*"
                )

                # Two-column layout: Class Probabilities & Sequence Diagnostics
                col_left, col_right = st.columns([3, 2])

                with col_left:
                    st.markdown("##### Class Probability Distribution")

                    prob_data = []
                    for ec_num in range(1, 7):
                        p = results["probabilities"][ec_num] * 100
                        prob_data.append({
                            "EC Class": f"EC {ec_num} — {EC_METADATA[ec_num]['name']}",
                            "Probability (%)": p,
                            "Score": p / 100.0,
                        })
                    df_probs = pd.DataFrame(prob_data)

                    # Streamlit native progress/table display with clean styling
                    for row in prob_data:
                        is_winner = row["EC Class"].startswith(f"EC {pred_ec}")
                        label_prefix = "⭐ " if is_winner else "   "
                        c1, c2, c3 = st.columns([3, 5, 2])
                        with c1:
                            st.markdown(f"**{label_prefix}{row['EC Class']}**")
                        with c2:
                            st.progress(min(max(row["Score"], 0.0), 1.0))
                        with c3:
                            st.markdown(f"`{row['Probability (%)']:.2f}%`")

                with col_right:
                    st.markdown("##### Sequence & Model Diagnostics")
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">ACTIVE INFERENCE MODEL</div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-top: 0.2rem;">{selected_model}</div>
                            <div style="font-size: 0.82rem; color: #475569; margin-top: 0.3rem;">{results['model_description']}</div>
                            <hr style="margin: 0.75rem 0; border: none; border-top: 1px solid #e2e8f0;">
                            <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">BASE MODEL AGREEMENT</div>
                            <div style="font-size: 0.88rem; color: #1e293b; margin-top: 0.3rem;">
                                • Random Forest: <strong>EC {results['base_model_predictions']['Random Forest']}</strong> ({EC_METADATA[results['base_model_predictions']['Random Forest']]['name']})<br>
                                • LightGBM: <strong>EC {results['base_model_predictions']['LightGBM']}</strong> ({EC_METADATA[results['base_model_predictions']['LightGBM']]['name']})<br>
                                • Linear SVM: <strong>EC {results['base_model_predictions']['Linear SVM']}</strong> ({EC_METADATA[results['base_model_predictions']['Linear SVM']]['name']})
                            </div>
                            <hr style="margin: 0.75rem 0; border: none; border-top: 1px solid #e2e8f0;">
                            <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">TOP 5 RESIDUES</div>
                            <div style="font-size: 0.88rem; color: #1e293b; margin-top: 0.3rem;">
                                {' | '.join(f"<strong>{item['amino_acid']}</strong>: {item['count']} ({item['frequency']*100:.1f}%)" for item in results['top_amino_acids'])}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Collapsible Sequence Details
                with st.expander("🔬 View Processed Amino-Acid Sequence"):
                    st.text(results["cleaned_sequence"])
                    st.caption(f"Total sequence length: {results['sequence_length']} standard amino acids.")

# Footer
st.markdown("---")
col_f1, col_f2 = st.columns([2, 1])
with col_f1:
    st.caption("Enzyme Function Classification (EC 1–6) | Machine Learning Pipeline & Interactive Inference Demo")
with col_f2:
    st.caption("Reproducible ML • SwissProt-EC Benchmark")
