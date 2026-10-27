"""RadInsight AI — Grounded Radiology Report Generation with Feature-Verified Findings

for Chest X-Ray Interpretation.

Streamlit Academic Research Dashboard.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config.settings import (
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_WEIGHTS_INFO_PATH,
    DATA_DIR,
    DEFAULT_DECISION_THRESHOLDS,
    FINDING_DESCRIPTIONS,
    SUPPORTED_FINDINGS,
    THEME_COLORS,
    UNCERTAINTY_MARGIN,
)
from evaluation import (
    compute_multilabel_metrics,
    create_confusion_matrix_figure,
    create_roc_curves_figure,
    evaluate_test_dataset,
)
from finding_verification import (
    FindingVerificationEngine,
    STATUS_BORDERLINE_UNCERTAIN,
    STATUS_VERIFIED_ABSENT,
    STATUS_VERIFIED_PRESENT,
)
from image_processing import (
    ImageValidationError,
    preprocess_chest_xray,
    validate_image_file,
)
from model_inference import ChestXRayClassifier
from report_generation import RadiologyReportGenerator

st.set_page_config(
    page_title="RadInsight AI — Radiology Research",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

                                     
CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Top banner and general spacing */
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }}

    /* Headings */
    h1, h2, h3, h4 {{
        color: {THEME_COLORS["primary"]};
        font-weight: 600;
        letter-spacing: -0.02em;
    }}

    /* Card styling */
    .metric-card {{
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}

    .section-title {{
        color: {THEME_COLORS["primary"]};
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 6px;
        margin-top: 14px;
        margin-bottom: 12px;
    }}

    /* Badges */
    .badge-present {{
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }}
    .badge-borderline {{
        background-color: #fef3c7;
        color: #92400e;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }}
    .badge-absent {{
        background-color: #f1f5f9;
        color: #475569;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
    }}

    /* Report container styling */
    .report-box {{
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 20px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.92rem;
        line-height: 1.5;
        color: #0f172a;
        white-space: pre-wrap;
    }}

    /* Clinical Disclaimer Box */
    .disclaimer-banner {{
        background-color: #f8fafc;
        border-left: 4px solid {THEME_COLORS["secondary"]};
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #334155;
        margin-top: 20px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource
def load_classifier():
    """Cache the classifier instance across user interactions."""
    return ChestXRayClassifier.get_instance()

@st.cache_resource
def load_report_generator():
    """Cache report generator instance."""
    return RadiologyReportGenerator()

def render_sidebar():
    """Render compact sidebar navigation and status summary."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding-bottom: 8px;">
                <h3 style="margin: 0; color: {THEME_COLORS['primary']};">🫁 RadInsight AI</h3>
                <p style="margin: 0; font-size: 0.78rem; color: #64748b;">
                    Grounded Radiology Report Generation
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        page = st.radio(
            "Navigation",
            options=["Analysis", "Evaluation", "About"],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown(
            f"""
            <div style="font-size: 0.78rem; color: #475569;">
                <b>System Status</b><br>
                • Architecture: MobileNetV2 (Keras)<br>
                • Report Gen: Grounded Clinical NLP<br>
                • Evaluation: Scikit-Learn Multilabel<br>
                • Mode: Academic Prototype
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        pdf_path = Path(__file__).resolve().parent / "RadInsightAI_Complete_Codebase.pdf"
        if pdf_path.exists():
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📥 Download Codebase PDF",
                    data=pdf_file.read(),
                    file_name="RadInsightAI_Complete_Codebase.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

       
        with st.expander("⚙️ Decision Thresholds", expanded=False):
            st.caption("Adjust per-finding classification cutoffs:")
            user_thresholds = {}
            for finding in SUPPORTED_FINDINGS:
                default_val = DEFAULT_DECISION_THRESHOLDS.get(finding, 0.50)
                user_thresholds[finding] = st.slider(
                    f"{finding}",
                    min_value=0.10,
                    max_value=0.90,
                    value=float(default_val),
                    step=0.05,
                    key=f"thresh_{finding}",
                )
            st.session_state["thresholds"] = user_thresholds

        return page

def page_analysis(classifier: ChestXRayClassifier, report_gen: RadiologyReportGenerator):
    """Analysis Page: Upload, inference, verification, report generation."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1rem;">
            <h2 style="margin: 0; color: {THEME_COLORS['primary']};">Chest Radiograph Interpretation & Report Synthesis</h2>
            <p style="margin: 4px 0 0 0; color: #475569; font-size: 0.95rem;">
                Upload a chest X-ray to predict findings, quantify decision uncertainty, and generate a grounded radiology report.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_preview = st.columns([1.2, 1.0])

    with col_upload:
        st.markdown('<div class="section-title">1. Radiograph Input</div>', unsafe_allow_html=True)
        input_mode = st.radio(
            "Choose Input Source",
            options=["Upload Local File", "Select from Sample Archive"],
            horizontal=True,
        )

        selected_image_input = None

        if input_mode == "Upload Local File":
            uploaded_file = st.file_uploader(
                "Upload Chest Radiograph (PNG or JPEG)",
                type=["png", "jpg", "jpeg"],
                help="High-quality frontal chest radiographs (PA or AP view)",
            )
            if uploaded_file is not None:
                selected_image_input = uploaded_file
        else:
            sample_dir = DATA_DIR / "sample_test"
            sample_files = []
            if sample_dir.exists():
                sample_files = sorted([f.name for f in sample_dir.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])

            if sample_files:
                sample_choice = st.selectbox(
                    "Choose Sample Radiograph Case",
                    options=sample_files,
                    format_func=lambda x: f"Case: {x.replace('_', ' ').replace('.png', '').replace('.jpg', '')}",
                )
                selected_image_input = str(sample_dir / sample_choice)
            else:
                st.warning("No sample files found in 'data/sample_test/'. Run create_sample_data.py to generate samples.")

        enable_clahe = st.checkbox(
            "Apply Radiological Contrast Enhancement (CLAHE)",
            value=True,
            help="Applies Contrast Limited Adaptive Histogram Equalization to optimize lung field visualization.",
        )

        analyze_button = st.button(
            "⚡ Run Complete Pipeline",
            type="primary",
            use_container_width=True,
            disabled=selected_image_input is None,
        )

    with col_preview:
        st.markdown('<div class="section-title">2. Radiograph Preview</div>', unsafe_allow_html=True)
        if selected_image_input is not None:
            try:
                is_valid, val_msg, meta = validate_image_file(selected_image_input)
                if not is_valid:
                    st.error(f"Image Validation Failed: {val_msg}")
                else:
                    _, display_pil, _ = preprocess_chest_xray(
                        selected_image_input, enable_clahe=enable_clahe
                    )
                    st.image(display_pil, caption="Processed Radiograph (224x224 RGB)", use_container_width=True)
                    st.caption(
                        f"Original Dimensions: {meta['width']}x{meta['height']} | "
                        f"Format: {meta['format']} | Enhanced (CLAHE): {enable_clahe}"
                    )
            except Exception as e:
                st.error(f"Error loading preview: {e}")
        else:
            st.info("Awaiting radiograph input. Upload a file or select a sample case above.")

    
    if analyze_button and selected_image_input is not None:
        st.divider()
        with st.spinner("Executing pipeline: Preprocessing → MobileNetV2 Inference → Finding Verification → Report Generation..."):
            try:
                
                tensor_batch, display_pil, prep_meta = preprocess_chest_xray(
                    selected_image_input, enable_clahe=enable_clahe
                )

                
                pred_result = classifier.predict(tensor_batch)
                raw_scores = pred_result["raw_scores"]
                inference_time = pred_result["inference_time_ms"]
                model_status = pred_result["model_status"]

               
                active_thresholds = st.session_state.get("thresholds", DEFAULT_DECISION_THRESHOLDS)
                verifier = FindingVerificationEngine(thresholds=active_thresholds)
                verification_result = verifier.verify_findings(raw_scores)

               
                report_result = report_gen.generate_report(verification_result)

                
                st.markdown('<div class="section-title">3. Model Predictions & Feature Verification</div>', unsafe_allow_html=True)

                col_metrics_status, col_findings_breakdown = st.columns([1, 1.6])

                with col_metrics_status:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <span style="font-size: 0.85rem; color: #64748b; font-weight: 600;">EXECUTION TELEMETRY</span><br>
                            <span style="font-size: 1.1rem; font-weight: 600; color: {THEME_COLORS['primary']};">
                                {inference_time:.1f} ms Latency
                            </span><br>
                            <span style="font-size: 0.8rem; color: #475569;">
                                Model State: <b>{model_status}</b><br>
                                Average Uncertainty: <b>{verification_result['average_uncertainty']:.2f}</b>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    
                    if verification_result["detected_conflicts"]:
                        for conflict in verification_result["detected_conflicts"]:
                            st.warning(f"⚠️ {conflict}")

                   
                    st.write("**Verified Findings Summary:**")
                    if verification_result["verified_present"]:
                        for vp in verification_result["verified_present"]:
                            st.markdown(f"<span class='badge-present'>+ {vp}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='badge-absent'>No acute pathology verified</span>", unsafe_allow_html=True)

                    if verification_result["borderline_findings"]:
                        st.write("**Borderline / Equivocal Observations:**")
                        for bf in verification_result["borderline_findings"]:
                            st.markdown(f"<span class='badge-borderline'>? {bf}</span>", unsafe_allow_html=True)

                with col_findings_breakdown:
                    st.write("**Predicted Finding Probabilities vs Decision Thresholds:**")
                    chart_data = []
                    details = verification_result["finding_details"]
                    for f in SUPPORTED_FINDINGS:
                        f_info = details.get(f, {})
                        chart_data.append({
                            "Finding": f,
                            "Score": f_info.get("raw_score", 0.0),
                            "Threshold": f_info.get("threshold", 0.50),
                            "Status": f_info.get("status", "UNKNOWN"),
                        })
                    df_chart = pd.DataFrame(chart_data)

                  
                    st.dataframe(
                        df_chart[["Finding", "Score", "Threshold", "Status"]],
                        use_container_width=True,
                        hide_index=True,
                    )

                
                st.markdown('<div class="section-title">4. Grounded Radiology Report</div>', unsafe_allow_html=True)

                st.caption(
                    f"Report Synthesis Mode: **{report_result['generation_mode']}** | "
                    f"Status: {report_result['generator_notes']}"
                )

                col_find, col_imp = st.columns(2)

                with col_find:
                    st.markdown("#### FINDINGS")
                    st.markdown(
                        f"<div class='report-box'>{report_result['findings']}</div>",
                        unsafe_allow_html=True,
                    )

                with col_imp:
                    st.markdown("#### IMPRESSION")
                    st.markdown(
                        f"<div class='report-box'>{report_result['impression']}</div>",
                        unsafe_allow_html=True,
                    )

               
                st.markdown('<div class="section-title">5. Traceability & Export</div>', unsafe_allow_html=True)
                with st.expander("🔍 View Finding-to-Report Traceability (Grounded Provenance)", expanded=False):
                    st.json(report_result["traceability_map"])

               
                full_report_text = (
                    "RADINSIGHT AI - CHEST RADIOGRAPHY REPORT\n"
                    "=========================================\n"
                    f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Model Architecture: MobileNetV2 (Multi-label)\n"
                    f"Report Mode: {report_result['generation_mode']}\n\n"
                    "FINDINGS:\n"
                    f"{report_result['findings']}\n\n"
                    "IMPRESSION:\n"
                    f"{report_result['impression']}\n\n"
                    "DISCLAIMER:\n"
                    f"{report_result['disclaimer']}\n"
                )

                col_d1, col_d2 = st.columns([1, 1])
                with col_d1:
                    st.download_button(
                        label="📄 Download Report as Text (.txt)",
                        data=full_report_text,
                        file_name="radinsight_report.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with col_d2:
                    export_json_payload = {
                        "telemetry": {
                            "inference_time_ms": inference_time,
                            "model_status": model_status,
                        },
                        "verification": verification_result,
                        "report": report_result,
                    }
                    st.download_button(
                        label="💾 Export Pipeline Payload (.json)",
                        data=json.dumps(export_json_payload, indent=2),
                        file_name="radinsight_analysis.json",
                        mime="application/json",
                        use_container_width=True,
                    )

            except ImageValidationError as ive:
                st.error(f"Image Preprocessing Error: {str(ive)}")
            except Exception as ex:
                st.error(f"Pipeline Execution Failed: {str(ex)}")

    st.markdown(
        f"""
        <div class="disclaimer-banner">
            <b>ACADEMIC RESEARCH PROTOTYPE DISCLAIMER:</b><br>
            RadInsight AI is designed exclusively for educational, engineering, and machine learning research.
            It does not constitute medical advice or a certified in-vitro medical device. Model outputs,
            uncertainty scores, and generated reports must not be used for clinical decision-making or primary diagnostic triage.
        </div>
        """,
        unsafe_allow_html=True,
    )

def page_evaluation(classifier: ChestXRayClassifier):
    """Evaluation Page: Genuine metrics, confusion matrices, ROC curves."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1rem;">
            <h2 style="margin: 0; color: {THEME_COLORS['primary']};">Empirical Evaluation & Performance Verification</h2>
            <p style="margin: 4px 0 0 0; color: #475569; font-size: 0.95rem;">
                Quantitative evaluation on verified ground-truth test data. No metrics are fabricated when unmeasured.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    manifest_path = DATA_DIR / "ground_truth.json"

    col_ctrl, col_info = st.columns([1.2, 1.0])
    with col_ctrl:
        st.write(f"**Test Manifest File:** `{manifest_path.name}`")
        if manifest_path.exists():
            st.success("Ground-truth test manifest found on disk.")
        else:
            st.error("Ground-truth manifest missing.")

        run_eval_btn = st.button(
            "📊 Run Empirical Evaluation",
            type="primary",
            disabled=not manifest_path.exists(),
        )

    with col_info:
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Evaluation Standard:</b><br>
                • Multi-label Scikit-Learn framework<br>
                • Evaluates Subset Accuracy, Hamming Loss, Macro F1, and ROC-AUC<br>
                • Per-class confusion matrix breakdown
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run_eval_btn or "eval_results" in st.session_state:
        if run_eval_btn:
            with st.spinner("Evaluating labeled test set through MobileNetV2..."):
                active_thresholds = st.session_state.get("thresholds", DEFAULT_DECISION_THRESHOLDS)
                eval_data = evaluate_test_dataset(
                    classifier, manifest_path=manifest_path, thresholds=active_thresholds
                )
                st.session_state["eval_results"] = eval_data

        eval_res = st.session_state.get("eval_results", {})

        if not eval_res.get("evaluated", False):
            st.warning(eval_res.get("message", "Evaluation has not yet been executed."))
        else:
            st.divider()
            st.markdown('<div class="section-title">Global Multi-Label Metrics</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric(
                    "Subset Accuracy",
                    f"{eval_res['subset_accuracy'] * 100:.1f}%",
                    help="Exact match across all binary labels simultaneously.",
                )
            with m2:
                st.metric(
                    "Hamming Loss",
                    f"{eval_res['hamming_loss']:.4f}",
                    help="Fraction of incorrect individual label predictions (lower is better).",
                )
            with m3:
                st.metric(
                    "Macro F1-Score",
                    f"{eval_res['macro_f1']:.3f}",
                    help="Unweighted average F1 across all supported finding classes.",
                )
            with m4:
                auc_display = f"{eval_res['macro_roc_auc']:.3f}" if eval_res["macro_roc_auc"] else "N/A"
                st.metric(
                    "Macro ROC-AUC",
                    auc_display,
                    help="Area under the ROC curve averaged across valid classes.",
                )

            st.markdown('<div class="section-title">Per-Finding Breakdown</div>', unsafe_allow_html=True)
            per_class = eval_res["per_class"]
            rows = []
            for c_name, c_metrics in per_class.items():
                auc_val = f"{c_metrics['roc_auc']:.3f}" if c_metrics["roc_auc"] is not None else "N/A"
                rows.append({
                    "Finding Class": c_name,
                    "Accuracy": f"{c_metrics['accuracy']:.3f}",
                    "Precision": f"{c_metrics['precision']:.3f}",
                    "Recall": f"{c_metrics['recall']:.3f}",
                    "F1-Score": f"{c_metrics['f1_score']:.3f}",
                    "ROC-AUC": auc_val,
                    "Support": c_metrics["support"],
                    "TP": c_metrics["true_positives"],
                    "FP": c_metrics["false_positives"],
                    "TN": c_metrics["true_negatives"],
                    "FN": c_metrics["false_negatives"],
                })
            df_per_class = pd.DataFrame(rows)
            st.dataframe(df_per_class, use_container_width=True, hide_index=True)

           
            st.markdown('<div class="section-title">Class-Specific Confusion Matrices</div>', unsafe_allow_html=True)
            cm_fig = create_confusion_matrix_figure(per_class)
            st.pyplot(cm_fig)

                        
            st.markdown('<div class="section-title">Receiver Operating Characteristic (ROC)</div>', unsafe_allow_html=True)
            roc_fig = create_roc_curves_figure(
                eval_res["y_true"], eval_res["y_prob"], list(SUPPORTED_FINDINGS)
            )
            if roc_fig:
                st.pyplot(roc_fig)
            else:
                st.info("ROC curves require at least one positive and one negative sample per class in the test split.")

    else:
        st.info("Click 'Run Empirical Evaluation' above to compute metrics on the verified test set.")

def page_about():
    """About Page: Architecture, datasets, tech stack, ethics."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <h2 style="margin: 0; color: {THEME_COLORS['primary']};">About RadInsight AI</h2>
            <p style="margin: 4px 0 0 0; color: #475569; font-size: 0.95rem;">
                Grounded Radiology Report Generation with Feature-Verified Findings for Chest X-Ray Interpretation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4 style="margin-top:0; color: {THEME_COLORS['primary']};">1. Project Objective</h4>
                <p style="font-size: 0.88rem; color: #334155;">
                    RadInsight AI is a Semester 5 computer vision and machine learning academic prototype.
                    It demonstrates a safe, grounded clinical pipeline for automated chest radiograph interpretation.
                    Instead of allowing a language model to hallucinate diagnoses from unconstrained images,
                    RadInsight AI explicitly chains deep feature extraction, calibrated uncertainty handling,
                    feature verification, and grounded report synthesis.
                </p>
            </div>

            <div class="metric-card">
                <h4 style="margin-top:0; color: {THEME_COLORS['primary']};">2. System Architecture</h4>
                <p style="font-size: 0.88rem; color: #334155;">
                    <b>Pipeline:</b><br>
                    1. <b>Radiograph Validation & CLAHE Preprocessing:</b> Verifies image dimensions, channel depth, and applies adaptive histogram equalization.<br>
                    2. <b>MobileNetV2 Classifier:</b> Extracts multi-label finding probabilities using a convolutional backbone and sigmoid activation.<br>
                    3. <b>Finding Verification Engine:</b> Evaluates raw probabilities against calibrated decision thresholds and flags borderline uncertainty.<br>
                    4. <b>Grounded Report Generator:</b> Synthesizes structured ACR-standard FINDINGS and IMPRESSION sections strictly anchored in verified features.<br>
                    5. <b>Multilabel Evaluation:</b> Measures empirical performance using Scikit-Learn without synthetic result inflation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4 style="margin-top:0; color: {THEME_COLORS['primary']};">3. Dataset & Model Standards</h4>
                <p style="font-size: 0.88rem; color: #334155;">
                    • <b>Public Reference Dataset:</b> Documented on NIH ChestX-ray14 (Wang et al., NIH Clinical Center) and CheXpert.<br>
                    • <b>Classifier Architecture:</b> MobileNetV2 with transfer learning and a dedicated multi-label dense classification head.<br>
                    • <b>Report Generator:</b> Hugging Face Transformers integration with transparent fallback to Feature-Grounded Clinical Synthesis.<br>
                    • <b>Supported Taxonomies:</b> 8 core findings: Atelectasis, Cardiomegaly, Consolidation, Edema, Infiltration, Pleural Effusion, Pneumothorax, and No Finding.
                </p>
            </div>

            <div class="metric-card">
                <h4 style="margin-top:0; color: {THEME_COLORS['primary']};">4. Ethical Considerations & Limitations</h4>
                <p style="font-size: 0.88rem; color: #334155;">
                    • <b>Not a Diagnostic Device:</b> This system is strictly an academic research prototype.<br>
                    • <b>Threshold Sensitivity:</b> Simple decision thresholds do not substitute clinical verification.<br>
                    • <b>Distribution Shift:</b> Performance on external radiographic hardware or differing projections (AP vs PA) may vary.<br>
                    • <b>Honest Transparency:</b> Uninitialized or unmeasured metrics are never faked.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

def main():
    """Main application entry point."""
    page = render_sidebar()
    classifier = load_classifier()
    report_gen = load_report_generator()

    if page == "Analysis":
        page_analysis(classifier, report_gen)
    elif page == "Evaluation":
        page_evaluation(classifier)
    elif page == "About":
        page_about()

if __name__ == "__main__":
    main()
