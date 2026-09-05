"""
app.py
Streamlit Dashboard for ShieldAgent Fraud Detection MVP.
Interactive window inspection, real-time rule classification,
Gemini AI explanations, and test set evaluation metric reports.

Self-starts automatically when run via `python app.py`.
"""

import json
import os
import sys
import subprocess
import pandas as pd
import streamlit as st

from detector import classify_window
from explain import explain_flag

DATA_TEST_PATH = os.path.join("data", "test.json")
RESULTS_PATH = "results.txt"


def main():
    st.set_page_config(
        page_title="ShieldAgent - Real-Time Fraud Detection MVP",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .badge-fraud {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .badge-normal {
            background-color: #DCFCE7;
            color: #166534;
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .explanation-box {
            background-color: #F8FAFC;
            border-left: 4px solid #3B82F6;
            padding: 1rem 1.25rem;
            border-radius: 0.375rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🛡️ ShieldAgent Fraud Detection MVP</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Transaction Window Inspection & AI-Powered Explainability</div>', unsafe_allow_html=True)

    # Load test windows
    if not os.path.exists(DATA_TEST_PATH):
        st.error(f"Test dataset not found at `{DATA_TEST_PATH}`. Please run `python generate_data.py` first.")
        st.stop()

    with open(DATA_TEST_PATH, "r", encoding="utf-8") as f:
        test_windows = json.load(f)

    # Sidebar Controls
    st.sidebar.header("🔍 Window Selector")
    
    # Format dropdown options
    window_options = {}
    for idx, win in enumerate(test_windows):
        lbl = "🔴 FRAUD" if win["label"] == 1 else "🟢 Normal"
        pattern = f" ({win['pattern_type']})" if win.get("pattern_type") else ""
        label_str = f"{win['window_id']} — {lbl}{pattern}"
        window_options[label_str] = idx

    selected_label = st.sidebar.selectbox("Choose Transaction Window from test.json:", list(window_options.keys()))
    selected_idx = window_options[selected_label]
    selected_window = test_windows[selected_idx]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Quick Stats")
    st.sidebar.metric("Total Test Windows", len(test_windows))
    st.sidebar.metric("Selected Window ID", selected_window["window_id"])
    st.sidebar.metric("Transactions Count", len(selected_window["transactions"]))

    # Main Tabs
    tab1, tab2 = st.tabs(["🔎 Window Inspection & AI Analysis", "📊 Overall Model Evaluation & Cost Report"])

    with tab1:
        st.subheader(f"Window Analysis: `{selected_window['window_id']}`")
        
        # Perform real-time detection
        det_result = classify_window(selected_window)
        pred_label = det_result["predicted_label"]
        pred_pattern = det_result.get("pattern_type")
        actual_label = selected_window["label"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Merchant ID", selected_window.get("merchant_id", "N/A"))
        with col2:
            gt_text = "FRAUD (1)" if actual_label == 1 else "NORMAL (0)"
            st.metric("Ground Truth Label", gt_text)
        with col3:
            pred_text = "FLAGGED (1)" if pred_label == 1 else "CLEARED (0)"
            st.metric("ShieldAgent Decision", pred_text)
        with col4:
            st.metric("Detected Pattern", pred_pattern if pred_pattern else "None (Normal)")

        # Result Banner
        if pred_label == 1:
            st.error(f"⚠️ **SUSPICIOUS WINDOW DETECTED** — Pattern Identified: `{pred_pattern}`")
        else:
            st.success("✅ **WINDOW CLEARED** — No anomaly rules triggered.")

        # AI Explanation Section
        st.markdown("### 🤖 Natural Language Explanation")
        with st.spinner("Generating explanation..."):
            explanation_text = explain_flag(selected_window, det_result)

        st.markdown(f"""
        <div class="explanation-box">
            <strong>Analysis:</strong> {explanation_text}
        </div>
        """, unsafe_allow_html=True)

        # Trigger Facts Breakdown
        with st.expander("🔍 View Decision Facts & Trigger Metrics"):
            st.json(det_result["facts"])

        # Transactions Table
        st.markdown("### 💳 Window Transactions List")
        txs = selected_window.get("transactions", [])
        if txs:
            df = pd.DataFrame(txs)
            df["amount"] = df["amount"].apply(lambda x: f"${x:.2f}")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No transactions found in this window.")

    with tab2:
        st.subheader("📊 Model Evaluation & Operational Cost Summary")
        st.caption("Results calculated exclusively on `data/test.json` (30% split).")

        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r", encoding="utf-8") as rf:
                results_text = rf.read()
            st.code(results_text, language="text")
        else:
            st.warning("`results.txt` not found. Run `python eval.py` to generate the evaluation report.")


if __name__ == "__main__":
    # Self-launch check for python app.py execution
    is_running_streamlit = False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            is_running_streamlit = True
    except Exception:
        pass

    if is_running_streamlit:
        main()
    else:
        print("Launching ShieldAgent Streamlit Web Dashboard...")
        subprocess.call([sys.executable, "-m", "streamlit", "run", __file__] + sys.argv[1:])
