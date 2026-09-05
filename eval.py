"""
eval.py
Evaluates ShieldAgent detector on data/test.json ONLY.
Calculates performance metrics (Precision, Recall, F1, Confusion Matrix)
and estimates operational/fraud costs. Writes full report to results.txt and stdout.
"""

import json
import os
import sys
from detector import classify_window

TEST_FILE = os.path.join("data", "test.json")
RESULTS_FILE = "results.txt"


def run_evaluation():
    if not os.path.exists(TEST_FILE):
        print(f"Error: {TEST_FILE} not found. Run generate_data.py first.")
        sys.exit(1)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_windows = json.load(f)

    tp = fp = tn = fn = 0
    fn_window_amounts = []
    fn_total_fraud_volume = 0.0

    for window in test_windows:
        res = classify_window(window)
        pred = res["predicted_label"]
        actual = window["label"]

        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 0:
            tn += 1
        elif pred == 0 and actual == 1:
            fn += 1
            # Track transaction amounts in missed fraud window
            amounts = [float(tx["amount"]) for tx in window["transactions"]]
            fn_total_fraud_volume += sum(amounts)
            if amounts:
                fn_window_amounts.append(sum(amounts) / len(amounts))

    total_test = len(test_windows)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # -----------------------------------------------------------------
    # COST ESTIMATION CALCULATIONS & ASSUMPTIONS
    # -----------------------------------------------------------------
    # Assumption 1: False Positive Cost = FP count * 15 minutes of analyst review time
    # (Valued at $15 per FP assuming analyst cost rate of $60/hr or 15 analyst minutes)
    ANALYST_MINUTES_PER_FP = 15
    COST_PER_FP_DOLLARS = 15.00
    fp_analyst_cost = fp * COST_PER_FP_DOLLARS

    # Assumption 2: False Negative Cost = Total dollar volume of uncaught fraud transactions
    avg_fn_tx_amount = (sum(fn_window_amounts) / len(fn_window_amounts)) if fn_window_amounts else 0.0
    fn_fraud_cost = fn_total_fraud_volume

    total_estimated_cost = fp_analyst_cost + fn_fraud_cost

    # Build report text
    report = []
    report.append("======================================================")
    report.append("          SHIELDAGENT EVALUATION REPORT               ")
    report.append("======================================================")
    report.append(f"Evaluated Test Dataset : {TEST_FILE}")
    report.append(f"Total Test Windows     : {total_test}\n")
    
    report.append("--- PERFORMANCE METRICS ---")
    report.append(f"Precision : {precision:.4f} ({precision * 100:.1f}%)")
    report.append(f"Recall    : {recall:.4f} ({recall * 100:.1f}%)")
    report.append(f"F1 Score  : {f1:.4f}")
    report.append("")
    
    report.append("--- CONFUSION MATRIX ---")
    report.append(f"True Positives  (TP) : {tp:2d} (Correctly flagged fraud)")
    report.append(f"False Positives (FP) : {fp:2d} (False alarms on normal windows)")
    report.append(f"True Negatives  (TN) : {tn:2d} (Correctly passed normal windows)")
    report.append(f"False Negatives (FN) : {fn:2d} (Missed fraud windows)")
    report.append("")

    report.append("--- ESTIMATED OPERATIONAL & FRAUD COST ANALYSIS ---")
    report.append(f"Assumptions:")
    report.append(f"  1. False Positive Cost: 15 minutes of analyst review time per FP (${COST_PER_FP_DOLLARS:.2f}/FP)")
    report.append(f"  2. False Negative Cost: Actual dollar volume of uncaught fraud (Avg tx amt in FN: ${avg_fn_tx_amount:.2f})")
    report.append(f"Calculated Costs:")
    report.append(f"  - False Positive Cost (FP * $15) : ${fp_analyst_cost:.2f} ({fp * ANALYST_MINUTES_PER_FP} analyst mins)")
    report.append(f"  - False Negative Cost (Uncaught) : ${fn_fraud_cost:.2f}")
    report.append(f"  - Total Estimated Financial Cost  : ${total_estimated_cost:.2f}")
    report.append("======================================================")

    output_text = "\n".join(report)

    # Print to stdout
    print(output_text)

    # Save to results.txt
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print(f"\nSaved evaluation summary to {RESULTS_FILE}")


if __name__ == "__main__":
    run_evaluation()
