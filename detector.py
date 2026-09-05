"""
detector.py
Pure Python rule-based fraud detection engine for ShieldAgent.
Evaluates transaction windows against threshold rules for velocity spikes,
off-hours clusters, and card testing pattern detection.
"""

import json
import os
from datetime import datetime

# =====================================================================
# THRESHOLD CONSTANTS (Configurable for tuning)
# =====================================================================

# Rule 1: Velocity Spike - 10+ transactions within a 5-minute (300s) window
VELOCITY_WINDOW_SECONDS = 300
VELOCITY_TX_COUNT_THRESHOLD = 10

# Rule 2: Off-Hours Cluster - 4+ transactions between 01:00 AM and 04:00 AM
OFF_HOURS_START = 1
OFF_HOURS_END = 4
OFF_HOURS_TX_COUNT_THRESHOLD = 4

# Rule 3: Card Testing - 5+ transactions with small near-identical amounts (<= $5.00)
CARD_TESTING_MAX_AMOUNT = 5.00
CARD_TESTING_TX_COUNT_THRESHOLD = 5
CARD_TESTING_VARIANCE = 0.10


def parse_timestamp(ts_str):
    """Parses standard ISO/datetime string into datetime object."""
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.fromisoformat(ts_str)


def classify_window(window, 
                    vel_win_sec=VELOCITY_WINDOW_SECONDS,
                    vel_count_thresh=VELOCITY_TX_COUNT_THRESHOLD,
                    off_hours_start=OFF_HOURS_START,
                    off_hours_end=OFF_HOURS_END,
                    off_hours_thresh=OFF_HOURS_TX_COUNT_THRESHOLD,
                    card_max_amt=CARD_TESTING_MAX_AMOUNT,
                    card_count_thresh=CARD_TESTING_TX_COUNT_THRESHOLD):
    """
    Classifies a transaction window.
    
    Returns:
        dict: {
            "predicted_label": 0 or 1,
            "pattern_type": None or "velocity_spike" | "off_hours_cluster" | "card_testing",
            "facts": dict of specific numerical triggers
        }
    """
    transactions = window.get("transactions", [])
    if not transactions:
        return {
            "predicted_label": 0,
            "pattern_type": None,
            "facts": {"total_tx_count": 0, "reason": "No transactions in window"}
        }

    parsed_txs = []
    for tx in transactions:
        parsed_txs.append({
            "amount": float(tx["amount"]),
            "dt": parse_timestamp(tx["timestamp"])
        })

    parsed_txs.sort(key=lambda x: x["dt"])

    # -----------------------------------------------------------------
    # Check 1: Velocity Spike (Max transactions in any VELOCITY_WINDOW_SECONDS)
    # -----------------------------------------------------------------
    max_vel_count = 0
    max_vel_span_sec = 0

    for i in range(len(parsed_txs)):
        start_time = parsed_txs[i]["dt"]
        end_limit = start_time.timestamp() + vel_win_sec
        
        cluster = [t for t in parsed_txs[i:] if t["dt"].timestamp() <= end_limit]
        count = len(cluster)
        
        if count > max_vel_count:
            max_vel_count = count
            if count > 1:
                max_vel_span_sec = int((cluster[-1]["dt"] - cluster[0]["dt"]).total_seconds())
            else:
                max_vel_span_sec = 0

    if max_vel_count >= vel_count_thresh:
        return {
            "predicted_label": 1,
            "pattern_type": "velocity_spike",
            "facts": {
                "max_tx_in_window": max_vel_count,
                "window_seconds": max_vel_span_sec,
                "time_frame_seconds": vel_win_sec,
                "threshold_tx_count": vel_count_thresh,
                "total_tx_in_window": len(parsed_txs)
            }
        }

    # -----------------------------------------------------------------
    # Check 2: Off-Hours Cluster (Transactions between 1:00 AM and 4:00 AM)
    # -----------------------------------------------------------------
    off_hours_txs = [t for t in parsed_txs if off_hours_start <= t["dt"].hour < off_hours_end]
    off_hours_count = len(off_hours_txs)

    if off_hours_count >= off_hours_thresh:
        return {
            "predicted_label": 1,
            "pattern_type": "off_hours_cluster",
            "facts": {
                "off_hours_tx_count": off_hours_count,
                "off_hours_window": f"{off_hours_start:02d}:00 - {off_hours_end:02d}:00",
                "threshold_off_hours_count": off_hours_thresh,
                "total_tx_in_window": len(parsed_txs)
            }
        }

    # -----------------------------------------------------------------
    # Check 3: Card Testing (Near-identical small amounts)
    # -----------------------------------------------------------------
    small_txs = [t for t in parsed_txs if t["amount"] <= card_max_amt]
    max_card_testing_cluster = 0
    cluster_avg_amt = 0.0

    if small_txs:
        # Group small transactions by close amounts
        for base_tx in small_txs:
            base_amt = base_tx["amount"]
            matching = [t for t in small_txs if abs(t["amount"] - base_amt) <= CARD_TESTING_VARIANCE]
            if len(matching) > max_card_testing_cluster:
                max_card_testing_cluster = len(matching)
                cluster_avg_amt = round(sum(t["amount"] for t in matching) / len(matching), 2)

    if max_card_testing_cluster >= card_count_thresh:
        return {
            "predicted_label": 1,
            "pattern_type": "card_testing",
            "facts": {
                "card_testing_tx_count": max_card_testing_cluster,
                "average_small_amount": cluster_avg_amt,
                "max_amount_threshold": card_max_amt,
                "threshold_count": card_count_thresh,
                "total_tx_in_window": len(parsed_txs)
            }
        }

    # -----------------------------------------------------------------
    # Normal Window
    # -----------------------------------------------------------------
    return {
        "predicted_label": 0,
        "pattern_type": None,
        "facts": {
            "total_tx_count": len(parsed_txs),
            "max_5min_velocity": max_vel_count,
            "off_hours_tx_count": off_hours_count,
            "card_testing_tx_count": max_card_testing_cluster
        }
    }


def tune_thresholds(train_data):
    """
    Evaluates candidate thresholds on training data to print precision and recall.
    Does NOT use test.json.
    """
    candidates = [
        {"name": "Strict (High Precision)", "vel": 12, "off": 5, "card": 6},
        {"name": "Balanced (Default)", "vel": 10, "off": 4, "card": 5},
        {"name": "Aggressive (High Recall)", "vel": 8, "off": 3, "card": 4},
    ]

    print("=== THRESHOLD TUNING REPORT (Training Data Only) ===")
    print(f"Total Train Windows: {len(train_data)}\n")

    for cand in candidates:
        tp = fp = tn = fn = 0
        for window in train_data:
            res = classify_window(
                window,
                vel_count_thresh=cand["vel"],
                off_hours_thresh=cand["off"],
                card_count_thresh=cand["card"]
            )
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

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"Candidate: {cand['name']}")
        print(f"  Thresholds -> Velocity: {cand['vel']} tx, Off-Hours: {cand['off']} tx, Card-Test: {cand['card']} tx")
        print(f"  Confusion  -> TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"  Metrics    -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")


if __name__ == "__main__":
    train_path = os.path.join("data", "train.json")
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            train_windows = json.load(f)
        tune_thresholds(train_windows)
    else:
        print("Training data not found. Run generate_data.py first.")
