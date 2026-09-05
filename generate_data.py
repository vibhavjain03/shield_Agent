"""
generate_data.py
Synthesizes realistic payment transaction windows for ShieldAgent MVP.
Splits data into train (70%) and test (30%) sets with fixed random seed.
"""

import json
import os
import random
from datetime import datetime, timedelta

DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
TEST_FILE = os.path.join(DATA_DIR, "test.json")

SEED = 42

def generate_transaction_windows():
    random.seed(SEED)
    merchant_id = "MCH_1001"
    windows = []
    
    base_time = datetime(2026, 9, 1, 0, 0, 0)
    
    # 1. Standard Normal Windows (Label 0: 117 windows)
    for i in range(117):
        win_id = f"win_norm_{i+1:03d}"
        # Day business hours: 8am to 10pm
        day_offset = random.randint(0, 14)
        hour = random.randint(8, 21)
        minute = random.randint(0, 30)
        win_start = base_time + timedelta(days=day_offset, hours=hour, minutes=minute)
        
        tx_count = random.randint(2, 6)
        txs = []
        curr_time = win_start
        for _ in range(tx_count):
            curr_time += timedelta(seconds=random.randint(120, 600))
            amount = round(random.uniform(12.50, 250.00), 2)
            txs.append({
                "amount": amount,
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
        windows.append({
            "window_id": win_id,
            "merchant_id": merchant_id,
            "transactions": txs,
            "label": 0,
            "pattern_type": None
        })
        
    # 2. Borderline Normal Windows (Label 0: 18 windows) - Flash sale / legitimate burst
    for i in range(18):
        win_id = f"win_border_{i+1:03d}"
        day_offset = random.randint(0, 14)
        hour = random.randint(12, 20)
        minute = random.randint(0, 45)
        win_start = base_time + timedelta(days=day_offset, hours=hour, minutes=minute)
        
        # 6-8 txs in 8-12 minutes (resembles spike but legitimate)
        tx_count = random.randint(6, 8)
        txs = []
        curr_time = win_start
        for _ in range(tx_count):
            curr_time += timedelta(seconds=random.randint(60, 100))
            amount = round(random.uniform(25.00, 180.00), 2)
            txs.append({
                "amount": amount,
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
        windows.append({
            "window_id": win_id,
            "merchant_id": merchant_id,
            "transactions": txs,
            "label": 0,
            "pattern_type": None
        })

    # 3. Velocity Spike Anomaly (Label 1: 15 windows)
    for i in range(15):
        win_id = f"win_vel_{i+1:03d}"
        day_offset = random.randint(0, 14)
        hour = random.randint(9, 21)
        minute = random.randint(0, 50)
        win_start = base_time + timedelta(days=day_offset, hours=hour, minutes=minute)
        
        # 11-16 transactions within 150-270 seconds (< 5 minutes)
        tx_count = random.randint(11, 16)
        txs = []
        curr_time = win_start
        for _ in range(tx_count):
            curr_time += timedelta(seconds=random.randint(5, 18))
            amount = round(random.uniform(15.00, 200.00), 2)
            txs.append({
                "amount": amount,
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
        windows.append({
            "window_id": win_id,
            "merchant_id": merchant_id,
            "transactions": txs,
            "label": 1,
            "pattern_type": "velocity_spike"
        })

    # 4. Off-Hours Cluster Anomaly (Label 1: 15 windows)
    for i in range(15):
        win_id = f"win_off_{i+1:03d}"
        day_offset = random.randint(0, 14)
        hour = random.randint(1, 3) # Between 01:00 and 04:00
        minute = random.randint(0, 45)
        win_start = base_time + timedelta(days=day_offset, hours=hour, minutes=minute)
        
        # 4-8 transactions during off-hours
        tx_count = random.randint(4, 8)
        txs = []
        curr_time = win_start
        for _ in range(tx_count):
            curr_time += timedelta(seconds=random.randint(30, 300))
            amount = round(random.uniform(40.00, 350.00), 2)
            txs.append({
                "amount": amount,
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
        windows.append({
            "window_id": win_id,
            "merchant_id": merchant_id,
            "transactions": txs,
            "label": 1,
            "pattern_type": "off_hours_cluster"
        })

    # 5. Card Testing Anomaly (Label 1: 15 windows)
    for i in range(15):
        win_id = f"win_card_{i+1:03d}"
        day_offset = random.randint(0, 14)
        hour = random.randint(8, 22)
        minute = random.randint(0, 50)
        win_start = base_time + timedelta(days=day_offset, hours=hour, minutes=minute)
        
        # 6-10 transactions with near-identical small amounts ($1.00 - $3.00)
        tx_count = random.randint(6, 10)
        base_amt = round(random.uniform(1.00, 2.50), 2)
        txs = []
        curr_time = win_start
        for _ in range(tx_count):
            curr_time += timedelta(seconds=random.randint(10, 45))
            amt = round(base_amt + random.choice([0.0, 0.01, 0.02, -0.01]), 2)
            txs.append({
                "amount": amt,
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
        windows.append({
            "window_id": win_id,
            "merchant_id": merchant_id,
            "transactions": txs,
            "label": 1,
            "pattern_type": "card_testing"
        })

    # Shuffle deterministically
    random.shuffle(windows)
    return windows

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    windows = generate_transaction_windows()
    
    # 70% train (~126), 30% test (~54)
    total = len(windows)
    train_count = int(round(total * 0.7))
    
    train_data = windows[:train_count]
    test_data = windows[train_count:]
    
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
        
    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
        
    def print_split_stats(name, data):
        l0 = sum(1 for w in data if w["label"] == 0)
        l1 = sum(1 for w in data if w["label"] == 1)
        pct = (l1 / len(data)) * 100 if data else 0
        p_types = {}
        for w in data:
            pt = w["pattern_type"] or "normal"
            p_types[pt] = p_types.get(pt, 0) + 1
        print(f"--- {name} Split Summary ---")
        print(f"Total Windows: {len(data)}")
        print(f"Label 0 (Normal): {l0} ({(l0/len(data))*100:.1f}%)")
        print(f"Label 1 (Fraud) : {l1} ({pct:.1f}%)")
        print(f"Pattern Types   : {p_types}\n")

    print(f"Generated {total} total transaction windows with random seed {SEED}.\n")
    print_split_stats("Train (70%)", train_data)
    print_split_stats("Test (30%)", test_data)

if __name__ == "__main__":
    main()
