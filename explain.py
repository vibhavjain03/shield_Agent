"""
explain.py
Generates concise, human-readable fraud explanations using Gemini LLM (gemini-2.5-flash)
with deterministic template fallback if API key is missing or call fails.
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()

def _fallback_explanation(window, detection_result):
    """Fallback generator using facts dictionary directly."""
    facts = detection_result.get("facts", {})
    pattern = detection_result.get("pattern_type")
    window_id = window.get("window_id", "Unknown")
    
    if detection_result.get("predicted_label") == 0:
        return f"Window {window_id} appears normal with {facts.get('total_tx_count', 0)} transactions and no detected anomaly patterns."
        
    if pattern == "velocity_spike":
        max_tx = facts.get("max_tx_in_window", "multiple")
        sec = facts.get("window_seconds", facts.get("time_frame_seconds", 300))
        return f"Flagged for velocity spike: Identified {max_tx} transactions occurring rapidly within a {sec}-second timeframe."
        
    elif pattern == "off_hours_cluster":
        count = facts.get("off_hours_tx_count", "multiple")
        hours = facts.get("off_hours_window", "01:00 - 04:00")
        return f"Flagged for off-hours activity: Detected {count} transactions processed during non-business hours ({hours})."
        
    elif pattern == "card_testing":
        count = facts.get("card_testing_tx_count", "multiple")
        avg = facts.get("average_small_amount", 0.0)
        return f"Flagged for card testing pattern: Detected {count} near-identical small-value transactions averaging ${avg:.2f}."
        
    return f"Flagged suspicious activity pattern ({pattern}) based on trigger facts: {facts}."


def explain_flag(window, detection_result):
    """
    Calls Gemini API (gemini-2.5-flash, temp=0) to generate a 1-2 sentence explanation.
    Falls back gracefully to template if API key is missing or error occurs.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key and api_key.strip() != "" and api_key != "your_key_here":
        prompt = (
            "You are an expert fraud investigator. Write a clear 1-2 sentence plain-language explanation "
            "for why this merchant transaction window was flagged. Cite exact numbers from the trigger facts.\n"
            f"Window ID: {window.get('window_id')}\n"
            f"Pattern Type: {detection_result.get('pattern_type')}\n"
            f"Detection Facts: {json.dumps(detection_result.get('facts'))}\n"
            "Output ONLY the 1-2 sentence explanation without intro or quotes."
        )

        # Attempt 1: google-genai SDK (genai.Client)
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"temperature": 0.0}
            )
            if response and response.text:
                return response.text.strip()
        except Exception:
            pass

        # Attempt 2: google-generativeai SDK
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=api_key)
            model = genai_legacy.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0}
            )
            if response and response.text:
                return response.text.strip()
        except Exception:
            pass

    # Fallback to deterministic template
    return _fallback_explanation(window, detection_result)


if __name__ == "__main__":
    # Self-test with mock window and detection result
    mock_win = {"window_id": "win_test_001"}
    mock_res = {
        "predicted_label": 1,
        "pattern_type": "velocity_spike",
        "facts": {"max_tx_in_window": 14, "window_seconds": 180, "threshold_tx_count": 10}
    }
    exp = explain_flag(mock_win, mock_res)
    print("Self-test Explanation Result:\n", exp)
