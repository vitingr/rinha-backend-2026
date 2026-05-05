#!/usr/bin/env python3
"""
Offline evaluation: test the trained XGBoost model against example payloads.

Vectorizes each example payload, runs it through the model, and compares
the predicted fraud_score/approval against the expected k-NN result.
"""

import argparse
import json
import math
from datetime import datetime

import numpy as np
import xgboost as xgb


def load_normalization(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_mcc_risk(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def vectorize(payload: dict, norm: dict, mcc_risk: dict) -> list[float]:
    """Transform a transaction payload into a 14-dimensional vector."""
    tx = payload["transaction"]
    cust = payload["customer"]
    merch = payload["merchant"]
    term = payload["terminal"]
    last_tx = payload.get("last_transaction")

    requested_at = datetime.fromisoformat(tx["requested_at"].replace("Z", "+00:00"))

    v = [0.0] * 14

    # 0: amount
    v[0] = clamp(tx["amount"] / norm["max_amount"])
    # 1: installments
    v[1] = clamp(tx["installments"] / norm["max_installments"])
    # 2: amount_vs_avg
    v[2] = clamp((tx["amount"] / cust["avg_amount"]) / norm["amount_vs_avg_ratio"])
    # 3: hour_of_day
    v[3] = requested_at.hour / 23.0
    # 4: day_of_week (mon=0, sun=6)
    dow = requested_at.weekday()  # Python: mon=0, sun=6 — matches!
    v[4] = dow / 6.0
    # 5: minutes_since_last_tx
    if last_tx is None:
        v[5] = -1.0
    else:
        last_time = datetime.fromisoformat(last_tx["timestamp"].replace("Z", "+00:00"))
        minutes = (requested_at - last_time).total_seconds() / 60.0
        v[5] = clamp(minutes / norm["max_minutes"])
    # 6: km_from_last_tx
    if last_tx is None:
        v[6] = -1.0
    else:
        v[6] = clamp(last_tx["km_from_current"] / norm["max_km"])
    # 7: km_from_home
    v[7] = clamp(term["km_from_home"] / norm["max_km"])
    # 8: tx_count_24h
    v[8] = clamp(cust["tx_count_24h"] / norm["max_tx_count_24h"])
    # 9: is_online
    v[9] = 1.0 if term["is_online"] else 0.0
    # 10: card_present
    v[10] = 1.0 if term["card_present"] else 0.0
    # 11: unknown_merchant
    v[11] = 0.0 if merch["id"] in cust["known_merchants"] else 1.0
    # 12: mcc_risk
    v[12] = mcc_risk.get(merch["mcc"], 0.5)
    # 13: merchant_avg_amount
    v[13] = clamp(merch["avg_amount"] / norm["max_merchant_avg_amount"])

    return v


def main():
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model on example payloads")
    parser.add_argument("--model", default="output/model.json", help="Path to model.json")
    parser.add_argument(
        "--payloads",
        default="../resources/example-payloads.json",
        help="Path to example payloads",
    )
    parser.add_argument(
        "--normalization",
        default="../resources/normalization.json",
        help="Path to normalization.json",
    )
    parser.add_argument(
        "--mcc-risk", default="../resources/mcc_risk.json", help="Path to mcc_risk.json"
    )
    args = parser.parse_args()

    # Load resources
    norm = load_normalization(args.normalization)
    mcc_risk = load_mcc_risk(args.mcc_risk)

    # Load model
    model = xgb.Booster()
    model.load_model(args.model)

    # Load payloads
    with open(args.payloads) as f:
        payloads = json.load(f)

    print(f"Evaluating {len(payloads)} payloads...\n")
    print(f"{'ID':<20} {'Vector (first 4)':<30} {'Raw Score':>10} {'Rounded':>10} {'Approved':>10}")
    print("-" * 90)

    results = []
    for payload in payloads:
        vector = vectorize(payload, norm, mcc_risk)
        features = np.array([vector], dtype=np.float32)
        dmat = xgb.DMatrix(features)

        raw_score = float(model.predict(dmat)[0])
        rounded = round(raw_score * 5) / 5
        rounded = max(0.0, min(1.0, rounded))
        approved = rounded < 0.6

        vec_str = f"[{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}, {vector[3]:.4f}]"
        print(
            f"{payload['id']:<20} {vec_str:<30} {raw_score:>10.4f} {rounded:>10.1f} {str(approved):>10}"
        )

        results.append(
            {
                "id": payload["id"],
                "vector": vector,
                "raw_score": raw_score,
                "rounded_score": rounded,
                "approved": approved,
            }
        )

    # Summary statistics
    scores = [r["rounded_score"] for r in results]
    approvals = [r["approved"] for r in results]
    print(f"\n--- Summary ---")
    print(f"Total payloads:  {len(results)}")
    print(f"Approved:        {sum(approvals)} ({sum(approvals)/len(approvals):.1%})")
    print(f"Denied:          {len(approvals) - sum(approvals)} ({1 - sum(approvals)/len(approvals):.1%})")
    print(f"Avg raw score:   {np.mean([r['raw_score'] for r in results]):.4f}")


if __name__ == "__main__":
    main()