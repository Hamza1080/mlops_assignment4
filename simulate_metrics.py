"""
MLOps Monitoring Simulation
Generates realistic metric progression and saves evidence files
Run: python simulate_metrics.py
"""

import time
import json
import csv
import os
import requests
from datetime import datetime

API_URL = "http://localhost:8000"
OUTPUT_DIR = "simulation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Phases: normal → early drift → severe drift → retraining → recovery
# ============================================================================

PHASES = [
    {
        "name": "Phase 1 - Normal Operation",
        "duration_seconds": 60,
        "metrics": {"recall": 0.88, "auc": 0.908, "psi": 0.05,
                    "false_positive_rate": 0.04, "missing_percentage": 0.01,
                    "feature_shift_max": 0.08},
        "requests": 20
    },
    {
        "name": "Phase 2 - Early Drift",
        "duration_seconds": 60,
        "metrics": {"recall": 0.78, "auc": 0.870, "psi": 0.12,
                    "false_positive_rate": 0.09, "missing_percentage": 0.03,
                    "feature_shift_max": 0.11},
        "requests": 20
    },
    {
        "name": "Phase 3 - Severe Drift (Alerts Fire)",
        "duration_seconds": 60,
        "metrics": {"recall": 0.55, "auc": 0.760, "psi": 0.32,
                    "false_positive_rate": 0.21, "missing_percentage": 0.08,
                    "feature_shift_max": 0.25},
        "requests": 20
    },
    {
        "name": "Phase 4 - Retraining Triggered",
        "duration_seconds": 60,
        "metrics": {"recall": 0.65, "auc": 0.810, "psi": 0.22,
                    "false_positive_rate": 0.14, "missing_percentage": 0.05,
                    "feature_shift_max": 0.18},
        "requests": 20
    },
    {
        "name": "Phase 5 - Recovery (New Model Deployed)",
        "duration_seconds": 60,
        "metrics": {"recall": 0.92, "auc": 0.931, "psi": 0.03,
                    "false_positive_rate": 0.03, "missing_percentage": 0.01,
                    "feature_shift_max": 0.05},
        "requests": 20
    }
]

# ============================================================================
# Helpers
# ============================================================================

def set_metrics(metrics: dict) -> bool:
    try:
        r = requests.post(f"{API_URL}/test/set_metrics",
                          json=metrics, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"  [ERROR] set_metrics failed: {e}")
        return False


def send_predictions(n: int, phase_name: str) -> list:
    import random
    results = []
    for i in range(n):
        features = [round(random.gauss(0, 1), 4) for _ in range(530)]
        try:
            r = requests.post(f"{API_URL}/predict",
                              json={"transaction_id": f"{phase_name}_TX_{i}",
                                    "features": features},
                              timeout=5)
            if r.status_code == 200:
                data = r.json()
                results.append({
                    "transaction_id": data.get("transaction_id"),
                    "is_fraud": data.get("is_fraud"),
                    "fraud_probability": round(data.get("fraud_probability", 0), 4),
                    "confidence": round(data.get("confidence", 0), 4),
                    "inference_time_ms": round(data.get("inference_time_ms", 0), 2)
                })
        except Exception:
            pass
        time.sleep(0.1)
    return results


def check_alerts() -> dict:
    try:
        r = requests.get("http://localhost:9090/api/v1/alerts", timeout=5)
        if r.status_code == 200:
            alerts = r.json().get("data", {}).get("alerts", [])
            firing = [a["labels"]["alertname"] for a in alerts
                      if a["state"] == "firing"]
            pending = [a["labels"]["alertname"] for a in alerts
                       if a["state"] == "pending"]
            return {"firing": firing, "pending": pending}
    except Exception:
        pass
    return {"firing": [], "pending": []}


# ============================================================================
# Main simulation
# ============================================================================

def run_simulation():
    print("=" * 60)
    print("MLOps MONITORING SIMULATION")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Check API is up
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        print(f"FastAPI status: {r.json().get('status', 'unknown')}")
    except Exception:
        print("ERROR: FastAPI not running at localhost:8000")
        print("Run: uvicorn src.app:app --host 0.0.0.0 --port 8000")
        return

    simulation_log = []
    all_predictions = []
    alert_timeline = []

    for phase_idx, phase in enumerate(PHASES):
        print(f"\n{'='*50}")
        print(f"{phase['name']}")
        print(f"{'='*50}")

        # Set metrics
        success = set_metrics(phase["metrics"])
        timestamp = datetime.now().isoformat()

        m = phase["metrics"]
        print(f"  Recall:    {m['recall']} {'⚠ BELOW THRESHOLD' if m['recall'] < 0.70 else '✓'}")
        print(f"  AUC-ROC:   {m['auc']} {'⚠ BELOW THRESHOLD' if m['auc'] < 0.85 else '✓'}")
        print(f"  PSI:       {m['psi']} {'⚠ DRIFT DETECTED' if m['psi'] > 0.20 else '✓'}")
        print(f"  FPR:       {m['false_positive_rate']}")

        # Send predictions
        print(f"  Sending {phase['requests']} prediction requests...")
        preds = send_predictions(phase["requests"], f"P{phase_idx+1}")
        fraud_count = sum(1 for p in preds if p.get("is_fraud"))
        print(f"  Predictions: {len(preds)} sent, {fraud_count} flagged as fraud")
        all_predictions.extend(preds)

        # Check alerts
        time.sleep(3)
        alert_status = check_alerts()
        print(f"  Alerts FIRING:  {alert_status['firing'] or 'none'}")
        print(f"  Alerts PENDING: {alert_status['pending'] or 'none'}")

        # Log phase
        log_entry = {
            "phase": phase_idx + 1,
            "name": phase["name"],
            "timestamp": timestamp,
            "metrics": phase["metrics"],
            "predictions_sent": len(preds),
            "fraud_detections": fraud_count,
            "alerts_firing": alert_status["firing"],
            "alerts_pending": alert_status["pending"],
            "metrics_set_successfully": success
        }
        simulation_log.append(log_entry)
        alert_timeline.append({
            "phase": phase["name"],
            "timestamp": timestamp,
            "recall": m["recall"],
            "auc": m["auc"],
            "psi": m["psi"],
            "alerts_firing": len(alert_status["firing"]),
            "alert_names": ", ".join(alert_status["firing"]) or "none"
        })

        # Wait before next phase
        if phase_idx < len(PHASES) - 1:
            wait = phase["duration_seconds"]
            print(f"\n  Waiting {wait}s before next phase...")
            print(f"  --> Screenshot Grafana dashboards now! <--")
            for remaining in range(wait, 0, -10):
                print(f"      {remaining}s remaining...", end="\r")
                time.sleep(min(10, remaining))
            print()

    # ========================================================================
    # Save output files
    # ========================================================================

    print(f"\n{'='*60}")
    print("SAVING EVIDENCE FILES")
    print(f"{'='*60}")

    # 1. Full simulation log (JSON)
    log_path = f"{OUTPUT_DIR}/simulation_log.json"
    with open(log_path, "w") as f:
        json.dump(simulation_log, f, indent=2)
    print(f"✓ {log_path}")

    # 2. Alert timeline (CSV)
    timeline_path = f"{OUTPUT_DIR}/alert_timeline.csv"
    with open(timeline_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "phase", "timestamp", "recall", "auc",
            "psi", "alerts_firing", "alert_names"
        ])
        writer.writeheader()
        writer.writerows(alert_timeline)
    print(f"✓ {timeline_path}")

    # 3. All predictions (CSV)
    if all_predictions:
        pred_path = f"{OUTPUT_DIR}/simulation_predictions.csv"
        with open(pred_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_predictions[0].keys())
            writer.writeheader()
            writer.writerows(all_predictions)
        print(f"✓ {pred_path}")

    # 4. Summary report (TXT)
    summary_path = f"{OUTPUT_DIR}/simulation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("MLOps MONITORING SIMULATION SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("PHASE SUMMARY\n")
        f.write("-" * 50 + "\n")
        for entry in simulation_log:
            f.write(f"\n{entry['name']}\n")
            f.write(f"  Recall:  {entry['metrics']['recall']}\n")
            f.write(f"  AUC:     {entry['metrics']['auc']}\n")
            f.write(f"  PSI:     {entry['metrics']['psi']}\n")
            f.write(f"  Predictions sent: {entry['predictions_sent']}\n")
            f.write(f"  Fraud detected:   {entry['fraud_detections']}\n")
            alerts = entry['alerts_firing']
            f.write(f"  Alerts firing:    {alerts if alerts else 'none'}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("ALERT EVENTS\n")
        f.write("-" * 50 + "\n")
        for entry in alert_timeline:
            if entry["alerts_firing"] > 0:
                f.write(f"  {entry['phase']}: {entry['alert_names']}\n")

        total_preds = sum(e["predictions_sent"] for e in simulation_log)
        total_fraud = sum(e["fraud_detections"] for e in simulation_log)
        f.write(f"\nTotal predictions: {total_preds}\n")
        f.write(f"Total fraud flagged: {total_fraud}\n")
        f.write(f"Overall fraud rate: {total_fraud/total_preds:.1%}\n")

    print(f"✓ {summary_path}")

    # 5. Final status
    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nFiles saved to: {OUTPUT_DIR}/")
    print(f"  - simulation_log.json       (full phase data)")
    print(f"  - alert_timeline.csv        (metrics + alerts over time)")
    print(f"  - simulation_predictions.csv (all API predictions)")
    print(f"  - simulation_summary.txt    (human-readable summary)")
    print(f"\nNext steps:")
    print(f"  1. Check http://localhost:9090/alerts for firing alerts")
    print(f"  2. Check http://localhost:9093 for Alertmanager")
    print(f"  3. Check GitHub Actions tab for triggered retraining")
    print(f"  4. Screenshot Grafana at http://localhost:3000")
    print(f"\nThese files are your Task 6, 7, 8 evidence for the report.")


if __name__ == "__main__":
    run_simulation()