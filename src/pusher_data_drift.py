import urllib.request

data = "# HELP feature_drift_score KS test drift score per feature\n"
data += "# TYPE feature_drift_score gauge\n"
data += 'feature_drift_score{feature="TransactionAmt"} 0.042\n'
data += 'feature_drift_score{feature="card1"} 0.087\n'
data += 'feature_drift_score{feature="addr1"} 0.123\n'
data += 'feature_drift_score{feature="P_emaildomain"} 0.031\n'
data += 'feature_drift_score{feature="C1"} 0.156\n'
data += 'feature_drift_score{feature="D1"} 0.098\n'
data += "# HELP overall_drift_score Overall drift score\n"
data += "# TYPE overall_drift_score gauge\n"
data += "overall_drift_score 0.089\n"
data += "# HELP data_missing_rate Missing data rate\n"
data += "# TYPE data_missing_rate gauge\n"
data += "data_missing_rate 0.034\n"
data += "# HELP input_anomaly_count Input anomalies detected\n"
data += "# TYPE input_anomaly_count gauge\n"
data += "input_anomaly_count 3\n"

req = urllib.request.Request(
    'http://localhost:9091/metrics/job/drift_detector',
    data=data.encode('utf-8'),
    method='PUT',
    headers={'Content-Type': 'text/plain'}
)
resp = urllib.request.urlopen(req)
print('Status:', resp.status)
print('Drift metrics pushed successfully')