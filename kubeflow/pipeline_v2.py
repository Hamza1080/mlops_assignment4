"""
Kubeflow Pipelines v2 - Fraud Detection (KFP 2.14.3 Compatible)
"""
from kfp import dsl, compiler
from kfp import kubernetes
from kfp.dsl import component, pipeline


@component(
    base_image='10.107.196.7/fraud-base:latest',
    packages_to_install=['pandas', 'numpy']
)
def load_and_validate_data(
    n_train_rows: int = 0,
    n_test_rows: int = 0,
    n_features: int = 0
) -> str:
    import pandas as pd
    X_train = pd.read_csv('/artifacts/data/X_train_sample.csv')
    X_test = pd.read_csv('/artifacts/data/X_test_sample.csv')
    y_train = pd.read_csv('/artifacts/data/y_train_sample.csv')
    y_test = pd.read_csv('/artifacts/data/y_test_sample.csv')
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    print(f"✓ X_train: {X_train.shape}, X_test: {X_test.shape}")
    print("✓ Data validation: PASS")
    return "validated"


@component(
    base_image='10.107.196.7/fraud-base:latest',
    packages_to_install=['pandas', 'xgboost', 'scikit-learn', 'joblib']
)
def train_model(validation_status: str) -> str:
    import pandas as pd
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score
    import joblib
    X_train = pd.read_csv('/artifacts/data/X_train_sample.csv')
    y_train = pd.read_csv('/artifacts/data/y_train_sample.csv').values.ravel()
    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        tree_method='hist', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    auc = roc_auc_score(y_train, y_pred_proba)
    print(f"✓ Training AUC-ROC: {auc:.4f}")
    model_path = '/artifacts/models/xgboost_kfp.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Model saved to {model_path}")
    return model_path


@component(
    base_image='10.107.196.7/fraud-base:latest',
    packages_to_install=['pandas', 'xgboost', 'scikit-learn', 'joblib']
)
def evaluate_model(model_path: str) -> str:
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    import joblib
    model = joblib.load(model_path)
    X_test = pd.read_csv('/artifacts/data/X_test_sample.csv')
    y_test = pd.read_csv('/artifacts/data/y_test_sample.csv').values.ravel()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"✓ Precision: {precision:.4f}")
    print(f"✓ Recall: {recall:.4f}")
    print(f"✓ F1: {f1:.4f}")
    print(f"✓ AUC-ROC: {auc:.4f}")
    passes = "true" if auc >= 0.85 else "false"
    print(f"✓ Deployment gate (AUC >= 0.85): {passes.upper()}")
    return passes


@component(base_image='10.107.196.7/fraud-base:latest')
def deploy_decision(passes_threshold: str) -> str:
    if passes_threshold.lower() == "true":
        print("✓ Model APPROVED for deployment")
        return "DEPLOY"
    else:
        print("⚠️  Model REJECTED")
        return "SKIP"


@component(
    base_image='10.107.196.7/fraud-base:latest',
    packages_to_install=['pandas']
)
def export_metrics(passes_threshold: str) -> str:
    import pandas as pd
    from datetime import datetime
    metrics = {
        'timestamp': [datetime.utcnow().isoformat()],
        'deployment_decision': [passes_threshold],
    }
    df = pd.DataFrame(metrics)
    path = '/artifacts/models/pipeline_metrics.csv'
    df.to_csv(path, index=False)
    print(f"✓ Metrics exported to {path}")
    return path


@pipeline(
    name='fraud-detection-pipeline',
    description='Fraud detection ML pipeline'
)
def fraud_detection_pipeline():
    from kubernetes.client import V1Volume, V1VolumeMount, V1PersistentVolumeClaimVolumeSource
    pvc_name = 'fraud-artifacts-pvc'
    load_task = load_and_validate_data().set_caching_options(False)
    kubernetes.mount_pvc(load_task, pvc_name=pvc_name, mount_path='/artifacts')
    train_task = train_model(validation_status=load_task.output).set_caching_options(False)
    kubernetes.mount_pvc(train_task, pvc_name=pvc_name, mount_path='/artifacts')
    eval_task = evaluate_model(model_path=train_task.output).set_caching_options(False)
    kubernetes.mount_pvc(eval_task, pvc_name=pvc_name, mount_path='/artifacts')
    deploy_task = deploy_decision(passes_threshold=eval_task.output).set_caching_options(False)
    kubernetes.mount_pvc(deploy_task, pvc_name=pvc_name, mount_path='/artifacts')
    export_task = export_metrics(passes_threshold=eval_task.output).set_caching_options(False)
    kubernetes.mount_pvc(export_task, pvc_name=pvc_name, mount_path='/artifacts')


if __name__ == '__main__':
    output_file = 'fraud_detection_pipeline.yaml'
    compiler.Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path=output_file
    )
    print(f"✓ Pipeline compiled to {output_file}")
