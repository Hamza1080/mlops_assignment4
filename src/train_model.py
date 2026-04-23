import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.utils import Bunch

class ProgressLogisticRegression(LogisticRegression):
    def fit(self, X, y, sample_weight=None):
        print("Starting training...")
        for i in range(1, self.max_iter + 1):
            super().fit(X, y, sample_weight)
            print(f"Iteration {i}/{self.max_iter} complete.")
        print("Training complete.")
        return self

# Load preprocessed data
X_train = pd.read_csv("data/X_train_preprocessed.csv")
X_test = pd.read_csv("data/X_test_preprocessed.csv")
y_train = pd.read_csv("data/y_train_smote.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Train a Logistic Regression model with progress logging
print("Initializing Logistic Regression model...")
model = ProgressLogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
print("Saving the model...")
joblib.dump(model, "models/logistic_regression_model.pkl")
print("Model saved to 'models/logistic_regression_model.pkl'")