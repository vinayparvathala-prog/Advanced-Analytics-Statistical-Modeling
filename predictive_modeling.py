from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customer_churn.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
target = "Churn"
features = ["TenureMonths", "MonthlyCharges", "TotalCharges", "SupportCalls", "ContractType", "InternetService"]
X = df[features]
y = df[target]
numeric = ["TenureMonths", "MonthlyCharges", "TotalCharges", "SupportCalls"]
categorical = ["ContractType", "InternetService"]

preprocess = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
}
metrics = []
for name, estimator in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    try:
        score = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    except Exception:
        score = float("nan")
    metrics.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": score
    })
    (OUT / f"{name.lower().replace(' ', '_')}_classification_report.txt").write_text(
        classification_report(y_test, pred, zero_division=0), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
    ax.set_title(f"{name} — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUT / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=150)
    plt.close(fig)

metrics_df = pd.DataFrame(metrics).sort_values("F1", ascending=False)
metrics_df.to_csv(OUT / "churn_model_metrics.csv", index=False)

# Feature importance from Random Forest after one-hot encoding
rf_pipe = Pipeline([("preprocess", preprocess),
                    ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"))])
rf_pipe.fit(X_train, y_train)
try:
    names = rf_pipe.named_steps["preprocess"].get_feature_names_out()
    importances = rf_pipe.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame({"Feature": names, "Importance": importances}).sort_values("Importance", ascending=False)
    importance_df.to_csv(OUT / "feature_importance.csv", index=False)
    top = importance_df.head(12).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["Feature"], top["Importance"])
    ax.set_title("Random Forest Feature Importance")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)
except Exception as exc:
    print("Feature importance plot skipped:", exc)

# Numeric prediction example: estimate TotalCharges from tenure and monthly charges
reg_features = ["TenureMonths", "MonthlyCharges", "SupportCalls"]
Xr = df[reg_features]
yr = df["TotalCharges"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
reg = Pipeline([("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
reg.fit(Xr_train, yr_train)
rpred = reg.predict(Xr_test)
reg_summary = (
    f"Ridge Regression for TotalCharges\n"
    f"MAE: {mean_absolute_error(yr_test, rpred):.2f}\n"
    f"RMSE: {mean_squared_error(yr_test, rpred) ** 0.5:.2f}\n"
    f"R2: {r2_score(yr_test, rpred):.4f}\n"
)
(OUT / "regression_metrics.txt").write_text(reg_summary, encoding="utf-8")

print("Predictive modeling completed.")
print("\nChurn model metrics:\n", metrics_df.to_string(index=False))
print("\n", reg_summary)
print(f"Outputs saved to: {OUT}")
