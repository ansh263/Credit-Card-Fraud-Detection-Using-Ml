# train_model.py
# FIXED + OPTIMIZED VERSION

import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# -----------------------------------
# CREATE FOLDERS
# -----------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -----------------------------------
# LOAD DATASET
# -----------------------------------
print("Loading dataset...")

df = pd.read_csv("dataset/sample_creditcard.csv")

# Split features and target
X = df.drop(columns=["Class"])
y = df["Class"]

print("Dataset Loaded Successfully")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------
print("\nSplitting train/test data...")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------------
# FEATURE SCALING
# -----------------------------------
print("Applying RobustScaler...")

scaler = RobustScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(X_train_raw),
    columns=X.columns
)

X_test = pd.DataFrame(
    scaler.transform(X_test_raw),
    columns=X.columns
)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# -----------------------------------
# MODELS
# -----------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=5000,
        solver="saga",
        n_jobs=-1,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ),

    # FIXED: Linear kernel makes SVM much faster
    "SVM": SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced",
        max_iter=2000
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1
    )
}

# -----------------------------------
# TRAIN + EVALUATE
# -----------------------------------
results = []
trained_models = {}

print("\nTraining Started...\n")

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, prob)

    results.append([
        name,
        acc,
        pre,
        rec,
        f1,
        auc
    ])

    trained_models[name] = model

    # Save individual model
    safe_name = name.replace(" ", "_")
    joblib.dump(model, f"models/{safe_name}.pkl")

# -----------------------------------
# RESULTS DATAFRAME
# -----------------------------------
result_df = pd.DataFrame(results, columns=[
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC AUC"
])

# Sort best first
result_df = result_df.sort_values(
    by=["F1 Score", "ROC AUC"],
    ascending=False
)

# Save CSV
result_df.to_csv("models/model_results.csv", index=False)

print("\nModel Results:\n")
print(result_df)

# -----------------------------------
# BEST MODEL
# -----------------------------------
best_name = result_df.iloc[0]["Model"]
best_model = trained_models[best_name]

print(f"\nBest Model Selected: {best_name}")

joblib.dump(best_model, "models/best_model.pkl")

# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------
best_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# -----------------------------------
# F1 SCORE GRAPH
# -----------------------------------
plt.figure(figsize=(10, 5))

plt.bar(
    result_df["Model"],
    result_df["F1 Score"]
)

plt.xticks(rotation=45)
plt.ylabel("F1 Score")
plt.title("Model Comparison")

plt.tight_layout()
plt.savefig("outputs/f1_scores.png")
plt.close()

# -----------------------------------
# DONE
# -----------------------------------
print("\nSuccess!")
print("Saved Files:")
print("models/best_model.pkl")
print("models/scaler.pkl")
print("models/model_results.csv")
print("outputs/confusion_matrix.png")
print("outputs/f1_scores.png")