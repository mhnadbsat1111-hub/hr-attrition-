# -*- coding: utf-8 -*-
"""
HR Employee Attrition Predictor (MLP Neural Network)

Dataset: HR_comma_sep.csv (Kaggle HR Analytics)
Target : left (1 = employee left, 0 = stayed)

Run from the project root:  python src/train.py
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"images/{name}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------
# 1. Load data (local path, overridable; Colab supported)
# ---------------------------------------------------------------
CSV_PATH = os.environ.get("HR_CSV", "data/HR_comma_sep.csv")
try:  # Google Colab (optional)
    from google.colab import drive
    drive.mount("/content/drive")
    CSV_PATH = os.environ.get(
        "HR_CSV", "/content/drive/MyDrive/Colab Notebooks/data seting/HR_comma_sep.csv")
except ImportError:
    pass

raw = pd.read_csv(CSV_PATH)

# Different versions of the dataset spell some columns differently
raw = raw.rename(columns={"average_montly_hours": "average_monthly_hours",
                          "Departments ": "sales", "Departments": "sales",
                          "Department": "sales"})

print("Shape:", raw.shape)
print("\nMissing values per column:\n", raw.isna().sum())
print("\nDuplicate rows:", raw.duplicated().sum())
print("\nTarget distribution:\n", raw["left"].value_counts())

# Duplicates would leak between train and test, so remove them
df = raw.drop_duplicates().reset_index(drop=True)
print("\nShape after removing duplicates:", df.shape)
print("Target distribution after cleaning (%):\n",
      (df["left"].value_counts(normalize=True) * 100).round(2))

# ---------------------------------------------------------------
# 2. Exploratory data analysis (original, un-encoded data)
# ---------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="left", hue="left", data=df, palette="Set2", legend=False)
plt.title("Distribution of Target (left)")
save_fig("eda_target_distribution")

discrete_features = ["number_project", "time_spend_company", "Work_accident",
                     "promotion_last_5years", "sales", "salary"]
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
for ax, feature in zip(axes.flatten(), discrete_features):
    sns.countplot(x=feature, hue="left", data=df, palette="viridis", ax=ax)
    ax.set_title(f"{feature} vs. left")
    ax.tick_params(axis="x", rotation=45)
save_fig("eda_discrete_features")

continuous_features = ["satisfaction_level", "last_evaluation", "average_monthly_hours"]
fig, axes = plt.subplots(2, len(continuous_features), figsize=(16, 8))
for i, feature in enumerate(continuous_features):
    sns.histplot(data=df, x=feature, hue="left", kde=True, ax=axes[0, i])
    axes[0, i].set_title(f"Distribution of {feature}")
    sns.boxplot(data=df, x="left", y=feature, ax=axes[1, i])
    axes[1, i].set_title(f"Boxplot of {feature} by left")
save_fig("eda_continuous_features")

# ---------------------------------------------------------------
# 3. Pre-processing
#    salary is ordinal -> ordered mapping; department is nominal -> one-hot
# ---------------------------------------------------------------
encoded = df.copy()
encoded["salary"] = encoded["salary"].map({"low": 0, "medium": 1, "high": 2})
encoded = pd.get_dummies(encoded, columns=["sales"], drop_first=True, dtype=int)

plt.figure(figsize=(14, 10))
sns.heatmap(encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f",
            annot_kws={"size": 7})
plt.title("Correlation Matrix")
save_fig("correlation_matrix")

# ---------------------------------------------------------------
# 4. Train / test split (70/30, stratified)
# ---------------------------------------------------------------
X = encoded.drop("left", axis=1)
y = encoded["left"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
print("Number of input features:", X.shape[1])

# ---------------------------------------------------------------
# 5. Models: MLP + two simple baselines on the SAME split
#    Scalers are fitted on training data only (no leakage)
# ---------------------------------------------------------------
models = {
    "MLP (100 hidden units)": Pipeline([
        ("scaler", MinMaxScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
                              solver="adam", max_iter=1000,
                              random_state=RANDOM_STATE)),
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=200,
                                            random_state=RANDOM_STATE),
}

results, preds = {}, {}
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    p = mdl.predict(X_test)
    preds[name] = p
    results[name] = {
        "Accuracy": accuracy_score(y_test, p),
        "Precision": precision_score(y_test, p),
        "Recall": recall_score(y_test, p),
        "F1": f1_score(y_test, p),
    }

summary = pd.DataFrame(results).T.round(4)
print("\n===== Model comparison (test set, left = positive class) =====")
print(summary)
summary.to_csv("results.csv")

mlp_name = "MLP (100 hidden units)"
mlp = models[mlp_name]
print(f"\n===== {mlp_name}: classification report =====")
print(classification_report(y_test, preds[mlp_name], target_names=["Stayed", "Left"]))

# ---------------------------------------------------------------
# 6. Confusion matrix + training loss (MLP)
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, preds[mlp_name])
print("Confusion matrix (rows = actual, cols = predicted):\n", cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
plt.title("Confusion Matrix (MLP)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
save_fig("confusion_matrix_mlp")

plt.figure(figsize=(6, 4))
plt.plot(mlp.named_steps["mlp"].loss_curve_)
plt.title("MLP Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
save_fig("mlp_training_loss")

# ---------------------------------------------------------------
# 7. Which features matter most? (permutation importance, F1 drop on test set)
# ---------------------------------------------------------------
imp = permutation_importance(mlp, X_test, y_test, scoring="f1", n_repeats=10,
                             random_state=RANDOM_STATE)
imp_df = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean})
          .sort_values("importance", ascending=False))
print("\nTop features (permutation importance):\n", imp_df.head(8).round(4))
imp_df.to_csv("feature_importance.csv", index=False)

top = imp_df.head(10).iloc[::-1]
plt.figure(figsize=(7, 5))
plt.barh(top["feature"], top["importance"])
plt.title("Top 10 features (permutation importance, MLP)")
plt.xlabel("Mean drop in F1 when feature is shuffled")
save_fig("feature_importance_mlp")

# ---------------------------------------------------------------
# 8. K-Means clustering of employees who left
# ---------------------------------------------------------------
left_emp = df.loc[df["left"] == 1, ["satisfaction_level", "last_evaluation"]].copy()
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(left_emp)
left_emp["label"] = kmeans.labels_

plt.figure(figsize=(6, 4))
plt.scatter(left_emp["satisfaction_level"], left_emp["last_evaluation"],
            c=left_emp["label"], cmap="Accent")
plt.xlabel("Satisfaction Level")
plt.ylabel("Last Evaluation")
plt.title("3 Clusters of Employees Who Left")
save_fig("kmeans_clusters")

print("\nCluster sizes:\n", left_emp["label"].value_counts().sort_index())
print("\nCluster centres (satisfaction, evaluation):\n",
      kmeans.cluster_centers_.round(3))

# ---------------------------------------------------------------
# 9. Save the model and the feature order for src/predict.py
# ---------------------------------------------------------------
joblib.dump({"model": mlp, "columns": list(X.columns)}, "models/mlp_attrition.joblib")
print("\nSaved models/mlp_attrition.joblib")
