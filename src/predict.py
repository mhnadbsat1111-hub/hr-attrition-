# -*- coding: utf-8 -*-
"""Estimate the probability that an employee will leave, using the trained MLP.

Run from the project root:  python src/predict.py
"""

import joblib
import pandas as pd

MODEL_PATH = "models/mlp_attrition.joblib"


def predict_employee(employee: dict, bundle=None) -> float:
    """Return the predicted probability (0-1) that the employee leaves."""
    bundle = bundle or joblib.load(MODEL_PATH)
    row = pd.DataFrame([employee])
    row["salary"] = row["salary"].map({"low": 0, "medium": 1, "high": 2})
    row = pd.get_dummies(row, columns=["sales"], dtype=int)
    row = row.reindex(columns=bundle["columns"], fill_value=0)  # match training columns
    return float(bundle["model"].predict_proba(row)[0, 1])


if __name__ == "__main__":
    bundle = joblib.load(MODEL_PATH)
    examples = {
        "Unhappy, overworked": dict(satisfaction_level=0.15, last_evaluation=0.90,
            number_project=7, average_monthly_hours=280, time_spend_company=4,
            Work_accident=0, promotion_last_5years=0, sales="sales", salary="low"),
        "Satisfied, balanced": dict(satisfaction_level=0.80, last_evaluation=0.70,
            number_project=3, average_monthly_hours=170, time_spend_company=3,
            Work_accident=0, promotion_last_5years=0, sales="sales", salary="medium"),
    }
    for label, emp in examples.items():
        print(f"{label:22} -> probability of leaving: {predict_employee(emp, bundle):.1%}")
