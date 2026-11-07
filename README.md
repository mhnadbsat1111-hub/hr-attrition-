# HR Employee Attrition Predictor

Predicts whether an employee will leave the company using an MLP neural network
(back-propagation), compared against two simple baselines. Course project, Neural Networks,
Computer Science, Taif University.

## Dataset
Kaggle HR Analytics (`HR_comma_sep.csv`), target `left` (1 = left, 0 = stayed).

- Raw rows: 14,999 (3,008 duplicates removed, 11,991 remain)
- Employees who left after cleaning: 16.6% (train 8,393 / test 3,598)
- Features: satisfaction level, last evaluation, number of projects, monthly hours,
  years at company, work accident, promotion in last 5 years, department (one-hot),
  salary (ordinal: low < medium < high)

Download the CSV and place it at `data/HR_comma_sep.csv`.

## Approach
- Duplicates removed before splitting so identical rows never appear in both train and test
- One stratified 70/30 split (seed 42) shared by all models
- `MinMaxScaler` fitted on training data only (no data leakage)
- Models: MLP (one hidden layer, 100 ReLU units, Adam), Logistic Regression, Random Forest
- Metrics: Accuracy, Precision, Recall, F1 (left = positive class)
- Extras: feature importance (permutation) and K-Means clustering of employees who left

## Results (held-out test set)

| Model                  | Accuracy | Precision | Recall | F1     |
|------------------------|----------|-----------|--------|--------|
| MLP (100 hidden units) | 0.9691   | 0.9175    | 0.8945 | 0.9059 |
| Logistic Regression    | 0.8341   | 0.5000    | 0.1826 | 0.2675 |
| Random Forest          | 0.9855   | 0.9875    | 0.9246 | 0.9550 |

**Findings**
- The MLP is far better than Logistic Regression, so the relationship between the
  features and attrition is strongly non-linear.
- The Random Forest scores higher than the MLP on this tabular dataset, so for production
  use it would be the stronger choice. The MLP is kept because this is a neural networks project.
- MLP confusion matrix: 534 of 597 leavers detected, 48 stayers wrongly flagged.
- Most influential features for the MLP: number of projects, satisfaction level,
  last evaluation, time at company, and monthly hours (`feature_importance.csv`).
- K-Means on satisfaction vs. evaluation splits leavers into three groups:
  low satisfaction with high evaluation, high satisfaction with high evaluation,
  and mid satisfaction with low evaluation.

Figures are in `images/` (confusion matrix, loss curve, feature importance, EDA, clusters).

## Project Structure
```
├── data/            # place HR_comma_sep.csv here
├── images/          # generated figures
├── models/          # saved MLP (generated)
├── src/
│   ├── train.py     # EDA, training, comparison, clustering
│   └── predict.py   # probability that an employee leaves
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py
```

## Tech Stack
Python, scikit-learn, pandas, matplotlib, seaborn

## Author
Muhannad Al-Thaqafi
