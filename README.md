
# Used Car Price Prediction Project

## Business Objective
Provide insights into factors influencing used car prices and build predictive models to support used car dealers in inventory pricing and strategy decisions.

---

## Data Summary
- **Features:** 142
- **Training Samples:** 287,764
- **Test Samples:** 71,941

---

## Model Performance

### Linear Regression
- **Test RMSE:** 0.70
- **R²:** 0.5149

### Ridge Regression
- **Test RMSE:** 0.70
- **R²:** 0.5149

**Best Model:**  Linear Regression

---

## Key Insights
- Car age and odometer strongly influence price.
- Manufacturer and fuel type also impact pricing.
- Regularization (Ridge) only slightly improves performance over Linear Regression.

---

## Business Interpretation
- Linear Regression is simpler, interpretable, and performs as well or better than Ridge Regression.
- Dealers can easily understand coefficient impacts (e.g., how car age or mileage affects price).

---

## Recommendations
- Focus on newer cars with lower mileage for higher resale value.
- Electric and diesel vehicles tend to have higher average prices.
- Use the saved model for inventory pricing optimization.

---

## How to Use the Saved Model

The best model (`best_price_model.pkl`) is saved using `joblib`. Here’s how you can use it to make predictions on new data:

```python
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load("best_price_model.pkl")

# Load new data (must match training features and preprocessing)
new_data = pd.read_csv("your_new_data.csv")

# Feature engineering (example)
current_year = 2025
new_data['car_age'] = current_year - new_data['year']
new_data['price_per_mile'] = new_data['price'] / new_data['odometer']

# One-hot encode categorical variables (must match training columns)
cat_cols = [col for col in new_data.columns if new_data[col].dtype == 'object']
new_data = pd.get_dummies(new_data, columns=cat_cols, drop_first=True)

# Scale numeric features (use the same scaler fitted during training)
scaler = StandardScaler()
numeric_cols = ['year', 'odometer', 'car_age', 'price_per_mile']
new_data[numeric_cols] = scaler.fit_transform(new_data[numeric_cols])

# Predict log(price) and convert back to price
log_price_pred = model.predict(new_data)
price_pred = np.expm1(log_price_pred)
print("Predicted prices:", price_pred)
```

> **Important:** To get consistent predictions, apply *exactly the same preprocessing* used in training (feature engineering, encoding, scaling). Ideally, build an inference pipeline that reuses the trained encoders/scalers instead of fitting new ones on incoming data.

---

## Files Produced by This Project
- `best_price_model.pkl` — Saved best model for inference.
- `X_train_cleaned.csv`, `X_test_cleaned.csv`, `y_train_cleaned.csv`, `y_test_cleaned.csv` — Cleaned train/test splits.
- `price_prediction_report.txt` — Plain-text summary.
- `Project_report_readme.md` — This Markdown report.

---

## How to Reproduce
1. Run **Step 1–4 notebook** to prepare data, train models, evaluate, and deploy.
2. Confirm the saved files are present in your working directory.
3. Use the example code above to load the model and predict on new inventory data.

---

## Contact & Next Steps
If you’d like, we can package the preprocessing and model into a single **sklearn Pipeline** and provide a small CLI script that takes a CSV and outputs predicted prices.
