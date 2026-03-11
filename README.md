<h1 align="center">🚗 Toyota Corolla Price Prediction — Multiple Linear Regression</h1>

> Predicting the resale price of a Toyota Corolla from its specs — age, mileage, fuel type, horsepower and more. Three progressively richer models are built and compared, with Ridge and Lasso regularization applied to check for overfitting.

---

# 📌 Problem

Given a set of car attributes, predict the **resale price** of a Toyota Corolla. This is a regression problem with a mix of continuous and categorical features. The interesting part here was building three models with different feature sets and seeing how much each additional feature group actually moved the needle.

---

# 📂 Dataset

| Property | Details |
|----------|---------|
| File | `ToyotaCorolla - MLR.csv` |
| Records | 1,436 cars |
| Features | 10 (9 numeric + 1 categorical) |
| Target | `Price` (resale price in €) |
| Missing Values | None |

**Features:**

| Feature | Type | Description |
|---------|------|-------------|
| `Age_08_04` | Numeric | Age of car in months (as of Aug 2004) |
| `KM` | Numeric | Kilometres driven |
| `Fuel_Type` | Categorical | Petrol (1264) · Diesel (155) · CNG (17) |
| `HP` | Numeric | Horsepower |
| `Automatic` | Binary | 0 = Manual (1356) · 1 = Automatic (80) |
| `cc` | Numeric | Engine displacement |
| `Doors` | Numeric | Number of doors |
| `Cylinders` | Numeric | Number of cylinders |
| `Gears` | Numeric | Number of gears |
| `Weight` | Numeric | Car weight (kg) |

---

# 🔍 Approach

**Outlier Treatment — Two-Stage Process:**
Standard winsorization alone didn't fully remove outliers (visible in before/after plots), so a two-stage method was applied:

1. **5% Winsorization** — caps bottom and top 5% of each numeric column
2. **IQR Clipping** — applied on the winsorized data: `clip(Q1 - 1.5×IQR, Q3 + 1.5×IQR)` to catch remaining outliers

> Why two stages? The `cc` column had an unusual flat distribution on a normal boxplot — log scale was needed just to see its outliers. Standard winsorization wasn't enough.

**Encoding:** `pd.get_dummies(drop_first=True)` on `Fuel_Type` → created `Fuel_Type_Diesel` and `Fuel_Type_Petrol` (CNG is the dropped reference)

**Three models built with increasing features:**

| Model | Features Used |
|-------|--------------|
| Model 1 | `Age_08_04`, `KM`, `HP`, `Automatic` |
| Model 2 | Model 1 + `Fuel_Type_Diesel`, `Fuel_Type_Petrol` |
| Model 3 | All 11 features including `cc`, `Doors`, `Cylinders`, `Gears`, `Weight` |

**Regularization:** Ridge (`α=1.0`) and Lasso (`α=1.0`) applied to all 3 models to check for overfitting.

---

# 🤖 Model / Algorithm

| Model | Type | Purpose |
|-------|------|---------|
| Linear Regression | OLS | Baseline price prediction |
| Ridge Regression | L2 regularization | Penalizes large coefficients, doesn't remove features |
| Lasso Regression | L1 regularization | Can zero out unimportant feature weights |

**Train/Test Split:** 80/20, `random_state=42`

---

# 📊 Results

**Model Comparison:**

| Model | R² | MAE (€) | MSE |
|-------|----|---------|-----|
| Model 1 (4 features) | 0.844 | 886.83 | 1,299,869 |
| Model 2 (+ Fuel Type) | 0.846 | 879.59 | 1,285,573 |
| **Model 3 (all features)** | **0.870** | **807.90** | **1,089,333** |

**Regularization check (R² scores):**

| | Model 1 | Model 2 | Model 3 |
|-|---------|---------|---------|
| Ridge | 0.844 | 0.846 | 0.870 |
| Lasso | 0.844 | 0.846 | 0.870 |

Ridge and Lasso scores are identical to Linear Regression across all 3 models — confirming the models are **not overfitting**. Ridge made small weight adjustments but didn't change accuracy. Lasso zeroed out `Cylinders` and `Gears` (coefficient = 0.0) — those features add no predictive value.

**Key coefficient insights from Model 3:**

| Feature | Coefficient | Interpretation |
|---------|------------|----------------|
| `Fuel_Type_Petrol` | +1351.08 | Petrol cars priced ~€1351 higher than CNG |
| `Fuel_Type_Diesel` | +1061.03 | Diesel cars priced ~€1061 higher than CNG |
| `Weight` | +27.22 | Heavier cars = higher price |
| `HP` | +24.50 | More horsepower = higher price |
| `Age_08_04` | -110.72 | Each additional month of age drops price by ~€111 |
| `Doors` | -110.57 | More doors = lower price (counterintuitive) |
| `KM` | -0.01 | More km driven = lower price |
| `Cylinders` | 0.00 | Zero weight — Lasso effectively removes this |
| `Gears` | 0.00 | Zero weight — Lasso effectively removes this |

---

# ▶️ How to Run

```bash
git clone https://github.com/Chaithanya449/toyota-corolla-price-prediction.git
cd toyota-corolla-price-prediction
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
python toyota_price_regression.py
```

> Ensure `ToyotaCorolla - MLR.csv` is in the same directory as the script.

---

# 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-lightgrey?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-blue?logo=numpy)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-LinearRegression%20%7C%20Ridge%20%7C%20Lasso-orange?logo=scikit-learn)
![SciPy](https://img.shields.io/badge/SciPy-Winsorization-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-9cf)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-yellow)

---

# 📁 Project Structure

```
toyota-corolla-price-prediction/
├── toyota_price_regression.py                      # Main script
├── ToyotaCorolla - MLR.csv     # Dataset (1,436 cars, 11 columns)
└── README.md
```

---

# 🔮 Next Steps

The `Doors` coefficient being negative is something I want to dig into — it's counterintuitive that more doors lower the price. Could be a multicollinearity issue with another feature, or just a dataset quirk. Want to run a VIF (Variance Inflation Factor) check to see which features are correlated with each other.

Also want to try hyperparameter tuning on Ridge and Lasso alpha values — right now both are set to `1.0` as default. Using `RidgeCV` or `LassoCV` would automatically find the best alpha via cross-validation.

Finally, the dataset is heavily skewed toward Petrol (1264 vs 155 Diesel vs 17 CNG) — worth checking if the model generalises well to Diesel and CNG cars specifically, or if it's effectively just a Petrol price predictor.

---

# 👤 Author

**Chaithanya Krishna** · [LinkedIn](https://www.linkedin.com/in/chaitanyakrishna-profile) · [GitHub](https://github.com/Chaithanya449)










