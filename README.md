# House Price Prediction Using Machine Learning

## Project Overview

This project focuses on predicting house prices using machine learning techniques. By analyzing property features such as bedrooms, bathrooms, square footage, lot size, location, and other house characteristics, the model estimates the sale price of homes in **King County, USA**.

The main objective is to build a regression model that can generalize well to unseen data and provide reasonably accurate price predictions.

---

## Dataset

The dataset contains historical house sale data with the following features:

| Feature         | Description                               |
| --------------- | ----------------------------------------- |
| `price`         | Sale price of the house (target variable) |
| `bedrooms`      | Number of bedrooms                        |
| `bathrooms`     | Number of bathrooms                       |
| `sqft_living`   | Living area square footage                |
| `sqft_lot`      | Lot size in square feet                   |
| `floors`        | Number of floors                          |
| `waterfront`    | Waterfront property (0 = No, 1 = Yes)     |
| `view`          | Quality of view (0–4)                     |
| `condition`     | Overall condition of the house            |
| `grade`         | Construction and design grade             |
| `sqft_above`    | Square footage above basement             |
| `sqft_basement` | Square footage of basement                |
| `yr_built`      | Year built                                |
| `yr_renovated`  | Year renovated                            |
| `zipcode`       | Zipcode of the house                      |
| `lat`           | Latitude                                  |
| `long`          | Longitude                                 |
| `sqft_living15` | Average living area of 15 nearby houses   |
| `sqft_lot15`    | Average lot size of 15 nearby houses      |

---

## Features Used

* **Numerical Features**
  `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `lat`, `long`, `sqft_living15`, `sqft_lot15`

* **Categorical Features**
  `floors`, `waterfront`, `view`, `condition`, `grade`, `zipcode` (encoded)

---

## Data Preprocessing

The following preprocessing steps were applied:

* Handling missing or invalid values
* Encoding categorical features (e.g., One-Hot Encoding for `zipcode`)
* Feature scaling using **StandardScaler** or **MinMaxScaler**
* Splitting the dataset into training and testing sets

---

## Machine Learning Models

Several regression models were explored and compared:

* **Linear Regression** – baseline model
* **Random Forest Regressor** – captures non-linear relationships
* **Gradient Boosting / XGBoost** – best overall performance

---

## Model Evaluation

Models were evaluated using standard regression metrics:

* **Mean Absolute Error (MAE)** – average prediction error
* **Mean Squared Error (MSE)** – error variance
* **R² Score** – proportion of variance explained

### Sample Results

| Model             | MAE     | R² Score |
| ----------------- | ------- | -------- |
| Linear Regression | ~25,000 | 0.75     |
| Random Forest     | ~15,000 | 0.88     |
| Gradient Boosting | ~12,000 | 0.90     |

---

## Making Predictions

Example of predicting the price of a new house:

```python
import pandas as pd

new_house = {
    "bedrooms": 3,
    "bathrooms": 1.75,
    "sqft_living": 1540,
    "sqft_lot": 8100,
    "floors": 1,
    "waterfront": 0,
    "view": 0,
    "condition": 4,
    "grade": 7,
    "sqft_above": 940,
    "sqft_basement": 600,
    "yr_built": 1947,
    "yr_renovated": 0,
    "zipcode": 98133,
    "lat": 47.749,
    "long": -122.351,
    "sqft_living15": 1840,
    "sqft_lot15": 8100
}

df = pd.DataFrame([new_house])
predicted_price = model.predict(df)
print("Predicted price:", predicted_price[0])
```

---

## Prediction Accuracy & Limitations ⚠️

**Important:** This model does **NOT** provide 100% accurate or guaranteed predictions.

House prices depend on many real-world factors that are **not included** in the dataset, such as:

* Interior quality and renovations
* Neighborhood demand
* School districts
* Economic and market trends

### Error Interpretation Guide

Predictions should be evaluated using **percentage error**, not exact price matching:

* **≤ 10% error** →  Good prediction
* **10–12% error** → Acceptable prediction
* **> 15% error** →  Poor prediction

A single inaccurate prediction does **not** mean the model is incorrect. However, repeated large errors indicate that the model needs further improvement.

This project is intended for **learning and experimentation purposes only** and should **not** be used for real financial or real-estate decisions.

---  
