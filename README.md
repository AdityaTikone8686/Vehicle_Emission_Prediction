# Vehicle Emission Prediction 🚗💨

A machine learning regression project that predicts vehicle CO₂ emissions in grams per kilometer (g/km) using vehicle specifications, fuel-consumption data, and categorical vehicle information.

## 📌 Project Overview

Vehicle CO₂ emissions are influenced by factors such as engine size, number of cylinders, fuel consumption, vehicle class, transmission, and fuel type.

This project applies machine learning regression techniques to predict:

**CO₂ emissions (g/km)**

The project follows a complete machine learning workflow:

**Data Exploration → EDA → Feature Selection → Train/Test Split → Preprocessing → Model Training → Evaluation → Model Comparison → Model Saving**

## 🎯 Objective

The objective of this project is to develop and evaluate machine learning models capable of predicting vehicle CO₂ emissions from vehicle specifications and fuel-consumption features.

## 📊 Dataset

The dataset contains **601 vehicle records** and **15 columns**.

### Features

* Model year
* Make
* Model
* Vehicle class
* Engine size (L)
* Cylinders
* Transmission
* Fuel type
* City (L/100 km)
* Highway (L/100 km)
* Combined (L/100 km)
* Combined (mpg)
* CO₂ emissions (g/km)
* CO₂ rating
* Smog rating

### Target Variable

**CO₂ emissions (g/km)**

The target represents the amount of carbon dioxide emitted by a vehicle in grams per kilometer.

## 🔍 Exploratory Data Analysis

The notebook includes:

* Dataset shape and structure analysis
* Missing-value analysis
* Duplicate-value checking
* Descriptive statistics
* Unique-value analysis
* CO₂ emissions distribution
* Engine size vs CO₂ emissions
* Combined fuel consumption vs CO₂ emissions
* Feature correlation heatmap
* Model performance comparison
* Actual vs predicted visualization

## 🧹 Feature Selection

The initial numerical model used:

* Engine size (L)
* Cylinders
* City (L/100 km)
* Highway (L/100 km)
* Combined (L/100 km)
* Combined (mpg)

`CO₂ rating` and `Smog rating` were excluded from the model features because they contain rating information related to vehicle emissions and could introduce information overlap or target leakage.

A reduced feature experiment was also performed by removing `Combined (mpg)` because it has a very strong relationship with `Combined (L/100 km)`.

## ⚙️ Data Preprocessing

For the full-feature experiments, categorical variables were converted into numerical representations using **One-Hot Encoding**.

Categorical features included:

* Make
* Model
* Vehicle class
* Transmission
* Fuel type

Scikit-learn's `ColumnTransformer` and `Pipeline` were used to combine preprocessing and model training.

The preprocessing pipeline also uses:

```python
OneHotEncoder(handle_unknown="ignore")
```

so that unseen categories can be handled when making predictions on new data.

## 🤖 Machine Learning Models

Three regression algorithms were evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Different feature configurations were tested to understand how feature selection and categorical information affect model performance.

## 📈 Model Evaluation

The models were evaluated using four regression metrics:

* **MAE** — Mean Absolute Error
* **MSE** — Mean Squared Error
* **RMSE** — Root Mean Squared Error
* **R²** — Coefficient of Determination

### Model Comparison

| Model                     | MAE ↓ |   MSE ↓ | RMSE ↓ |   R² ↑ |
| ------------------------- | ----: | ------: | -----: | -----: |
| Linear Regression         | 3.956 |  40.550 |  6.368 | 0.9921 |
| Reduced Linear Regression | 3.109 |  27.237 |  5.219 | 0.9947 |
| Full Linear Regression    | 3.857 |  28.396 |  5.329 | 0.9945 |
| Decision Tree             | 4.640 | 306.969 | 17.521 | 0.9404 |
| Reduced Decision Tree     | 4.665 | 307.060 | 17.523 | 0.9404 |
| Full Decision Tree        | 2.661 |  57.736 |  7.598 | 0.9888 |
| Random Forest             | 3.134 |  71.417 |  8.451 | 0.9861 |
| Reduced Random Forest     | 3.116 |  69.889 |  8.360 | 0.9864 |
| Full Random Forest        | 2.662 |  61.852 |  7.865 | 0.9880 |

### Key Observations

The experiments demonstrate that model performance varies with both the algorithm and feature configuration.

The **Reduced Linear Regression** model achieved an R² of **0.9947** with an RMSE of **5.219 g/km** on the test set.

The **Full Decision Tree** achieved an MAE of **2.661 g/km**, while the **Full Random Forest** achieved a very similar MAE of **2.662 g/km**.

These results demonstrate the importance of comparing multiple algorithms and feature configurations instead of evaluating a single model in isolation.

## 📊 Feature Importance

Feature importance was examined using the Random Forest model.

Among the reduced numerical features, **Combined (L/100 km)** contributed the largest relative feature importance in the trained Random Forest model.

This is consistent with the strong relationship between fuel consumption and CO₂ emissions.

Feature importance is model-specific and should not be interpreted as proof of a causal relationship.

## 💾 Saved Model

The trained Full Decision Tree pipeline is saved as:

```text
models/vehicle_emission_model.pkl
```

The saved pipeline contains the preprocessing steps and trained model together, allowing the model to be loaded later without manually rebuilding the preprocessing process.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Google Colab
* Jupyter Notebook
* Git & GitHub

## 📁 Project Structure

```text
vehicle-emission-prediction/
│
├── data/
│   └── co2_emission_dataset.csv
│
├── models/
│   └── vehicle_emission_model.pkl
│
├── notebooks/
│   └── vehicle_emission_prediction.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd vehicle-emission-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Open:

```text
notebooks/vehicle_emission_prediction.ipynb
```

Run the notebook cells sequentially to reproduce the analysis and model experiments.

## 🔮 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* Additional feature engineering
* Testing additional regression algorithms
* Building a prediction interface
* Deploying the model as an API
* Evaluating the model on additional datasets
* Monitoring model performance on new data

## 👨‍💻 Author

**Aditya Tikone**
PCET's PIMPRI CHINCHWAD UNIVERSITY
B.Tech Computer Science & Engineering

Interested in Artificial Intelligence, Machine Learning.
