# Medical Insurance Cost Estimator & Risk Analyzer

A Streamlit application that estimates annual medical insurance costs from age, gender, BMI, number of dependents, smoking status, and region. The app presents a step-by-step estimate, monthly equivalent, BMI/risk category, and a non-smoker cost comparison for smokers.

## Features

- Six-step guided input wizard
- Annual and monthly insurance cost estimates
- BMI and lifestyle risk classification
- Non-smoker comparison and potential annual savings
- Interactive summary of submitted information
- Linear regression model with saved scaler and model artifacts

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Joblib

## Project Structure

```text
Medical-insurance-cost-estimator/
├── app.py                         # Streamlit application
├── assets/                        # Background and interface images
├── data_cleaning/
│   ├── insurance.csv              # Original dataset
│   ├── insurance_cleaned.csv      # Cleaned dataset
│   └── data_cleaning.ipynb        # Data preparation notebook
├── data_visualization/
│   └── data_visualization.ipynb  # Exploratory analysis notebook
├── models/
│   ├── models.ipynb               # Model training and evaluation notebook
│   └── saved_models/
│       ├── linear_regression_best_model.pkl
│       └── scaler.pkl
└── requirements.txt
```

## Model

The selected model is Linear Regression, with an R² score of **0.9057**. The app also compares the prediction with a non-smoker scenario by changing the smoking and obesity-smoking interaction inputs while keeping the other values unchanged.

> Estimates are generated from a historical dataset and are for educational purposes only. They should not be treated as an insurance quote or medical/financial advice.

## Setup

Python 3.9 or newer is recommended.

```powershell
# From the project directory
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the App

```powershell
streamlit run app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`.

## Example Inputs

| Input | Example |
|---|---|
| Age | 30 years |
| Gender | Male or Female |
| Dependents | 0 |
| BMI | 27.5 |
| Smoking status | Non-smoker or Smoker |
| Region | Northeast, Northwest, Southeast, or Southwest |

## Development Notebooks

- `data_cleaning/data_cleaning.ipynb` prepares the source dataset.
- `data_visualization/data_visualization.ipynb` explores cost patterns and feature relationships.
- `models/models.ipynb` compares regression models and saves the best model and scaler.
