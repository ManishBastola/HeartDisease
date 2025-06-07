# Heart Disease Prediction 🫀

A machine learning project to predict the presence of heart disease using patient health data.

## Project Overview

This project builds a **classification model** to predict whether a person has heart disease based on features such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Max Heart Rate
- Exercise-induced Angina
- Oldpeak
- ST Slope

The workflow includes:

- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Engineering
- Model Building & Training
- Model Evaluation

## Dataset

- **Source**: Not specified in the notebook (commonly UCI Heart Disease Dataset)
- **Features**: 11 input features, 1 target variable (`HeartDisease`: 0 or 1)

## Models Used

- Logistic Regression  
- Random Forest Classifier  
- *(Add other models here if used, e.g. XGBoost)*

## Evaluation Metrics

- Accuracy  
- F1-Score  
- *(Add others if used: ROC-AUC, Precision, Recall, etc.)*

## Project Structure


## How to Run

1. Clone this repository:

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Install required dependencies:

    ```bash
    make install
    ```

3. Launch the Jupyter Notebook:

    ```bash
    make notebook
    ```

## How to Load the Model and Use It

If you've saved the trained model as `model.pkl` or `model.joblib`, you can load and use it like this:

### Example with `pickle`:
```python
import pickle
import numpy as np

# Load the saved model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example input
sample_input = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0]])  # example values

# Make prediction
prediction = model.predict(sample_input)

# Interpret prediction
if prediction[0] == 1:
    print("Prediction: High risk of heart disease")
else:
    print("Prediction: Low risk of heart disease")
```
