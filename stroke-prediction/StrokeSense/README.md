# Stroke Prediction Project

## Overview
This project predicts the likelihood of strokes using machine learning models based on patient health data. It includes a Python based pipeline for data preprocessing, model training, and evaluation, as well as a framework for deploying the best-performing model.

## Files in This Repository

### 1. `stroke-predict-model.ipynb`
**Purpose**: Handles data cleaning and preprocessing tasks.
- **Contents & Steps**:
  - **Data Loading & Exploration**: Reads the stroke dataset and performs exploratory data analysis (EDA) to understand distributions and correlations.
  - **Data Cleaning**: Handles missing values, removes duplicates, and validates data integrity.
  - **Feature Engineering**: Creates new features (e.g., BMI categories, age groups) to improve model performance.

**Model**: Trains and evaluates multiple machine learning models for stroke prediction.
- **Contents & Steps**:
  - **Model Selection**: Implements and compares models such as Random Forest (RF), Extra Trees (ET), and Quadratic Discriminant Analysis (QDA).
  - **Hyperparameter Tuning**: Optimises model parameters using grid search or random search.
  - **Evaluation Metrics**: Reports precision, recall, F1 score, and accuracy for each model.

### 3. `StrokeSense folder`
**Purpose**: A Streamlit web app for interactive stroke risk prediction.
- **Features**:
  - Accepts user input for patient health parameters (e.g., age, BMI, health condition and etc).
  - Predicts stroke risk in real-time and displays results in a user-friendly interface.
  - Includes “What If” Scenario Explorer, Let users play with different values to see how their risk changes.
  - Example:  “What if I stop smoking?” → Stroke risk drops from 48% to 38%
- Interactive sliders or dropdowns to modify features

### 4. `healthcare-dataset-stroke-data.csv`
**Purpose**: The dataset used for training and evaluation.
- **Source**: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset 
- **Description**: Contains patient demographic and health information, including age, gender, BMI, glucose levels, and etc.

## Project Workflow

### 1. Data Preprocessing
- Load and explore the dataset.
- Clean and preprocess the data to handle missing values and outliers.
- Engineer new features to enhance model performance.

### 2. Model Training & Evaluation
- Train multiple models (RF, ET, QDA) and compare their performance.
- Evaluate models using precision, recall, F1 score, and accuracy.
- Select the best-performing model for deployment.

### 3. Deployment
- Save the best model and integrate it into the Streamlit app for real-time predictions.



## Notes
- The QDA model currently provides the best precision, recall, and F1 score for stroke prediction compare to other models.
- The app includes educational content to help users understand the impact of each feature on