# HR Attrition Modeling

This project uses machine learning to predict employee attrition, helping HR teams identify employees at risk of leaving and take proactive steps to improve retention.

## Objective

Build and evaluate classification models using structured HR data to:
- Predict whether an employee is likely to leave the company.
- Identify top drivers of attrition.
- Balance business needs with predictive accuracy — particularly prioritizing recall for employees likely to leave.

## Project Structure

```
HR Attrition Modeling/
│
├── data/
│ ├── ibm_attrition.csv # Original raw dataset
│ └── processed_attrition.csv # Cleaned and encoded version used for modeling
│
├── notebooks/
│ ├── eda_attrition.ipynb # Exploratory Data Analysis
│ ├── modeling_attrition.ipynb # Baseline model building and evaluation
│ ├── feature_engineering_trees.ipynb# Feature refinement for tree-based models
│ └── tuned_models_thresholding.ipynb# Hyperparameter tuning and threshold optimization
│
├── visuals/ # Folder for plots and figures (optional, to store .pngs etc.)
│
├── requirements.txt # Dependencies used in this project
├── README.md # Project overview and documentation
```


## Dataset

- Source: [IBM HR Analytics Employee Attrition & Performance – Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- 35+ features related to employee demographics, job characteristics, satisfaction scores, and performance metrics.

## Methodology

- **Preprocessing**: Encoded categorical variables, standardized numerical features, handled class imbalance using SMOTE.
- **Models Explored**:
  - Logistic Regression (baseline)
  - Decision Tree (with and without SMOTE)
  - Random Forest (with SMOTE and hyperparameter tuning)
  - XGBoost (with SMOTE and hyperparameter tuning)
- **Threshold Tuning**:
  - Used precision-recall analysis to select a custom classification threshold that maximizes F1 score and recall for attrition class (1).

## Final Model Performance (Tuned XGBoost @ Threshold 0.28)

- Accuracy: 83%
- Recall (Attrition = 1): 0.62
- Precision (Attrition = 1): 0.47
- F1 Score (Attrition = 1): 0.53
- ROC AUC: 0.76

This model offers a stronger ability to detect employees likely to leave, which is essential in a human resources context where early intervention can reduce turnover.

## Next Steps

- Build stakeholder-friendly visualizations to explain model decisions and risk drivers.
- Explore SHAP or other model explainability tools.
- Package and prepare the model for deployment or dashboard integration.