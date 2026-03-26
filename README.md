# Supply Chain Order Regression
*Regression Model built for STAT 303-3 at Northwestern University - May 2025*

## Overview
This project focuses on predicting the delivery timeliness and completion of supply chain orders using regression techniques. The goal was to leverage multiple models in a stacking ensemble to improve predictive performance while balancing computational efficiency. 

## Methodology

### Data Preprocessing
- Cleaned and merged training and test datasets  
- Handled missing values using median imputation  
- Applied log transformations to highly skewed numeric features  
- One-hot encoded categorical features for compatibility with models  
- Binned continuous features such as days between order and due date  

### Feature Engineering
- Created time-based features like week of the year  
- Generated transformed features (log ratios, binned counts) to stabilize variance  
- Ensured features were aligned between training and test datasets  

### Model Development
- Applied a stacking ensemble with three tuned base models:  
  - Random Forest 
  - XGBoost
  - CatBoost  
- Opted for these models to balance strong predictive performance with manageable computation time  
- Experimented with other models (e.g., KNN, LightGBM) but found the three-model stack performed best  
- Each base learner was tuned with Optuna using defined hyperparameter grids  

- The final stacking model used Logistic Regression as a meta-learner  
- Evaluated using:  
  - Training accuracy and ROC AUC  
  - 5-fold cross-validation  
  - Kaggle leaderboard submission  

## Key Findings
- Stacking ensemble achieved ~86% training accuracy
- ROC AUC indicates excellent ranking ability (0.937)

## Data Sources
- `train_X.csv` and `train_y.csv` – historical order data with features and target labels  
- `public_private_X.csv` – test data used for Kaggle leaderboard submission  

## Files
- `regression_model_code.ipynb` – Full code with preprocessing, feature engineering, model tuning, and evaluation  
- `predictions_stacking.csv` – Final predictions for Kaggle leaderboard submission  
