# Credit Card Fraud Detection

## Project Overview
This project aims to build a machine learning model that can predict fraudulent credit card transactions based on historical transaction data. The dataset contains various features of transactions, such as time, amount, and anonymized information about the customer and transaction. The goal is to classify whether a transaction is fraudulent or legitimate.

## Dataset
The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) from Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where there are 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.

### Features
- Time: Number of seconds elapsed between this transaction and the first transaction in the dataset
- Amount: Transaction amount
- V1-V28: Anonymized features (result of a PCA transformation)
- Class: Target variable (1 for fraudulent transactions, 0 for legitimate)

## Project Structure
```
Credit Card Fraud Detection/
│
├── data/
│   └── raw/
│       └── creditcard.csv
│
├── VisualizeData.py          # Script for data visualization and exploration
├── ModelTraining.py          # Script for training and evaluating models
├── ModelPrediction.py        # Script for making predictions with the trained model
├── README.md                 # Project documentation
│
├── class_distribution.png    # Visualization outputs
├── time_distribution.png
├── amount_distribution_linear.png
├── amount_distribution_log_y.png
├── amount_vs_class_boxplot_log.png
├── feature_correlation_with_class.png
├── model_comparison.png
├── decision_tree_confusion_matrix.png
├── random_forest_confusion_matrix.png
├── xgboost_confusion_matrix.png
├── neural_network_confusion_matrix.png
│
├── random_forest_model.joblib  # Example of saved model 
└── standard_scaler.joblib      # Saved scaler for preprocessing new data
```

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Setup
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Unix or MacOS
   ```

3. Install the required packages:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib
   ```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) and place it in the `data/raw/` directory.

## Usage

### Data Visualization
To explore and visualize the dataset:
```
python VisualizeData.py
```
This script will generate various plots to help understand the data distribution and relationships.

### Model Training
To train and evaluate the machine learning models:
```
python ModelTraining.py
```
This script will:
1. Load and preprocess the dataset
2. Handle the class imbalance using SMOTE
3. Train multiple models (Decision Tree, Random Forest, XGBoost, Neural Network)
4. Evaluate and compare the models
5. Save the best performing model

### Making Predictions
To use the trained model for making predictions:
```
python ModelPrediction.py
```
This script demonstrates how to:
1. Load the saved model and scaler
2. Preprocess new data
3. Make predictions on transactions
4. Evaluate the prediction results

## Approach and Methodology

### Data Preprocessing
- Feature scaling using StandardScaler
- Handling imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique)

### Models Implemented
1. **Decision Tree**: A simple model that makes decisions based on feature values
2. **Random Forest**: An ensemble of decision trees that improves performance and reduces overfitting
3. **XGBoost**: A gradient boosting algorithm known for its performance and efficiency
4. **Neural Network**: A multi-layer perceptron with hidden layers

### Evaluation Metrics
Since the dataset is highly imbalanced, we use metrics beyond accuracy:
- Precision: Ability of the model to avoid false positives
- Recall: Ability of the model to find all fraudulent transactions
- F1-score: Harmonic mean of precision and recall
- ROC-AUC: Area under the ROC curve, measuring the model's ability to distinguish between classes

## Results and Conclusions

The models are evaluated based on their performance on the test set. The F1-score is particularly important for this imbalanced dataset, as it balances precision and recall.

The Random Forest and XGBoost models typically perform well on this type of problem due to their ability to handle imbalanced data and complex relationships between features.

For detailed results, run the `ModelTraining.py` script, which will output performance metrics for each model and generate a comparison plot.