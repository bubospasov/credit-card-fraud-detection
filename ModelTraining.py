import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# File path
file_path = 'data/raw/creditcard.csv'

def load_and_explore_data(file_path):
    """
    Load and perform basic exploration of the dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"\nClass distribution:")
    class_counts = df['Class'].value_counts()
    print(class_counts)
    print(f"Percentage of fraudulent transactions: {(class_counts[1] / df.shape[0]) * 100:.4f}%")
    
    return df

def preprocess_data(df):
    """
    Preprocess the data for model training
    """
    print("\nPreprocessing data...")
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def handle_imbalanced_data(X_train, y_train, method='smote'):
    """
    Apply techniques to handle imbalanced data
    """
    print(f"\nHandling imbalanced data using {method}...")
    
    if method == 'smote':
        # Synthetic Minority Over-sampling Technique
        smote = SMOTE(random_state=RANDOM_STATE)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
    elif method == 'undersampling':
        # Random Under-sampling
        rus = RandomUnderSampler(random_state=RANDOM_STATE)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        
    elif method == 'hybrid':
        # Combination of over-sampling and under-sampling
        over = SMOTE(sampling_strategy=0.1, random_state=RANDOM_STATE)
        under = RandomUnderSampler(sampling_strategy=0.5, random_state=RANDOM_STATE)
        steps = [('over', over), ('under', under)]
        pipeline = Pipeline(steps=steps)
        X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
    
    else:
        # No resampling
        X_resampled, y_resampled = X_train, y_train
    
    print(f"Resampled data shape: {X_resampled.shape}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Train and evaluate a model
    """
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
    
    return {
        'model': model,
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def compare_models(results):
    """
    Compare the performance of different models
    """
    print("\nComparing Models:")
    
    # Create a DataFrame with the results
    df_results = pd.DataFrame({
        'Model': [r['model_name'] for r in results],
        'Accuracy': [r['accuracy'] for r in results],
        'Precision': [r['precision'] for r in results],
        'Recall': [r['recall'] for r in results],
        'F1 Score': [r['f1'] for r in results],
        'ROC AUC': [r['roc_auc'] for r in results]
    })
    
    print(df_results)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    df_results_melted = pd.melt(df_results, id_vars=['Model'], value_vars=metrics, var_name='Metric', value_name='Score')
    
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_results_melted)
    plt.title('Model Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    
    # Find the best model based on F1 score (good for imbalanced datasets)
    best_model_idx = df_results['F1 Score'].idxmax()
    best_model = df_results.iloc[best_model_idx]
    print(f"\nBest model based on F1 Score: {best_model['Model']} with F1 Score: {best_model['F1 Score']:.4f}")
    
    return results[best_model_idx]

def save_model(model_result, scaler):
    """
    Save the best model and scaler for future use
    """
    model = model_result['model']
    model_name = model_result['model_name'].replace(' ', '_').lower()
    
    # Save the model
    model_filename = f"{model_name}_model.joblib"
    joblib.dump(model, model_filename)
    
    # Save the scaler
    scaler_filename = "standard_scaler.joblib"
    joblib.dump(scaler, scaler_filename)
    
    print(f"\nBest model saved as {model_filename}")
    print(f"Scaler saved as {scaler_filename}")

def main():
    try:
        # Load and explore the data
        df = load_and_explore_data(file_path)
        
        # Preprocess the data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        # Handle imbalanced data
        X_train_resampled, y_train_resampled = handle_imbalanced_data(X_train, y_train, method='smote')
        
        # Define models
        models = [
            (DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'), "Decision Tree"),
            (RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced'), "Random Forest"),
            (XGBClassifier(random_state=RANDOM_STATE, scale_pos_weight=len(y_train) - sum(y_train) / sum(y_train)), "XGBoost"),
            (MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=RANDOM_STATE), "Neural Network")
        ]
        
        # Train and evaluate models
        results = []
        for model, model_name in models:
            result = train_and_evaluate_model(model, X_train_resampled, y_train_resampled, X_test, y_test, model_name)
            results.append(result)
        
        # Compare models
        best_model_result = compare_models(results)
        
        # Save the best model
        save_model(best_model_result, scaler)
        
        print("\nModel training and evaluation completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure you have downloaded 'creditcard.csv' from Kaggle")
        print("and placed it in the correct directory, OR update the 'file_path' variable in the script.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()