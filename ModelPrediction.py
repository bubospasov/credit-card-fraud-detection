import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def load_model_and_scaler(model_path, scaler_path):
    """
    Load the trained model and scaler
    """
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    print(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def preprocess_new_data(data, scaler):
    """
    Preprocess new data for prediction
    """
    # If data is a DataFrame, ensure it has the right format
    if isinstance(data, pd.DataFrame):
        # Make sure 'Class' column is not in the features if present
        if 'Class' in data.columns:
            X = data.drop('Class', axis=1)
            y = data['Class']  # Save the true labels if available
        else:
            X = data
            y = None
    else:
        # If data is a numpy array or list, convert to DataFrame
        X = pd.DataFrame(data)
        y = None
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    return X_scaled, y

def predict_transactions(model, X_scaled):
    """
    Make predictions on new transactions
    """
    # Get class predictions
    y_pred = model.predict(X_scaled)
    
    # Get probability estimates
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    return y_pred, y_pred_proba

def evaluate_predictions(y_true, y_pred, y_pred_proba):
    """
    Evaluate the predictions if true labels are available
    """
    if y_true is None:
        print("No true labels available for evaluation.")
        return
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Predictions')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("prediction_confusion_matrix.png")
    plt.show()

def main():
    try:
        # Paths to the saved model and scaler
        # Note: Update these paths with your actual model name
        model_path = "random_forest_model.joblib"  # Example, use your best model
        scaler_path = "standard_scaler.joblib"
        
        # Load the model and scaler
        model, scaler = load_model_and_scaler(model_path, scaler_path)
        
        # Load new data for prediction
        # For demonstration, we'll use the test data from the original dataset
        print("\nLoading new data for prediction...")
        file_path = 'data/raw/creditcard.csv'
        df = pd.read_csv(file_path)
        
        # For demonstration purposes, let's use a small sample
        # In a real scenario, this would be new, unseen data
        sample_size = 1000
        df_sample = df.sample(sample_size, random_state=42)
        
        # Preprocess the data
        X_scaled, y_true = preprocess_new_data(df_sample, scaler)
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred, y_pred_proba = predict_transactions(model, X_scaled)
        
        # Add predictions to the sample DataFrame for inspection
        df_sample['Predicted_Class'] = y_pred
        df_sample['Fraud_Probability'] = y_pred_proba
        
        # Show sample of predictions
        print("\nSample of predictions:")
        print(df_sample[['Class', 'Predicted_Class', 'Fraud_Probability']].head(10))
        
        # Evaluate predictions
        if y_true is not None:
            print("\nEvaluating predictions...")
            evaluate_predictions(y_true, y_pred, y_pred_proba)
        
        # Example of how to use the model for a single transaction
        print("\nExample: Predicting a single transaction")
        # Get a random transaction (for demonstration)
        single_transaction = df_sample.drop(['Class', 'Predicted_Class', 'Fraud_Probability'], axis=1).iloc[0:1]
        
        # Preprocess and predict
        X_single_scaled, _ = preprocess_new_data(single_transaction, scaler)
        single_pred, single_prob = predict_transactions(model, X_single_scaled)
        
        print(f"Transaction details: {single_transaction.values}")
        print(f"Prediction: {'Fraudulent' if single_pred[0] == 1 else 'Legitimate'}")
        print(f"Fraud probability: {single_prob[0]:.4f}")
        
        print("\nPrediction demonstration completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have trained the model and the model files exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()