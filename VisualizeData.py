import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = 'data/raw/creditcard.csv'

try:
    # Load the dataset
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")

    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    print("\nBasic information about the dataset:")
    df.info()

    print("\nStatistical summary of the dataset:")
    print(df.describe())

    # --- Data Visualization ---
    print("\n--- Starting Data Visualizations ---")
    print("Plots will be displayed and saved as PNG files in the script's directory.")

    # Set the style for plots
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Distribution of the target variable 'Class'
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df, palette=['#3498db', '#e74c3c'])
    plt.title('Distribution of Transaction Class (0: Legitimate, 1: Fraudulent)')
    plt.xlabel('Class')
    plt.ylabel('Number of Transactions')
    legitimate_count = df['Class'].value_counts().get(0, 0)
    fraud_count = df['Class'].value_counts().get(1, 0)
    plt.xticks([0, 1], [f'Legitimate ({legitimate_count})', f'Fraudulent ({fraud_count})'])
    class_dist_path = "class_distribution.png"
    plt.savefig(class_dist_path)
    print(f"\nClass distribution plot saved to {class_dist_path}")
    plt.show()

    print(f"\nNumber of legitimate transactions: {legitimate_count}")
    print(f"Number of fraudulent transactions: {fraud_count}")
    if df.shape[0] > 0 and fraud_count > 0 : # Avoid division by zero
        print(f"Percentage of fraudulent transactions: {(fraud_count / df.shape[0]) * 100:.4f}%")

    # 2. Distribution of 'Time' feature
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Time'], bins=100, kde=True, color='skyblue')
    plt.title('Distribution of Transaction Time')
    plt.xlabel('Time (seconds since first transaction in dataset)')
    plt.ylabel('Frequency')
    time_dist_path = "time_distribution.png"
    plt.savefig(time_dist_path)
    print(f"Time distribution plot saved to {time_dist_path}")
    plt.show()

    # 3. Distribution of 'Amount' feature
    # Plotting with and without log scale for Amount because it's highly skewed
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Amount'], bins=100, kde=True, color='lightcoral')
    plt.title('Distribution of Transaction Amount')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency')
    amount_dist_path_linear = "amount_distribution_linear.png"
    plt.savefig(amount_dist_path_linear)
    print(f"Amount distribution (linear y-axis) plot saved to {amount_dist_path_linear}")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.histplot(df['Amount'], bins=100, kde=True, color='lightcoral')
    plt.title('Distribution of Transaction Amount (Log Scale for Y-axis)')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency (Log Scale)')
    plt.yscale('log') # Using log scale for y-axis
    amount_dist_path_log = "amount_distribution_log_y.png"
    plt.savefig(amount_dist_path_log)
    print(f"Amount distribution (log y-axis) plot saved to {amount_dist_path_log}")
    plt.show()


    # 4. Distribution of 'Amount' for fraudulent vs. legitimate transactions
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='Class', y='Amount', data=df, palette=['#3498db', '#e74c3c'])
    plt.title('Transaction Amount vs. Class')
    plt.xlabel('Class (0: Legitimate, 1: Fraudulent)')
    plt.ylabel('Transaction Amount')
    plt.xticks([0, 1], [f'Legitimate', f'Fraudulent'])
    # It might be useful to see this on a log scale for Amount due to outliers
    plt.yscale('log')
    plt.suptitle('Transaction Amount vs. Class (Log Scale for Y-axis)', y=1.00)
    amount_class_boxplot_path = "amount_vs_class_boxplot_log.png"
    plt.savefig(amount_class_boxplot_path)
    print(f"Amount vs Class boxplot (log y-axis) saved to {amount_class_boxplot_path}")
    plt.show()

    # 5. Correlation Heatmap
    # For better readability, we'll show correlations with 'Class' separately
    # and then a general heatmap for a subset.
    plt.figure(figsize=(8, 12))
    heatmap_data = df.corr()[['Class']].sort_values(by='Class', ascending=False)
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Feature Correlation with Class (Target Variable)')
    correlation_class_heatmap_path = "feature_correlation_with_class.png"
    plt.savefig(correlation_class_heatmap_path)
    print(f"Feature correlation with Class heatmap saved to {correlation_class_heatmap_path}")
    plt.show()

    # General correlation heatmap for V1-V28, Amount, Time
    # This can be very large, so let's pick a smaller, more relevant subset or focus on 'Class' as above.
    # For now, the correlation with 'Class' is often the most insightful first step for classification.
    # If you want a full heatmap (can be dense):
    # plt.figure(figsize=(20, 18))
    # sns.heatmap(df.corr(), annot=False, cmap='coolwarm') # annot=False for full heatmap for readability
    # plt.title('Full Correlation Heatmap')
    # plt.savefig("full_correlation_heatmap.png")
    # plt.show()


    print("\n--- Data Visualization Complete ---")
    print("Generated plot files (check your script's directory):")
    print(f"- {class_dist_path}")
    print(f"- {time_dist_path}")
    print(f"- {amount_dist_path_linear}")
    print(f"- {amount_dist_path_log}")
    print(f"- {amount_class_boxplot_path}")
    print(f"- {correlation_class_heatmap_path}")


except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please ensure you have downloaded 'creditcard.csv' from Kaggle")
    print("and placed it in the correct directory, OR update the 'file_path' variable in the script.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check the file format, integrity, and that all libraries are installed.")
