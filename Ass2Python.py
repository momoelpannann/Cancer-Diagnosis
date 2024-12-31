# Mohamed Elpannann 40251343
# Assignment 2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import shuffle
import numpy as np
import time
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Preprocess the data
data = data.drop(columns=['id', 'Unnamed: 32'])
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Define N values
N_values = [40, 140, 240, 340, 440]

# Initialize results storage
results = []
all_predictions = []

# Evaluate the model for each training size (N)
for N in N_values:
    T = N // 4

    # Shuffle and split the data dynamically
    data_shuffled = shuffle(data, random_state=None)
    
    # Select N instances for training
    train_data = data_shuffled.iloc[:N]
    
    # Select T non-overlapping instances for testing
    remaining_data = data_shuffled.iloc[N:]  # Exclude training data
    test_data = remaining_data.iloc[:T]     # Take the next T instances
    
    # Extract features and labels for training and testing
    X_train = train_data.drop(columns=['diagnosis'])
    y_train = train_data['diagnosis']
    X_test = test_data.drop(columns=['diagnosis'])
    y_test = test_data['diagnosis']
    
    # Train the Random Forest model with default parameters
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Predict and measure accuracy
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    test_time = end_time - start_time

    # Save instance-level predictions
    predictions = pd.DataFrame({
        'N': N,
        'True Label': y_test.values,
        'Predicted Label': y_pred,
        'Accuracy': [accuracy] * len(y_test)
    })
    all_predictions.append(predictions)
    
    # Store results for the current N
    results.append({
        'N': N,
        'T': T,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'Test Time (s)': test_time
    })

# Combine instance-level predictions
all_predictions_df = pd.concat(all_predictions, ignore_index=True)

# Save predictions and results to a single output file
output_file = 'final_output.txt'
with open(output_file, 'w') as f:
    f.write("Random Forest Results with Default Hyperparameters\n")
    f.write("\nOverall Results:\n")
    results_df = pd.DataFrame(results)
    f.write(results_df.to_string(index=False))
    
    # Add a table for instance-level predictions
    f.write("\n\nInstance-Level Predictions:\n")
    f.write(all_predictions_df.to_markdown(index=False, tablefmt="grid"))

# Generate and save graphs
plt.figure(figsize=(10, 6))
plt.plot(results_df['N'], results_df['Precision'], marker='o', label='Precision')
plt.title('Precision vs. N')
plt.xlabel('Number of Training Instances (N)')
plt.ylabel('Precision')
plt.legend()
plt.grid()
plt.savefig('precision_vs_N.png')

plt.figure(figsize=(10, 6))
plt.plot(results_df['N'], results_df['F1 Score'], marker='o', label='F1 Score')
plt.title('F1 Score vs. N')
plt.xlabel('Number of Training Instances (N)')
plt.ylabel('F1 Score')
plt.legend()
plt.grid()
plt.savefig('F1_Score_vs_N.png')