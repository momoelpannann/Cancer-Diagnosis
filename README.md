# Cancer-Diagnosis
This script implements a Random Forest classifier to evaluate performance dynamically across different training sizes (`N`). The workflow includes data preprocessing, dynamic data splitting, training, testing, and result analysis with visualizations.

## Key Features

### Dependencies
The following libraries are used:
- `pandas`: For data manipulation.
- `scikit-learn`: For machine learning modeling and evaluation metrics.
- `numpy`: For numerical computations.
- `matplotlib`: For visualizations.
- `time`: For measuring test time.

### Workflow
1. **Data Loading**:
   - The dataset (`data.csv`) is loaded using `pandas`.
   - Columns `id` and `Unnamed: 32` are dropped.
   - The target column `diagnosis` is encoded: `M` (malignant) as 1, `B` (benign) as 0.

2. **Dynamic Data Splitting**:
   - For each training size (`N`), `T` non-overlapping instances are dynamically selected for testing.
   - Data is shuffled before splitting into training and testing sets.

3. **Model Training and Evaluation**:
   - A `RandomForestClassifier` is trained using default hyperparameters.
   - Evaluation metrics are computed, including:
     - Accuracy
     - Precision
     - Recall
     - F1 Score
     - ROC AUC
   - Test time is recorded for each configuration.

4. **Results Storage**:
   - Instance-level predictions are stored alongside overall results.
   - Results are saved in a single output file (`final_output.txt`) in both tabular and markdown formats.

5. **Visualizations**:
   - Graphs are generated to visualize:
     - **Precision vs. N** (`precision_vs_N.png`)
     - **F1 Score vs. N** (`F1_Score_vs_N.png`)

## File Outputs
- **`final_output.txt`**: Consolidated results and predictions.
- **`precision_vs_N.png`**: Precision vs. training size graph.
- **`F1_Score_vs_N.png`**: F1 Score vs. training size graph.

## Purpose

This project is ideal for:
- Evaluating machine learning model performance under varying training sizes.
- Gaining insights into model behavior with dynamic data splitting.
- Analyzing metrics and their relationship to training size (N).
