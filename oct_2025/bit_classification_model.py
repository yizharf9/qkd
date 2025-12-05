import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import os

def get_decision_rule(target_r0, noise_var, csv_path="./massive_output.csv"):
    """
    Reads the CSV, filters for a specific r0_ref AND noise_var.
    Splits data 80/20 (Train/Test).
    Returns the threshold (from Train) and the Cross-Validation Accuracy (on Train).
    """
    
    # 1. Load Data
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return None

    df = pd.read_csv(csv_path)

    # 2. Filter for the specific r0_ref AND the specific noise_var
    subset = df[(df['r0_ref'] == target_r0) & (df['noise_var'] == noise_var)]
    
    # --- Edge Case Handling ---
    if subset.empty:
        return f"No data found for r0_ref = {target_r0} and noise_var = {noise_var}"
    
    unique_signals = subset['sent_signal'].unique()
    if len(unique_signals) == 1:
        return f"For r0 {target_r0}, noise {noise_var}: Always predict {unique_signals[0]}"

    # 3. Prepare Data
    X = subset[['power1_in_bucket']]
    y = subset['sent_signal']

    # 4. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Initialize Classifier
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)

    # 6. Cross Validation (on Training Data)
    # Using 5-fold CV to get a robust accuracy metric
    # Note: Requires enough samples; for tiny datasets, reduce cv folds.
    cv_folds = 5 if len(X_train) > 20 else 2
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv_folds)
    mean_cv_acc = cv_scores.mean()

    # 7. Fit on Training Data to get the Threshold
    clf.fit(X_train, y_train)
    threshold = clf.tree_.threshold[0]
    
    # Determine which side of the threshold is True/False
    test_val = threshold - 0.001
    prediction_below = clf.predict([[test_val]])[0]
    
    if prediction_below == False:
        rule_desc = f"If power > {threshold} -> Signal TRUE"
    else:
        rule_desc = f"If power <= {threshold} -> Signal TRUE"
    print(f"Threshold: {threshold} | CV Accuracy: {mean_cv_acc:.1%} | {rule_desc}")
    return threshold , mean_cv_acc

# 1. Call the function for specific values
get_decision_rule(0.075,1e4)


# print(get_decision_rule(0.2))
# print(get_decision_rule(0.99)) # Example of missing data