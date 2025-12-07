import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import os

def get_decision_rule(target_r0, noise_var, csv_path="./massive_output.csv"):
    """
    Reads the CSV, filters for a specific r0_ref AND noise_var.
    Splits data 80/20 (Train/Test).
    Returns: (threshold, cv_accuracy)
    If a rule cannot be formed (e.g. no data or single class), returns (None, None) or (None, Acc).
    """
    
    # 1. Load Data
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return None, None

    df = pd.read_csv(csv_path)

    # 2. Filter for the specific r0_ref AND the specific noise_var
    subset = df[(df['r0_ref'] == target_r0) & (df['noise_var'] == noise_var)]
    
    # --- Edge Case Handling ---
    if subset.empty:
        # print(f"No data found for r0_ref = {target_r0} and noise_var = {noise_var}")
        return None, None
    
    unique_signals = subset['sent_signal'].unique()
    if len(unique_signals) == 1:
        # print(f"For r0 {target_r0}, noise {noise_var}: Always predict {unique_signals[0]}")
        return None, 1.0

    # 3. Prepare Data
    X = subset[['power1_in_bucket']]
    y = subset['sent_signal']

    # 4. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Initialize Classifier
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)

    # 6. Cross Validation (on Training Data)
    cv_folds = 5 if len(X_train) > 20 else 2
    try:
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv_folds)
        mean_cv_acc = cv_scores.mean()
    except ValueError:
        mean_cv_acc = 0.0

    # 7. Fit on Training Data to get the Threshold
    clf.fit(X_train, y_train)
    
    if clf.tree_.node_count > 1:
        threshold = clf.tree_.threshold[0]
        return threshold, mean_cv_acc
    else:
        return None, mean_cv_acc

def process_all_conditions(csv_path="./massive_output.csv", output_path="./decision_rules_summary.csv"):
    """
    Iterates over all unique (r0_ref, noise_var) pairs in the CSV,
    calculates the decision rule for each, and saves the results to a new CSV.
    """
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Check columns
    if 'r0_ref' not in df.columns or 'noise_var' not in df.columns:
        print("Error: CSV missing required columns 'r0_ref' or 'noise_var'")
        return

    # Get unique combinations
    conditions = df[['r0_ref', 'noise_var']].drop_duplicates().sort_values(by=['r0_ref', 'noise_var'])
    
    results = []
    print(f"--- Processing {len(conditions)} unique conditions ---")

    for _, row in conditions.iterrows():
        r0 = row['r0_ref']
        noise = row['noise_var']
        
        # Call the single-rule function
        thresh, acc = get_decision_rule(r0, noise, csv_path)
        
        if thresh is not None:
            print(f"Processed r0={r0}, noise={noise}: Threshold={thresh:.4f}, Acc={acc:.2%}")
            results.append({
                'r0_ref': r0,
                'noise_var': noise,
                'threshold': thresh,
                'cv_accuracy': acc
            })
        else:
            print(f"Skipping r0={r0}, noise={noise} (No valid split or single class)")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"\nSuccess! Saved {len(results)} rules to '{output_path}'")
        print(results_df.head())
    else:
        print("\nNo valid rules were generated.")

def plot_ber_vs_noise(summary_csv_path="./decision_rules_summary.csv", pixels_in_bucket=1.0):
    """
    Reads the summary CSV and plots BER (1 - accuracy) vs Total Bucket Noise Variance.
    
    Args:
        pixels_in_bucket (float): Scaling factor. 
            If 'noise_var' in CSV is per-pixel variance, this should be the number of pixels in the bucket.
            If 'noise_var' is already total variance, leave as 1.0.
    """
    if not os.path.exists(summary_csv_path):
        print(f"Summary file {summary_csv_path} not found. Please run process_all_conditions() first.")
        return

    df = pd.read_csv(summary_csv_path)
    
    # Calculate BER
    df['BER'] = 1 - df['cv_accuracy']
    
    # Adjust noise variance to represent the total variance in the bucket
    # Var(Sum of N pixels) = N * Var(pixel) (assuming independence)
    df['bucket_noise_var'] = df['noise_var'] * pixels_in_bucket
    
    plt.figure(figsize=(10, 6))
    
    # Get unique r0_ref values to create separate curves
    unique_r0s = sorted(df['r0_ref'].unique())
    
    for r0 in unique_r0s:
        subset = df[df['r0_ref'] == r0].sort_values(by='bucket_noise_var')
        plt.plot(subset['bucket_noise_var'], subset['BER'], marker='o', label=f'r0_ref = {r0}')
    
    plt.title('BER (1 - Accuracy) vs. Total Noise Variance in Bucket')
    plt.xlabel('Total Noise Variance (noise_var * pixels_in_bucket)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.xscale('log') 
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_power_distribution(target_r0, noise_var, csv_path="./massive_output.csv"): 
    """
    Plots the histogram of power in bucket for a specific condition
    and overlays the decision threshold.
    """
    if not os.path.exists(csv_path):
        print(f"CSV {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    subset = df[(df['r0_ref'] == target_r0) & (df['noise_var'] == noise_var)]
    
    if subset.empty:
        print(f"No data for r0={target_r0}, noise={noise_var}")
        return

    # Get the decision rule for this specific subset
    threshold, acc = get_decision_rule(target_r0, noise_var, csv_path)
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms for Signal vs No Signal
    # We use density=True to compare shapes, or standard counts.
    plt.hist(subset[subset['sent_signal'] == False]['power1_in_bucket'], 
            bins=30, alpha=0.5, label='Signal: False (0)', color='red', edgecolor='black')
    plt.hist(subset[subset['sent_signal'] == True]['power1_in_bucket'], 
            bins=30, alpha=0.5, label='Signal: True (1)', color='green', edgecolor='black')
    
    # Add vertical line for threshold
    if threshold is not None:
        plt.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Decision Rule ({threshold:.4f})')
        plt.title(f'Power Distribution (r0={target_r0}, noise={noise_var})\nAccuracy: {acc:.2%}')
    else:
        plt.title(f'Power Distribution (r0={target_r0}, noise={noise_var})\nNo valid split found')

    plt.xlabel('Power in Bucket')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- Dummy Data Generation (Updated) ---
if not os.path.exists("./massive_output.csv"):
    print("Generating dummy data for testing...")
    np.random.seed(42)
    dummy_df = pd.DataFrame({
        'r0_ref': np.random.choice([0.075, 0.2, 0.3], 1000),
        'noise_var': np.random.choice([1e4, 1e5, 1e6], 1000),
        'power1_in_bucket': np.random.uniform(0, 10, 1000)
    })
    
    # Create synthetic logic
    conds = [
        (dummy_df['r0_ref'] == 0.075) & (dummy_df['power1_in_bucket'] > 3),
        (dummy_df['r0_ref'] == 0.2) & (dummy_df['power1_in_bucket'] > 7),
        (dummy_df['r0_ref'] == 0.3) & (dummy_df['power1_in_bucket'] > 5) 
    ]
    dummy_df['sent_signal'] = np.select(conds, [True, True, True], default=False)
    # Add some random noise to labels to make accuracy < 100%
    flip_indices = np.random.choice(dummy_df.index, size=int(0.1 * len(dummy_df)), replace=False)
    dummy_df.loc[flip_indices, 'sent_signal'] = ~dummy_df.loc[flip_indices, 'sent_signal']
    
    dummy_df.to_csv("./massive_output.csv", index=False)

if __name__ == "__main__":
    # 1. Process ALL conditions and save results to CSV
    process_all_conditions()
    
    # 2. Plot the BER curves
    # IMPORTANT: Update pixels_in_bucket to the actual number of pixels used in your simulation bucket.
    # Default is 1.0 (assuming noise_var is already total bucket variance or 1 pixel bucket).
    plot_ber_vs_noise(pixels_in_bucket=9e-6)
    
    # 3. Example of plotting distribution for a specific case
    # You can pick a specific r0 and noise from your CSV to visualize
    print("\nDisplaying distribution for example condition...")
    # Example: Picking the first combination found in the CSV
    df_temp = pd.read_csv("./massive_output.csv")
    if not df_temp.empty:
        sample_r0 = df_temp['r0_ref'].iloc[1]
        sample_noise = df_temp['noise_var'].iloc[1]
        plot_power_distribution(sample_r0, sample_noise)