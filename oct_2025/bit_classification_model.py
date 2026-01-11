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
    Returns: (threshold, cv_accuracy_mean, cv_accuracy_std, mean_signal_power)
    """
    
    # 1. Load Data
    if not os.path.exists(csv_path):
        return None, None, None, None

    df = pd.read_csv(csv_path)

    # 2. Filter for the specific r0_ref AND the specific noise_var
    subset = df[(df['r0_ref'] == target_r0) & (df['noise_var'] == noise_var)]
    
    if subset.empty:
        return None, None, None, None
    
    # Calculate Mean Signal Power from this specific subset
    signal_present = subset[subset['sent_signal'] == True]
    if not signal_present.empty:
        current_mean_power = signal_present['power1_in_bucket'].mean()
    else:
        current_mean_power = subset['power1_in_bucket'].mean()

    unique_signals = subset['sent_signal'].unique()
    if len(unique_signals) == 1:
        # Single class: Accuracy is 100% (if matching) or 0%, Std Dev is 0
        return None, 1.0, 0.0, current_mean_power

    # 3. Prepare Data
    X = subset[['power1_in_bucket']]
    y = subset['sent_signal']

    # 4. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Initialize Classifier
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)

    # 6. Cross Validation
    cv_folds = 5 if len(X_train) > 20 else 2
    try:
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv_folds)
        mean_cv_acc = cv_scores.mean()
        std_cv_acc = cv_scores.std() # Calculate variance/std of the folds
    except ValueError:
        mean_cv_acc = 0.0
        std_cv_acc = 0.0

    # 7. Fit on Training Data
    clf.fit(X_train, y_train)
    
    if clf.tree_.node_count > 1:
        threshold = clf.tree_.threshold[0]
        return threshold, mean_cv_acc, std_cv_acc, current_mean_power
    else:
        return None, mean_cv_acc, std_cv_acc, current_mean_power

def process_all_conditions(csv_path="./massive_output.csv", output_path="./decision_rules_summary.csv"):
    """
    Iterates over all unique (r0_ref, noise_var) pairs.
    """
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    if 'r0_ref' not in df.columns or 'noise_var' not in df.columns:
        print("Error: CSV missing required columns.")
        return

    # Get unique r0_ref values
    unique_r0s = sorted(df['r0_ref'].unique())
    
    results = []
    print(f"--- Processing {len(unique_r0s)} r0_ref groups ---")

    for r0 in unique_r0s:
        # Get all noise vars for this r0
        r0_subset = df[df['r0_ref'] == r0]
        unique_noises = sorted(r0_subset['noise_var'].unique())
        
        # 1. Find Reference Signal Power (from lowest noise case)
        min_noise = unique_noises[0]
        _, _, _, ref_signal_power = get_decision_rule(r0, min_noise, csv_path)
        
        if ref_signal_power is None:
            continue 

        # 2. Iterate over all noise levels
        for noise in unique_noises:
            # Unpack 4 values now (added std_cv_acc)
            thresh, acc, std_acc, _ = get_decision_rule(r0, noise, csv_path)
            
            if acc is not None:
                results.append({
                    'r0_ref': r0,
                    'noise_var': noise,
                    'threshold': thresh,
                    'cv_accuracy': acc,
                    'std_cv_accuracy': std_acc, # Save the standard deviation
                    'mean_signal_power': ref_signal_power
                })
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"\nSuccess! Saved rules to '{output_path}'.")
    else:
        print("\nNo valid rules were generated.")

def estimate_pixels_in_bucket(wavelength, F_num, q, bucket_diam):
    pixel_size = (wavelength * F_num) / q
    pixel_area = pixel_size ** 2
    bucket_area = np.pi * (bucket_diam / 2)**2
    num_pixels = bucket_area / pixel_area
    return max(1.0, num_pixels)

def plot_ber_vs_noise(summary_csv_path="./decision_rules_summary.csv", pixels_in_bucket=1.0, use_log_y=False):
    """
    Reads the summary CSV and plots BER vs SNR with Error Bars.
    """
    if not os.path.exists(summary_csv_path):
        print(f"Summary file {summary_csv_path} not found.")
        return

    df = pd.read_csv(summary_csv_path)
    
    # SNR Calculation
    df['total_noise_var'] = df['noise_var'] * pixels_in_bucket
    df['SNR'] = df['mean_signal_power'] / df['total_noise_var']
    df['BER'] = 1 - df['cv_accuracy']
    
    plt.figure(figsize=(10, 6))
    
    unique_r0s = sorted(df['r0_ref'].unique())
    
    for r0 in unique_r0s:
        subset = df[df['r0_ref'] == r0].sort_values(by='SNR')
        
        # Plot with Error Bars
        # yerr is the standard deviation of the accuracy (which is same as std of BER)
        plt.errorbar(
            subset['SNR'], 
            subset['BER'], 
            yerr=subset['std_cv_accuracy'], # The variance bars
            marker='o', 
            capsize=4,       # Adds the little "caps" to the error bars (scattering visualization)
            label=f'r0_ref = {r0}',
            alpha=0.8
        )
    
    # Scale setup
    plt.xscale('log')
    if use_log_y:
        plt.yscale('log')
        plt.ylabel('Bit Error Rate (BER) [Log Scale]')
        plt.title(f'BER vs. SNR (Log-Log) with Variance\nPixels in Bucket $\\approx$ {pixels_in_bucket:.2f}')
        plt.ylim(bottom=1e-4) 
    else:
        plt.ylabel('Bit Error Rate (BER) [Linear Scale]')
        plt.title(f'BER vs. SNR (Linear) with Variance\nPixels in Bucket $\\approx$ {pixels_in_bucket:.2f}')

    plt.xlabel('SNR (Signal / Total Noise Variance)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Dummy Data Generation ---
if not os.path.exists("./massive_output.csv"):
    np.random.seed(42)
    dummy_df = pd.DataFrame({
        'r0_ref': np.random.choice([0.075, 0.2, 0.3], 1000),
        'noise_var': np.random.choice([1e4, 1e5, 1e6], 1000),
        'power1_in_bucket': np.random.uniform(0, 10, 1000)
    })
    conds = [
        (dummy_df['r0_ref'] == 0.075) & (dummy_df['power1_in_bucket'] > 3),
        (dummy_df['r0_ref'] == 0.2) & (dummy_df['power1_in_bucket'] > 7),
        (dummy_df['r0_ref'] == 0.3) & (dummy_df['power1_in_bucket'] > 5) 
    ]
    dummy_df['sent_signal'] = np.select(conds, [True, True, True], default=False)
    flip_indices = np.random.choice(dummy_df.index, size=int(0.1 * len(dummy_df)), replace=False)
    dummy_df.loc[flip_indices, 'sent_signal'] = ~dummy_df.loc[flip_indices, 'sent_signal']
    dummy_df.to_csv("./massive_output.csv", index=False)

if __name__ == "__main__":
    # # 1. Process conditions (Calculates stable SNR + Standard Deviation)
    # process_all_conditions()
    
    # # 2. Parameters
    # wl = 1.55e-6
    # f_num = 50.0
    # q_factor = 8.0
    # bucket_diameter = 9e-6
    # N_pixels = estimate_pixels_in_bucket(wl, f_num, q_factor, bucket_diameter)
    
    # # 3. Plot BER vs SNR (Linear Y - Default)
    # print("Plotting Linear Y scale with Error Bars...")
    # plot_ber_vs_noise(pixels_in_bucket=N_pixels, use_log_y=True)
    
    
    def create_bit_stream(gain = 1,var = 1,N=1000):
        
        stream = np.hstack((np.zeros(N//2),np.ones(N - N//2)))
        stream = np.vstack((stream,stream))

        noise = np.random.randn(N) * var
        stream[0] = stream[0] * gain + noise
        return np.arange(N),stream
    
    
    def set_decision_rule(data):
        clf = DecisionTreeClassifier(max_depth=1, random_state=42)
        X,y = data[0].reshape(-1,1),data[1]==1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf.fit(X_train,y_train)
        acc = np.sum(clf.predict(X_test) == y_test)/len(X_test)
        return clf,acc

    
    def plot_decision_rule(Var = 0.1,Gain = 0.5):
        # train
        time_stamps,train_stream = create_bit_stream(gain=Gain,var=Var,N=100)
        clf,_ = set_decision_rule(train_stream)
        
        # test
        time_stamps,test_stream = create_bit_stream(gain=Gain,var=Var,N=1000)
        predictions = clf.predict(test_stream[0].reshape(-1,1))
        y = test_stream[1]
        
        acc = np.sum(predictions == y) / len(y)
        print(f"acc : {acc}")
        
        threshold = clf.tree_.threshold[0]
        colors = np.where(predictions == y, 'green', 'red')
        
        # 5. Plotting
        plt.figure(figsize=(10, 6))

        # Scatter plot with the dynamic colors
        plt.scatter(time_stamps, test_stream[0], c=colors, alpha=0.7, s=50,label = "noisy signal pred.")
        plt.scatter(time_stamps, test_stream[1], alpha=0.7, s=50,label = "sent signal")

        threshold = clf.tree_.threshold[0]
        plt.axhline(y=threshold, color='blue', linestyle='--', label=f'Decision Threshold ({threshold:.2f})')
        plt.title(f"Classification Accuracy (Green=Correct, Red=Wrong), BER : {1-acc :.1%}\n G = {Gain} , Var = {Var}")
        plt.xlabel("Time Stamp")
        plt.ylabel("Signal Value [Watt]")
        plt.legend()
        plt.show()
    
    
    # plot_decision_rule()
    
    num = 10 
    N = 5
    accuracies = []
    Vars = np.logspace(-8,-1,num)
    Gains = np.logspace(-4,0,4)
    
    for var in Vars:
        for gain in Gains:    
            avg = 0
            print(f"var = {var}")
            plot_decision_rule(Var = var , Gain = gain)
    
    # print(len(accuracies))
    plt.plot(np.log10(Vars),accuracies)
    plt.show()