
import numpy as np
import os
from sklearn.linear_model import LinearRegression
# Import the working loader from Code.py
from Code import load_data_numpy

# We'll use the clean normalized file as it represents the "ground truth" logic
INPUT_FILE = "fl_sim_outputs/clean_normalized.csv"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    print(f"Loading {INPUT_FILE} using Code.load_data_numpy...")
    try:
        X, y = load_data_numpy(INPUT_FILE)
        # X is (N, 3), y is (N,)
        # Features: Temp, Hum, Pressure
    except Exception as e:
        print(f"Load failed: {e}")
        return

    feat_names = ['Temp', 'Hum', 'Press']
    # Redirect print to file
    with open("correlation_results.txt", "w") as f:
        def log(s):
            print(s)
            f.write(s + "\n")
            
        log(f"Data shape: X={X.shape}, y={y.shape}")

        log("\n--- Correlation Analysis ---")
        for i, name in enumerate(feat_names):
            # Pearson correlation
            corr = np.corrcoef(X[:, i], y)[0, 1]
            log(f"{name} vs Target: {corr:.4f}")

        log("\n--- Linear Regression Analysis ---")
        reg = LinearRegression().fit(X, y)
        log(f"Intercept: {reg.intercept_:.4f}")
        weights = reg.coef_
        formula = f"Target = {reg.intercept_:.2f}"
        for name, w in zip(feat_names, weights):
            log(f"{name} Weight: {w:.4f}")
            formula += f" + ({w:.2f} * {name})"
            
        r2 = reg.score(X, y)
        log(f"R2 Score: {r2:.4f}")
        
        log("\n--- Model Inference ---")
        if r2 > 0.99:
            log(f"The target is likely a DIRECT linear combination:\n  {formula}")
        else:
            log("The target has a non-linear or noisy relationship.")

if __name__ == "__main__":
    main()
