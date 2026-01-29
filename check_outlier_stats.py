
import numpy as np
import os
from Code import load_data_numpy

file_path = "fl_sim_outputs/outliers_normalized.csv"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    print(f"Analyzing {file_path} using numpy...")
    try:
        X, y = load_data_numpy(file_path)
        # X is (N, 3), y is (N,)
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Analyze Features (Columns 0, 1, 2)
        feat_names = ["Temp", "Humidity", "Pressure"]
        for i in range(3):
            col = X[:, i]
            print(f"\n--- Feature {i} ({feat_names[i]}) ---")
            print(f"  Min: {np.min(col):.6f}")
            print(f"  Max: {np.max(col):.6f}")
            print(f"  Mean: {np.mean(col):.6f}")
            print(f"  Std: {np.std(col):.6f}")
            print(f"  25%: {np.percentile(col, 25):.6f}")
            print(f"  50%: {np.median(col):.6f}")
            print(f"  75%: {np.percentile(col, 75):.6f}")
            
        # Analyze Target
        print("\n--- Target (Aggregator Index) ---")
        print(f"  Min: {np.min(y):.6f}")
        print(f"  Max: {np.max(y):.6f}")
        print(f"  Mean: {np.mean(y):.6f}")
        print(f"  Std: {np.std(y):.6f}")
        print(f"  25%: {np.percentile(y, 25):.6f}")
        print(f"  50%: {np.median(y):.6f}")
        print(f"  75%: {np.percentile(y, 75):.6f}")

    except Exception as e:
        print(f"Error analyzing file: {e}")
