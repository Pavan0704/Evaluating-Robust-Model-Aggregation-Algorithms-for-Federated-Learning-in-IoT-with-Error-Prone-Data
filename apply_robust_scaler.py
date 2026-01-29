
import csv
import numpy as np
from sklearn.preprocessing import RobustScaler
import os

# Paths
INPUT_FILE = "fl_sim_outputs/iot_outliers_with_aggregator.csv"
OUTPUT_FILE = "fl_sim_outputs/outliers_normalized.csv"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    print(f"Loading {INPUT_FILE} using pure csv...")
    
    # Read data
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        data = list(reader)
        
    if not data:
        print("Error: Empty file.")
        return

    # Extract numeric columns for scaling
    numeric_keys = ['temperature_C', 'humidity_pct', 'pressure_hPa', 'aggregator_sensor_index']
    
    # Check if keys exist
    for k in numeric_keys:
        if k not in fieldnames:
            print(f"Error: Column {k} not found in CSV. Fields: {fieldnames}")
            return

    # Convert to numpy array
    print("Converting to numpy...")
    raw_vals = []
    for row in data:
        row_vals = []
        for k in numeric_keys:
            try:
                val = float(row[k])
            except ValueError:
                val = np.nan
            row_vals.append(val)
        raw_vals.append(row_vals)
        
    X = np.array(raw_vals, dtype=np.float64)
    
    # Impute NaNs (Median)
    print("Imputing NaNs...")
    for i in range(X.shape[1]):
        col = X[:, i]
        mask = np.isnan(col)
        if np.any(mask):
            med = np.nanmedian(col)
            X[mask, i] = med
            print(f"Imputed NaNs in col {i} ({numeric_keys[i]}) with {med}")

    # Apply RobustScaler
    print("Applying RobustScaler...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Update data list with scaled values
    print("Updating data structure...")
    for i, row in enumerate(data):
        for j, k in enumerate(numeric_keys):
            row[k] = str(X_scaled[i, j])
            
    # Write back
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
        
    # Print stats manually
    print("\n--- New Stats (Robust Scaled) ---")
    for j, k in enumerate(numeric_keys):
        col = X_scaled[:, j]
        print(f"{k}: Min={np.min(col):.4f}, Max={np.max(col):.4f}, Median={np.median(col):.4f}, Std={np.std(col):.4f}")

    print("Done.")

if __name__ == "__main__":
    main()
