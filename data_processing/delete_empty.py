import os
import shutil
import pandas as pd

# List of columns to check
columns_to_check = [
    "Max Temp (°C)", "Max Temp Flag",
    "Min Temp (°C)", "Min Temp Flag",
    "Mean Temp (°C)", "Mean Temp Flag",
    "Heat Deg Days (°C)", "Heat Deg Days Flag",
    "Cool Deg Days (°C)", "Cool Deg Days Flag",
    "Total Rain (mm)", "Total Rain Flag",
    "Total Snow (cm)", "Total Snow Flag",
    "Total Precip (mm)", "Total Precip Flag",
    "Snow on Grnd (cm)", "Snow on Grnd Flag",
    "Dir of Max Gust (10s deg)", "Dir of Max Gust Flag",
    "Spd of Max Gust (km/h)", "Spd of Max Gust Flag"
]

# Minimum number of non-empty columns required to keep a file
min_non_empty_columns = 3

# Directory to move "kept" CSVs
keep_dir = "keep"
os.makedirs(keep_dir, exist_ok=True)

# Walk through all CSVs
for root, _, files in os.walk("."):
    for file in files:
        if file.lower().endswith(".csv"):
            file_path = os.path.join(root, file)

            # Avoid moving already-kept files again
            if os.path.abspath(keep_dir) in os.path.abspath(file_path):
                continue

            try:
                df = pd.read_csv(file_path)
                non_empty_count = 0

                for col in columns_to_check:
                    if col in df.columns:
                        cleaned_col = df[col].replace("", pd.NA).dropna()
                        if not cleaned_col.empty:
                            non_empty_count += 1

                if non_empty_count < min_non_empty_columns:
                    print(f"[Would delete] {file_path} — only {non_empty_count} non-empty column(s).")
                else:
                    print(f"[Keeping] {file_path} — {non_empty_count} non-empty column(s). Moving to '{keep_dir}/'")
                    shutil.move(file_path, os.path.join(keep_dir, file))
            except Exception as e:
                print(f"[Error] Failed to process {file_path}: {e}")
