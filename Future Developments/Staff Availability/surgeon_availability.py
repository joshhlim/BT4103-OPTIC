import pandas as pd

# === 1. Load CSV with relevant columns only ===
file_path = "Final_Cleaned_Dataset_OPTIC_7.csv"
columns_to_read = ["SURGEON", "PLANNED_ENTER_OR_TIME", "PLANNED_EXIT_OR_TIME"]
df = pd.read_csv(file_path, usecols=columns_to_read) 

# === 2. Convert datetime ===
df["PLANNED_ENTER_OR_TIME"] = pd.to_datetime(df["PLANNED_ENTER_OR_TIME"], errors="coerce")
df["PLANNED_EXIT_OR_TIME"] = pd.to_datetime(df["PLANNED_EXIT_OR_TIME"], errors="coerce")

# Drop rows with missing values in the relevant columns
df = df.dropna()

# Remove surgeon == Unknown
df = df[df["SURGEON"] != "Unknown"]

# === 3. Sort by surgeon and start time ===
df = df.sort_values(["SURGEON", "PLANNED_ENTER_OR_TIME"]).reset_index(drop=True)

# === 4. Merge consecutive or close intervals (<= 30 min apart) ===
merged_rows = []
gap_threshold = pd.Timedelta(minutes=30)

for surgeon, group in df.groupby("SURGEON"):
    group = group.sort_values("PLANNED_ENTER_OR_TIME").reset_index(drop=True)
    current_start = group.loc[0, "PLANNED_ENTER_OR_TIME"]
    current_end = group.loc[0, "PLANNED_EXIT_OR_TIME"]
    
    for i in range(1, len(group)):
        next_start = group.loc[i, "PLANNED_ENTER_OR_TIME"]
        next_end = group.loc[i, "PLANNED_EXIT_OR_TIME"]
        
        # Merge if the gap ≤ 30 minutes
        if next_start - current_end <= gap_threshold:
            current_end = max(current_end, next_end)
        else:
            merged_rows.append([surgeon, current_start, current_end])
            current_start = next_start
            current_end = next_end
    
    merged_rows.append([surgeon, current_start, current_end])

# === 5. Create result dataframe ===
availability_periods = pd.DataFrame(merged_rows, columns=["SURGEON", "PLANNED_START", "PLANNED_END"])

# === 6. (Optional) Export ===
availability_periods.to_excel("surgeon_continuous_availability.xlsx", index=False)
