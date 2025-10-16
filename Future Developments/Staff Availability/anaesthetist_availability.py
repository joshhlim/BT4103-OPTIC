import pandas as pd

# === 1. Load CSV with relevant columns only ===
file_path = "Final_Cleaned_Dataset_OPTIC_7.csv"
columns_to_read = ["ANAESTHETIST_MCR_NO", "PLANNED_ENTER_OR_TIME", "PLANNED_EXIT_OR_TIME"]
df = pd.read_csv(file_path, usecols=columns_to_read)

# === 2. Convert datetime ===
df["PLANNED_ENTER_OR_TIME"] = pd.to_datetime(df["PLANNED_ENTER_OR_TIME"], errors="coerce")
df["PLANNED_EXIT_OR_TIME"] = pd.to_datetime(df["PLANNED_EXIT_OR_TIME"], errors="coerce")

# Drop rows with missing values in the relevant columns
df = df.dropna() 

# Remove anaesthetist == Unknown
df = df[df["ANAESTHETIST_MCR_NO"] != "Unknown"]

# === 3. Sort by anaesthetist and start time ===
df = df.sort_values(["ANAESTHETIST_MCR_NO", "PLANNED_ENTER_OR_TIME"]).reset_index(drop=True)

# === 4. Merge consecutive or close intervals (<= 30 min apart) ===
merged_rows = []
gap_threshold = pd.Timedelta(minutes=30)

for anaesthetist, group in df.groupby("ANAESTHETIST_MCR_NO"):
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
            merged_rows.append([anaesthetist, current_start, current_end])
            current_start = next_start
            current_end = next_end
    
    merged_rows.append([anaesthetist, current_start, current_end])

# === 5. Create result dataframe ===
availability_periods = pd.DataFrame(merged_rows, columns=["ANAESTHETIST_MCR_NO", "PLANNED_START", "PLANNED_END"])

# === 6. (Optional) Export ===
availability_periods.to_excel("anaesthetist_continuous_availability.xlsx", index=False)
