import pandas as pd

# 1. Read Excel with selected columns
cols_to_read = [
    'LOCATION', 'ROOM'
]

df = pd.read_excel("Final_Cleaned_Dataset_OPTIC_7.xlsx", usecols=cols_to_read)

# Drop rows with missing values in the relevant columns
df_subset = df[['LOCATION', 'ROOM']].dropna()

# Check unique mappings from LOCATION → ROOM
location_to_room = df_subset.groupby('LOCATION')['ROOM'].nunique()
room_to_location = df_subset.groupby('ROOM')['LOCATION'].nunique()

# Print potential issues
print("Locations mapping to multiple Rooms:")
print(location_to_room[location_to_room > 1])

print("\nRooms mapping to multiple Locations:")
print(room_to_location[room_to_location > 1])

# Locations mapping to multiple Rooms:
# LOCATION
# AH Endoscopy Center     2
# DDI                     3
# Endoscopy Center        8
# ICL                     3
# Kent Ridge Wing OT      9
# Main Building OT       19
# Medical Center OT      10
# Name: ROOM, dtype: int64

# Rooms mapping to multiple Locations:
# Series([], Name: LOCATION, dtype: int64)