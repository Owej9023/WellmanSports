#sandbox
import zipfile
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

import os
# from WellmanFunctions import create_dataframes, analyze_targets
    
# merged_df = create_dataframes()

# corr = merged_df.select_dtypes(include=['number']).corr()

# #plot the correlation matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
# plt.title('Correlation Matrix between Foreplate Metrics and Training Data')
# plt.show()

# df = merged_df.copy()

# def create_dataframes():
training_df = pd.read_csv(r'C:\Users\dickh\Downloads\WellmanSports\WellmanWorkoutTracker1-28-2026.csv')
spartan_folder = r"C:\Users\dickh\Downloads\WellmanSports\SpartanData"
workouts= pd.read_csv(r'C:\Users\dickh\Downloads\WellmanSports\WellmanExercises.csv')
all_dfs = []
for file_name in os.listdir(spartan_folder):
    if file_name.endswith(".zip"):
        zip_path = os.path.join(spartan_folder, file_name)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for inner_file in zip_ref.namelist():
                if inner_file.endswith(".csv"):
                    with zip_ref.open(inner_file) as f:
                        df = pd.read_csv(f)
                        all_dfs.append(df)

if not all_dfs:
    raise RuntimeError("No CSVs were read from the zip files in: " + spartan_folder)

foreplate_df = pd.concat(all_dfs, ignore_index=True)

col_cut = 'P1|P2 Propulsive Impulse Index'
if col_cut in foreplate_df.columns:
    cut_idx = foreplate_df.columns.get_loc(col_cut)
    foreplate_df = foreplate_df.iloc[:, :cut_idx+1]
else:
    raise KeyError(f"Column not found: {col_cut}")


training_df = training_df.rename(columns={
'First Name': 'first_name',
'Last Name': 'last_name'
})

workouts = workouts.rename(columns={
    'First Name': 'first_name',
    'Last Name': 'last_name',
    'Date Completed Exercise': 'Date Completed',
    'Reps': 'Exercise Reps'
})

training_df['Full Name'] = (
    training_df['first_name'].astype(str).str.strip() + ' ' +
    training_df['last_name'].astype(str).str.strip()
)

workouts['Full Name'] = (
    workouts['first_name'].astype(str).str.strip() + ' ' +
    workouts['last_name'].astype(str).str.strip()
)

foreplate_df = foreplate_df.rename(columns={'Name': 'Full Name'})

for df in [training_df, workouts, foreplate_df]:
    df['Full Name'] = (
        df['Full Name']
        .astype(str)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.upper()
    )

workouts = workouts.drop(columns=["first_name", "last_name","Unnamed: 7"], errors='ignore')
training_df = training_df.drop(columns=["first_name", "last_name"], errors='ignore')
foreplate_df['Date Completed'] = pd.to_datetime(foreplate_df['Date'], errors='coerce')
foreplate_df=foreplate_df.drop(columns=["Date"], errors='ignore')
training_df['Date Completed'] = pd.to_datetime(training_df['Date Completed'], errors='coerce')
workouts['Date Completed'] = pd.to_datetime(workouts['Date Completed'], errors='coerce')




for df in [foreplate_df, training_df, workouts]:
    df['Date Completed'] = df['Date Completed'].dt.normalize()


# merge the dataframes on Full Name and Date Completed
merged_df = (
    foreplate_df
    .merge(training_df, on='Full Name', how='inner')
    .merge(workouts, on='Full Name', how='inner')
)

merged_df.replace(['-', 'N/A', ''], np.nan, inplace=True)
# exclude peak landing force, jump momentum,mRSI
#they are not interested in these variables
merged_df = merged_df.drop(columns=["Peak Landing Force", "Jump Momentum", "mRSI","Readiness","Intensity"])


#now display every unique name in that dataframe
unique_names = merged_df['Full Name'].unique()

unique_names_foreplate = foreplate_df['Full Name'].unique()

unique_names_training = training_df['Full Name'].unique()
unique_names_workouts = workouts['Full Name'].unique()

#print the total amount of unique names in each dataframe
# print(f"Total unique names in foreplate dataframe: {len(unique_names_foreplate)}")
# print(f"Total unique names in training dataframe: {len(unique_names_training)}")
# print(f"Total unique names in workouts dataframe: {len(unique_names_workouts)}")
# print(f"Total unique names in merged dataframe: {len(unique_names)}")

# for name in training_df['Full Name'].unique():
#     if name not in unique_names:
#         print(f"{name} is not in both dataframes")

# fp_names = set(foreplate_df['Full Name'])
# tr_names = set(training_df['Full Name'])
# wo_names = set(workouts['Full Name'])

# intersection = fp_names & tr_names & wo_names
# print("Intersection:", len(intersection))

# print("In workouts but not in foreplate:",
#       len(wo_names - fp_names))

# print("In workouts but not in training:",
#       len(wo_names - tr_names))
print(merged_df.head())