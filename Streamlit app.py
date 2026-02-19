# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import zipfile
# import seaborn as sns
# # --- new imports & helper function: compute top predictive features ---
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.impute import SimpleImputer
# from sklearn.inspection import permutation_importance
# import numpy as np

# def compute_top_predictive_features(df, target, n_top=10, missing_threshold=0.8):
#     """
#     Returns dict with r2, n_rows, top permutation importances and top absolute correlations.
#     """
#     if target not in df.columns:
#         return {"error": f"target '{target}' not found"}

#     y = pd.to_numeric(df[target], errors="coerce")
#     X = df.select_dtypes(include=[np.number]).copy()
#     if target in X.columns:
#         X = X.drop(columns=[target])

#     # require rows with valid target
#     mask = y.notna()
#     X = X.loc[mask]
#     y = y.loc[mask]

#     if len(y) < 10:
#         return {"error": f"insufficient rows ({len(y)}) for target"}

#     # drop sparse features
#     keep = (X.isna().mean() < missing_threshold)
#     X = X.loc[:, keep]
#     if X.shape[1] == 0:
#         return {"error": "no numeric features remaining after filtering"}

#     # impute and fit RF
#     imputer = SimpleImputer(strategy="median")
#     X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

#     rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
#     rf.fit(X_imp, y)
#     r2 = rf.score(X_imp, y)

#     pi = permutation_importance(rf, X_imp, y, n_repeats=10, random_state=0, n_jobs=-1)
#     importances = pd.Series(pi.importances_mean, index=X_imp.columns).sort_values(ascending=False)

#     corrs = X_imp.corrwith(y).abs().sort_values(ascending=False)

#     return {
#         "r2": r2,
#         "n_rows": len(y),
#         "importances": importances.head(n_top),
#         "correlations": corrs.head(n_top)
#     }

# def check_athlete_has_sufficient_data(merged_df, first_name, last_name, target, min_rows=10):
#     """
#     Check if an athlete has sufficient data for a given target metric.
#     Returns True if they have at least min_rows of valid data.
#     """
#     mask = (
#         merged_df["first_name"].astype(str).str.strip().str.lower() == first_name.strip().lower()
#     ) & (
#         merged_df["last_name"].astype(str).str.strip().str.lower() == last_name.strip().lower()
#     )
#     athlete_data = merged_df.loc[mask]
    
#     if athlete_data.empty:
#         return False
    
#     if target not in athlete_data.columns:
#         return False
    
#     y = pd.to_numeric(athlete_data[target], errors="coerce")
#     valid_rows = y.notna().sum()
    
#     return valid_rows >= min_rows

# def get_r2_interpretation(r2):
#     """Return interpretation of R² score"""
#     if r2 < 0.3:
#         return "⚠️ **Weak model** - These features have limited ability to predict the target metric. The relationships are either weak or non-linear."
#     elif r2 < 0.5:
#         return "📊 **Moderate model** - These features show some predictive ability but there's substantial unexplained variation."
#     elif r2 < 0.7:
#         return "✅ **Good model** - These features are fairly strong predictors of the target metric."
#     else:
#         return "🎯 **Excellent model** - These features are very strong predictors. The model explains most of the variation in the target."

# df = pd.read_csv(r'C:\Users\dickh\Downloads\WellmanSports\WellmanWorkoutTracker1-28-2026.csv')

# df["Date Completed"] = pd.to_datetime(df["Date Completed"])
# df["Blocks Completed"] = pd.to_numeric(df["Blocks Completed"], errors="coerce")
# df["Blocks Prescribed"] = pd.to_numeric(df["Blocks Prescribed"], errors="coerce")

# df["month"] = df["Date Completed"].dt.to_period("M")
# monthly = (
#     df
#     .groupby(["First Name", "Last Name", "month"], as_index=False)
#     .agg(
#         total_workouts=("Date Completed", "count"),
#         total_blocks_prescribed=("Blocks Prescribed", "sum"),
#         total_blocks_completed=("Blocks Completed", "sum")
#     )
# )
# monthly["days_in_month"] = monthly["month"].dt.days_in_month

# # Allow user to choose whether to compare workouts to days-in-month or to an
# # explicitly assigned number of workouts via the Streamlit sidebar.
# st.sidebar.markdown("**Assigned workouts options**")
# use_single_assigned = st.sidebar.checkbox(
#     "Use a single assigned workouts value (for all months)",
#     value=False
# )
# use_per_month = st.sidebar.checkbox(
#     "Use per-month assigned workouts (enter a value for each month)",
#     value=False
# )

# if use_per_month:
#     # Build per-month inputs and store them in session_state via unique keys.
#     months = monthly["month"].astype(str).sort_values().unique()
#     st.sidebar.markdown("Enter assigned workouts for each month:")
#     for m in months:
#         default = int(
#             monthly.loc[monthly["month"].astype(str) == m, "days_in_month"].iloc[0]
#         )
#         st.sidebar.number_input(
#             m,
#             min_value=1,
#             value=default,
#             step=1,
#             key=f"assigned_{m}"
#         )

#     # Read back values from session_state and map to the DataFrame
#     mapping = {}
#     for m in months:
#         mapping[m] = st.session_state.get(f"assigned_{m}",
#                                          int(monthly.loc[monthly["month"].astype(str) == m, "days_in_month"].iloc[0]))

#     monthly["assigned_workouts"] = monthly["month"].astype(str).map(mapping)
# elif use_single_assigned:
#     assigned_value = st.sidebar.number_input(
#         "Assigned workouts per month",
#         min_value=1,
#         value=16,
#         step=1
#     )
#     monthly["assigned_workouts"] = assigned_value
# else:
#     monthly["assigned_workouts"] = monthly["days_in_month"]

# monthly["Number_Workouts_consistency_score"] = (
#     monthly["total_workouts"] / monthly["assigned_workouts"]
# ).clip(upper=1)

# monthly["block_consistency_score"] = (
#     monthly["total_blocks_completed"] /
#     monthly["total_blocks_prescribed"]
# ).fillna(0).clip(upper=1)

# monthly["overall_consistency_score"] = (
#     0.5 * monthly["Number_Workouts_consistency_score"] +
#     0.5 * monthly["block_consistency_score"]
# )
# latest_month = monthly["month"].max()

# print(
#     monthly[monthly["month"] == latest_month]
#     .sort_values("overall_consistency_score", ascending=False)
#     .head(10)
# )
# monthly[
#     (monthly["First Name"] == "Caleb") &
#     (monthly["Last Name"] == "Auyeung")
# ].sort_values("month")


# monthly["month"] = monthly["month"].astype(str)

# st.title("Athlete Monthly Consistency Dashboard")

# # --- Predictive features UI (new) ---

# spartan_folder = r"C:\Users\dickh\Downloads\WellmanSports\SpartanData"
# training_df = pd.read_csv(r'C:\Users\dickh\Downloads\WellmanSports\WellmanWorkoutTracker1-28-2026.csv')

# all_dfs = []
# for file_name in os.listdir(spartan_folder):
#     if file_name.endswith(".zip"):
#         zip_path = os.path.join(spartan_folder, file_name)
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             for inner_file in zip_ref.namelist():
#                 if inner_file.endswith(".csv"):
#                     with zip_ref.open(inner_file) as f:
#                         df = pd.read_csv(f)
#                         all_dfs.append(df)

# if not all_dfs:
#     raise RuntimeError("No CSVs were read from the zip files in: " + spartan_folder)


# # print(all_dfs[0].head())

# # combine all read CSV DataFrames into a single DataFrame
# foreplate_df = pd.concat(all_dfs, ignore_index=True)

# # keep only columns up to and including 'p1|p2 propulsive impulse index'
# col_cut = 'P1|P2 Propulsive Impulse Index'
# if col_cut in foreplate_df.columns:
#     cut_idx = foreplate_df.columns.get_loc(col_cut)
#     foreplate_df = foreplate_df.iloc[:, :cut_idx+1]
# else:
#     raise KeyError(f"Column not found: {col_cut}")

# #combine the two dataframes
# foreplate_df[['first_name', 'last_name']] = foreplate_df['Name'].str.strip().str.split(n=1, expand=True)

# # print(foreplate_df.head())
# # print(training_df.head())

# merged_df = pd.merge(foreplate_df, training_df,
#                      left_on=['first_name', 'last_name'],
#                      right_on=['First Name', 'Last Name'],
#                      how='inner')

# # keep first_name/last_name for per-athlete filtering; drop source name cols from training_df
# merged_df = merged_df.drop(columns=["First Name", "Last Name"])
# merged_df.replace(['-', 'N/A', ''], np.nan, inplace=True)
    


# numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
# if numeric_cols:
#     st.sidebar.markdown("## Predictive features")

#     # scope: whole dataset vs individual athlete
#     scope = st.sidebar.radio("Scope", ["Whole dataset", "Individual athlete"])

#     # target selection
#     target_choice = st.sidebar.selectbox(
#         "Choose target metric",
#         options=numeric_cols,
#         index=0
#     )

#     # athlete selector only for individual scope (reuse monthly athlete list)
#     selected_athlete_for_predict = None
#     if scope == "Individual athlete":
#         athlete_names = (
#             monthly[["First Name", "Last Name"]]
#             .drop_duplicates()
#             .sort_values(["Last Name", "First Name"])
#         )
        
#         # Filter to only athletes with sufficient data for the selected target
#         valid_athletes = []
#         for _, row in athlete_names.iterrows():
#             first = row["First Name"]
#             last = row["Last Name"]
#             if check_athlete_has_sufficient_data(merged_df, first, last, target_choice, min_rows=10):
#                 valid_athletes.append(f"{first} {last}")
        
#         if not valid_athletes:
#             st.sidebar.warning(f"⚠️ No athletes have sufficient data (10+ observations) for '{target_choice}'")
#             st.sidebar.info("Try selecting a different target metric or use 'Whole dataset' scope.")
#         else:
#             st.sidebar.success(f"✅ {len(valid_athletes)} athlete(s) with sufficient data")
#             selected_athlete_for_predict = st.sidebar.selectbox(
#                 "Select athlete for predictive analysis", 
#                 valid_athletes
#             )

#     run_btn = st.sidebar.button("Compute top features")

#     if run_btn:
#         # choose dataframe to analyze
#         if scope == "Whole dataset":
#             df_to_analyze = merged_df.copy()
#         else:
#             if not selected_athlete_for_predict:
#                 st.error("Please select an athlete with sufficient data.")
#                 st.stop()
                
#             first, last = selected_athlete_for_predict.split(" ", 1)
#             mask = (
#                 merged_df["first_name"].astype(str).str.strip().str.lower() == first.strip().lower()
#             ) & (
#                 merged_df["last_name"].astype(str).str.strip().str.lower() == last.strip().lower()
#             )
#             df_to_analyze = merged_df.loc[mask]
#             if df_to_analyze.empty:
#                 st.sidebar.error("No rows found for selected athlete in merged data.")
#                 st.stop()

#         res = compute_top_predictive_features(df_to_analyze, target_choice, n_top=10)
#         if "error" in res:
#             st.sidebar.error(res["error"])
#         else:
#             # Display header with model quality
#             st.subheader(f"🎯 Predictive Analysis: {target_choice}")
#             st.write(f"**Scope:** {scope} | **R² Score:** {res['r2']:.3f} | **Sample Size:** {res['n_rows']} observations")
            
#             # Add interpretation
#             st.info(get_r2_interpretation(res['r2']))
            
#             # Permutation importances
#             st.markdown("---")
#             st.markdown("### 🌲 Permutation Importance (Top 10)")
#             st.markdown("""
#             **What this shows:** How much the model's prediction accuracy drops when each feature is randomly shuffled. 
#             Higher values mean the feature is more important for accurate predictions.
            
#             **Use case:** These are the features that **causally influence** the target metric. If you want to improve 
#             the target, focus on these features. If all the values in this table are low, it means none of the features
#             are very predictive of the target.
#             """)
#             st.table(res["importances"].rename("Importance Score"))
            
#             # Correlations
#             st.markdown("---")
#             st.markdown("### 📊 Correlation Strength (Top 10)")
#             st.markdown("""
#             **What this shows:** The strength of the linear relationship between each feature and the target. 
#             Values closer to 1.0 indicate stronger direct relationships.
            
#             **Use case:** These features **move together** with the target metric. They're useful for monitoring 
#             and understanding patterns, but correlation doesn't prove causation. A feature can be highly correlated
#             with the target but not actually influence it.
#             """)
#             st.table(res["correlations"].rename("Absolute Correlation"))
            
#             # Key differences explanation
#             with st.expander("ℹ️ What's the difference between importance and correlation?"):
#                 st.markdown("""
#                 **Permutation Importance** tells you which features the model **needs** to make accurate predictions. 
#                 It accounts for complex, non-linear relationships and interactions between features.
                
#                 **Correlation** only measures simple, linear relationships. A feature can have high importance but 
#                 low correlation if it interacts with other features or has a non-linear effect.
                
#                 **Example:** Imagine predicting jump height:
#                 - Squat strength might have high correlation (linear relationship)
#                 - Training age might have low correlation but high importance (it moderates other effects)
                
#                 **Best practice:** Use importance scores to identify what to focus on for improvement, and correlation 
#                 to understand simple relationships and monitor trends.
#                 """)

# # --- Athlete selector ---
# athletes = (
#     monthly[["First Name", "Last Name"]]
#     .drop_duplicates()
#     .sort_values(["Last Name", "First Name"])
# )

# athlete_names = (
#     athletes["First Name"] + " " + athletes["Last Name"]
# ).tolist()

# selected_athlete = st.selectbox(
#     "Select an athlete",
#     athlete_names
# )

# first, last = selected_athlete.split(" ", 1)

# athlete_df = (
#     monthly[
#         (monthly["First Name"] == first) &
#         (monthly["Last Name"] == last)
#     ]
#     .sort_values("month")
# )

# # --- Plot ---
# fig, ax = plt.subplots()

# ax.plot(
#     athlete_df["month"],
#     athlete_df["overall_consistency_score"],
#     marker="o",
#     label="Overall"
# )

# ax.plot(
#     athlete_df["month"],
#     athlete_df["Number_Workouts_consistency_score"],
#     marker="o",
#     linestyle="--",
#     label="Number_Workouts"
# )

# ax.plot(
#     athlete_df["month"],
#     athlete_df["block_consistency_score"],
#     marker="o",
#     linestyle=":",
#     label="Blocks"
# )

# ax.set_ylim(0, 1.05)
# ax.set_xlabel("Month")
# ax.set_ylabel("Consistency Score")
# ax.set_title(f"Monthly Consistency – {selected_athlete}")
# ax.legend()

# plt.xticks(rotation=45)

# st.pyplot(fig)

# # --- Summary stats ---
# st.subheader("Monthly Summary")
# st.dataframe(
#     athlete_df[
#         [
#             "month",
#             "total_workouts",
#             "assigned_workouts",
#             "Number_Workouts_consistency_score",
#             "total_blocks_prescribed",
#             "total_blocks_completed",
#             "overall_consistency_score"
#         ]
#     ],
#     use_container_width=True
# )


####################################################################################################################
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import zipfile
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Session Title Impact Analysis",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data_efficiently(sample_size=None, date_filter=None, session_filter=None):
    """Memory-efficient loader that mirrors create_dataframes logic"""

    try:
        # -----------------------
        # Load Training + Workouts
        # -----------------------
        training_df = pd.read_csv(
            r'C:\Users\dickh\Downloads\WellmanSports\WellmanWorkoutTracker1-28-2026.csv',
            low_memory=False
        )

        workouts = pd.read_csv(
            r'C:\Users\dickh\Downloads\WellmanSports\WellmanExercises.csv',
            low_memory=False
        )

        # Rename to match create_dataframes()
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

        # -----------------------
        # Process Spartan Data (Chunked)
        # -----------------------
        spartan_folder = r"C:\Users\dickh\Downloads\WellmanSports\SpartanData"
        all_dfs = []

        for file_name in os.listdir(spartan_folder):
            if file_name.endswith(".zip"):
                zip_path = os.path.join(spartan_folder, file_name)

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for inner_file in zip_ref.namelist():
                        if inner_file.endswith(".csv"):
                            with zip_ref.open(inner_file) as f:

                                chunk_iter = pd.read_csv(
                                    f,
                                    chunksize=50000,
                                    low_memory=False
                                )

                                for chunk in chunk_iter:

                                    # Trim columns early
                                    col_cut = 'P1|P2 Propulsive Impulse Index'
                                    if col_cut in chunk.columns:
                                        cut_idx = chunk.columns.get_loc(col_cut)
                                        chunk = chunk.iloc[:, :cut_idx+1]

                                    all_dfs.append(chunk)

        if not all_dfs:
            raise RuntimeError("No CSVs found in SpartanData")

        foreplate_df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs

        # -----------------------
        # Standardize Names (ALL DATAFRAMES)
        # -----------------------

        # Create Full Name in training + workouts
        for df in [training_df, workouts]:
            df['Full Name'] = (
                df['first_name'].astype(str).str.strip() + ' ' +
                df['last_name'].astype(str).str.strip()
            )

        # Rename Spartan Name column
        foreplate_df = foreplate_df.rename(columns={'Name': 'Full Name'})

        # Clean names consistently
        for df in [training_df, workouts, foreplate_df]:
            df['Full Name'] = (
                df['Full Name']
                .astype(str)
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True)
                .str.upper()
            )

        # Drop unused columns early
        workouts = workouts.drop(columns=["first_name", "last_name", "Unnamed: 7"], errors='ignore')
        training_df = training_df.drop(columns=["first_name", "last_name"], errors='ignore')

        # -----------------------
        # Standardize Dates
        # -----------------------

        foreplate_df['Date Completed'] = pd.to_datetime(
            foreplate_df['Date'], errors='coerce'
        )
        foreplate_df = foreplate_df.drop(columns=["Date"], errors='ignore')

        training_df['Date Completed'] = pd.to_datetime(
            training_df['Date Completed'], errors='coerce'
        )

        workouts['Date Completed'] = pd.to_datetime(
            workouts['Date Completed'], errors='coerce'
        )

        # Normalize all dates
        for df in [foreplate_df, training_df, workouts]:
            df['Date Completed'] = df['Date Completed'].dt.normalize()

        # -----------------------
        # Merge (Same Logic as create_dataframes)
        # -----------------------

        merged_df = (
        foreplate_df
        .merge(training_df, on=['Full Name', 'Date Completed'], how='inner')
        .merge(workouts, on=['Full Name', 'Date Completed'], how='inner'))


        # -----------------------
        # Cleaning
        # -----------------------

        merged_df.replace(['-', 'N/A', ''], np.nan, inplace=True)

        merged_df = merged_df.drop(
            columns=["Peak Landing Force", "Jump Momentum", "mRSI",
                     "Readiness", "Intensity"],
            errors='ignore'
        )

        # -----------------------
        # Filters (AFTER merge)
        # -----------------------

        if session_filter and session_filter != 'All Sessions':
            if 'Session Title' in merged_df.columns:
                merged_df = merged_df[
                    merged_df['Session Title'] == session_filter
                ]

        if date_filter:
            chunk['Date Completed'] = pd.to_datetime(chunk['Date'], errors='coerce')
            cutoff_date = datetime.now() - timedelta(days=date_filter)
            chunk = chunk[chunk['Date Completed'] >= cutoff_date]


        # -----------------------
        # Sampling
        # -----------------------

        if sample_size and sample_size < len(merged_df):
            merged_df = merged_df.sample(n=sample_size, random_state=42)

        # -----------------------
        # Memory Optimization
        # -----------------------

        for col in merged_df.select_dtypes(include=['float64']).columns:
            merged_df[col] = pd.to_numeric(
                merged_df[col], downcast='float'
            )

        for col in merged_df.select_dtypes(include=['int64']).columns:
            merged_df[col] = pd.to_numeric(
                merged_df[col], downcast='integer'
            )

        return merged_df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_data
def get_data_summary(df):
    """Get quick summary statistics without loading full dataset"""
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'sessions': df['Session Title'].nunique() if 'Session Title' in df.columns else 0,
        'date_range': None
    }
    
    if 'Date Completed' in df.columns:
        try:
            dates = pd.to_datetime(df['Date Completed'], errors='coerce')
            summary['date_range'] = (dates.min(), dates.max())
        except:
            pass
    
    return summary


def get_performance_variables(df):
    """Identify potential performance variables in the dataset"""
    
    # Common performance variable names
    common_vars = [
        'Jump Height', 'Peak Propulsive Power', 'Avg. Propulsive Power',
        'Peak Propulsive Force', 'Avg. Propulsive Force', 'RSI',
        'Takeoff Velocity', 'Peak Velocity', 'Countermovement Depth',
        'Braking RFD', 'Time To Takeoff', 'Flight Time',
        'Peak Braking Power', 'Avg. Braking Power', 'Impulse Ratio',
        'Positive Net Impulse', 'Propulsive Net Impulse', 'Braking Net Impulse',
        'Stiffness', 'Peak Braking Force', 'Avg. Braking Force',
        'Landing Height', 'Avg. Landing Force',
        'Braking Impulse', 'Propulsive Impulse'
    ]
    
    # Find variables that exist in the dataframe
    available_vars = [var for var in common_vars if var in df.columns]
    
    # If none of the common vars found, get all numeric columns
    if not available_vars:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude ID and date-like columns
        available_vars = [col for col in numeric_cols 
                         if not any(x in col.lower() for x in ['id', 'date', 'time', 'year', 'unnamed'])]
    
    return available_vars


@st.cache_data
def calculate_statistics(df, session_col, var):
    """Calculate descriptive statistics by session"""
    stats_df = df.groupby(session_col)[var].agg([
        ('n', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('median', 'median')
    ]).round(2)
    
    return stats_df


def perform_statistical_test(df, session_col, var):
    """Perform ANOVA or Kruskal-Wallis test"""
    sessions = df[session_col].unique()
    
    if len(sessions) < 2:
        return None, None, None
    
    # Group data by session
    groups = [df[df[session_col] == session][var].dropna() for session in sessions]
    valid_groups = [g for g in groups if len(g) > 0]
    
    if len(valid_groups) < 2:
        return None, None, None
    
    # Check normality
    try:
        normal = all(stats.shapiro(g)[1] > 0.05 if len(g) >= 3 else True for g in valid_groups)
        
        if normal and all(len(g) >= 3 for g in valid_groups):
            stat, p_value = f_oneway(*valid_groups)
            test_used = "ANOVA"
        else:
            stat, p_value = kruskal(*valid_groups)
            test_used = "Kruskal-Wallis"
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        return test_used, p_value, significance
    except:
        return None, None, None


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d for effect size"""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0
    
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def create_boxplot(df, session_col, variables, max_points=10000):
    """Create interactive box plots using Plotly with sampling for large datasets"""
    # Sample data if too large for visualization
    if len(df) > max_points:
        df_sample = df.sample(n=max_points, random_state=42)
    else:
        df_sample = df
    
    n_vars = len(variables)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=variables,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set2
    sessions = df_sample[session_col].unique()
    
    for idx, var in enumerate(variables):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        for i, session in enumerate(sessions):
            session_data = df_sample[df_sample[session_col] == session][var].dropna()
            
            fig.add_trace(
                go.Box(
                    y=session_data,
                    name=session,
                    marker_color=colors[i % len(colors)],
                    showlegend=(idx == 0),
                    legendgroup=session
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=400 * n_rows,
        title_text="Distribution Comparison by Session Title",
        title_font_size=20,
        showlegend=True
    )
    
    return fig


def create_violin_plot(df, session_col, variables, max_points=10000):
    """Create interactive violin plots using Plotly with sampling"""
    # Sample data if too large
    if len(df) > max_points:
        df_sample = df.sample(n=max_points, random_state=42)
    else:
        df_sample = df
    
    n_vars = len(variables)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=variables,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set3
    sessions = df_sample[session_col].unique()
    
    for idx, var in enumerate(variables):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        for i, session in enumerate(sessions):
            session_data = df_sample[df_sample[session_col] == session][var].dropna()
            
            fig.add_trace(
                go.Violin(
                    y=session_data,
                    name=session,
                    marker_color=colors[i % len(colors)],
                    showlegend=(idx == 0),
                    legendgroup=session,
                    box_visible=True,
                    meanline_visible=True
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=400 * n_rows,
        title_text="Distribution Shape by Session Title",
        title_font_size=20,
        showlegend=True
    )
    
    return fig


def create_bar_chart(df, session_col, variables):
    """Create bar chart with error bars - uses aggregated data only"""
    n_vars = len(variables)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=variables,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Plotly
    sessions = df[session_col].unique()
    
    for idx, var in enumerate(variables):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        means = []
        stds = []
        session_labels = []
        
        for session in sessions:
            session_data = df[df[session_col] == session][var].dropna()
            if len(session_data) > 0:
                means.append(session_data.mean())
                stds.append(session_data.std())
                session_labels.append(session)
        
        fig.add_trace(
            go.Bar(
                x=session_labels,
                y=means,
                error_y=dict(type='data', array=stds),
                marker_color=colors[idx % len(colors)],
                showlegend=False,
                text=[f'{m:.2f}' for m in means],
                textposition='outside'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=400 * n_rows,
        title_text="Mean Comparison with Standard Deviation",
        title_font_size=20
    )
    
    return fig


def create_radar_chart(df, session_col, variables):
    """Create radar chart comparing sessions - uses aggregated data"""
    sessions = df[session_col].unique()
    
    if len(sessions) < 2 or len(variables) < 3:
        return None
    
    # Normalize data for radar chart (0-100 scale)
    radar_data = {}
    for session in sessions:
        radar_data[session] = []
        for var in variables:
            session_values = df[df[session_col] == session][var].dropna()
            if len(session_values) > 0:
                global_min = df[var].min()
                global_max = df[var].max()
                mean_val = session_values.mean()
                if global_max != global_min:
                    normalized = ((mean_val - global_min) / (global_max - global_min)) * 100
                else:
                    normalized = 50
                radar_data[session].append(normalized)
            else:
                radar_data[session].append(0)
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for idx, (session, values) in enumerate(radar_data.items()):
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=variables,
            fill='toself',
            name=session,
            line_color=colors[idx % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Performance Profile by Session (Normalized 0-100)",
        height=600
    )
    
    return fig


def create_correlation_heatmap(df, variables, max_rows=50000):
    """Create correlation heatmap with sampling for large datasets"""
    # Sample if too large
    if len(df) > max_rows:
        df_sample = df.sample(n=max_rows, random_state=42)
    else:
        df_sample = df
    
    numeric_data = df_sample[variables].select_dtypes(include=[np.number])
    
    if len(numeric_data.columns) < 2:
        return None
    
    corr_matrix = numeric_data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        height=600,
        xaxis={'side': 'bottom'}
    )
    
    return fig


def create_time_series(df, session_col, variables, date_col, max_points_per_series=1000):
    """Create time series plot with aggregation for large datasets"""
    if date_col not in df.columns:
        return None
    
    try:
        df_copy = df.copy()
        df_copy['Date_parsed'] = pd.to_datetime(df_copy[date_col])
    except:
        return None
    
    n_vars = len(variables)
    n_cols = min(2, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=variables,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set2
    sessions = df_copy[session_col].unique()
    
    for idx, var in enumerate(variables):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        for i, session in enumerate(sessions):
            session_df = df_copy[df_copy[session_col] == session].copy()
            if len(session_df) == 0:
                continue

            # Ensure datetime index and aggregate to daily means
            try:
                session_df = session_df.sort_values('Date_parsed')
                session_df = session_df.set_index('Date_parsed')
                daily = session_df[[var]].resample('D').mean().dropna().reset_index()
            except Exception:
                continue

            # If still too many points after daily aggregation, fall back to weekly
            if len(daily) > max_points_per_series:
                daily = daily.set_index('Date_parsed').resample('W').mean().reset_index()

            if len(daily) == 0:
                continue

            fig.add_trace(
                go.Scatter(
                    x=daily['Date_parsed'],
                    y=daily[var],
                    mode='lines+markers',
                    name=session,
                    marker_color=colors[i % len(colors)],
                    showlegend=(idx == 0),
                    legendgroup=session
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=400 * n_rows,
        title_text="Performance Trends Over Time",
        title_font_size=20,
        showlegend=True
    )
    
    return fig


# Main App
def main():
    st.markdown('<p class="main-header">🏃 Session Title Impact Analysis</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Filtering")
        
        # Data size controls
        st.subheader("Memory Management")
        
        sample_size = st.number_input(
            "Sample size (0 = all data)",
            min_value=0,
            max_value=1000000,
            value=100000,
            step=10000,
            help="Limit dataset size to prevent memory issues. Set to 0 to load all data (not recommended for large datasets)"
        )
        
        # Date filtering
        date_filter_options = {
            'All Time': None,
            'Last 30 Days': 30,
            'Last 90 Days': 90,
            'Last 6 Months': 180,
            'Last Year': 365
        }
        
        date_filter_selection = st.selectbox(
            "Filter by Date",
            options=list(date_filter_options.keys()),
            index=0
        )
        date_filter = date_filter_options[date_filter_selection]
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info(
            "This app analyzes how different training session types affect "
            "athletic performance variables from jump testing data.\n\n"
            "**Features:**\n"
            "- Memory-efficient data loading\n"
            "- Interactive visualizations\n"
            "- Statistical testing\n"
            "- Variable selection\n"
            "- Export capabilities"
        )
        
        # Load data button
        if st.button("🔄 Load/Reload Data", type="primary"):
            st.session_state.data_loaded = False
            st.cache_data.clear()
    
    # Load data
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            df = load_data_efficiently(
                sample_size=sample_size if sample_size > 0 else None,
                date_filter=date_filter
            )
            
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                summary = get_data_summary(df)
                st.success(f"✅ Data loaded! {summary['total_rows']:,} rows, {summary['total_columns']} columns, {summary['memory_usage_mb']:.1f} MB in memory")
            else:
                st.error("Failed to load data")
                return
    else:
        df = st.session_state.df
    
    # Main analysis
    if df is not None and len(df) > 0:
        
        # Configuration section
        st.markdown('<p class="sub-header">⚙️ Configuration</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select session column
            session_col = st.selectbox(
                "Select Session Title Column",
                options=df.columns.tolist(),
                index=df.columns.tolist().index('Session Title') if 'Session Title' in df.columns else 0,
                help="Column containing session/workout type names"
            )
        
        with col2:
            # Select date column (optional)
            date_cols = ['None'] + df.columns.tolist()
            date_col = st.selectbox(
                "Select Date Column (Optional)",
                options=date_cols,
                index=date_cols.index('Date Completed') if 'Date Completed' in date_cols else 0,
                help="For time series analysis"
            )
        
        # Athlete Filtering Section
        st.markdown("---")
        st.markdown("#### 👤 Filter by Athlete")
        
        # Create full name if First and Last Name columns exist
        athlete_filter = None
        if 'First Name' in df.columns and 'Last Name' in df.columns:
            df['Full Name'] = df['First Name'].astype(str) + ' ' + df['Last Name'].astype(str)
            
            # Get unique athletes sorted
            all_athletes = ['All Athletes'] + sorted(df['Full Name'].unique().tolist())
            
            athlete_filter = st.selectbox(
                "Select an athlete (or view all athletes)",
                options=all_athletes,
                index=0,
                help="Filter data to a single athlete or view all athletes combined"
            )
            
            # Filter dataframe by athlete
            if athlete_filter != 'All Athletes':
                df = df[df['Full Name'] == athlete_filter].copy()
                st.info(f"👤 Viewing data for **{athlete_filter}** - {len(df):,} observations")
        else:
            st.warning("⚠️ First Name and Last Name columns not found in dataset. Cannot filter by athlete.")
        
        # Session Filtering Section
        st.markdown("---")
        st.markdown("#### 🎯 Filter by Session Type")
        
        # Get unique session values
        all_sessions = sorted(df[session_col].unique().tolist())
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            selected_sessions = st.multiselect(
                "Select session types to include (leave empty for all)",
                options=all_sessions,
                default=[],
                help="Choose specific session types to analyze. Empty = all sessions included"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("Select All Sessions"):
                selected_sessions = all_sessions
            if st.button("Clear Sessions"):
                selected_sessions = []
        
        # Filter dataframe by selected sessions
        if selected_sessions and len(selected_sessions) > 0:
            df = df[df[session_col].isin(selected_sessions)].copy()
            st.info(f"📊 Filtered to {len(df):,} observations from {len(selected_sessions)} session type(s)")
        
        # Update session count after filtering
        if len(df) == 0:
            st.error("❌ No data remaining after filtering. Please adjust your session selection.")
            return
        
        # Get available performance variables
        available_vars = get_performance_variables(df)
        
        if not available_vars:
            st.error("No performance variables detected in the dataset!")
            return
        
        # Variable selection
        st.markdown('<p class="sub-header">📊 Select Variables to Analyze</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_vars = st.multiselect(
                "Choose performance variables",
                options=available_vars,
                default=available_vars[:min(6, len(available_vars))],
                help="Select variables to include in the analysis"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("Select All"):
                selected_vars = available_vars
            if st.button("Clear All"):
                selected_vars = []
        
        if not selected_vars:
            st.warning("⚠️ Please select at least one variable to analyze")
            return
        
        # Data Overview
        st.markdown('<p class="sub-header">📈 Data Overview</p>', unsafe_allow_html=True)
        
        # Adjust columns based on whether athlete is selected
        if athlete_filter and athlete_filter != 'All Athletes':
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Athlete", athlete_filter)
            
            with col2:
                st.metric("Total Observations", f"{len(df):,}")
            
            with col3:
                st.metric("Unique Sessions", df[session_col].nunique())
            
            with col4:
                st.metric("Variables Selected", len(selected_vars))
        else:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Observations", f"{len(df):,}")
            
            with col2:
                st.metric("Unique Athletes", df['Full Name'].nunique() if 'Full Name' in df.columns else 'N/A')
            
            with col3:
                st.metric("Unique Sessions", df[session_col].nunique())
            
            with col4:
                st.metric("Session Types Shown", len(selected_sessions) if selected_sessions else "All")
            
            with col5:
                st.metric("Variables Selected", len(selected_vars))
        
        # Missing data metric (always show)
        missing_pct = (df[selected_vars].isnull().sum().sum() / (len(df) * len(selected_vars)) * 100) if len(selected_vars) > 0 else 0
        st.metric("Missing Data %", f"{missing_pct:.1f}%")
        
        # Personal bests and progress (only when viewing individual athlete)
        if athlete_filter and athlete_filter != 'All Athletes' and len(selected_vars) > 0:
            with st.expander("🏆 Personal Bests & Progress", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Personal Best Values**")
                    pb_data = []
                    for var in selected_vars:
                        var_data = df[var].dropna()
                        if len(var_data) > 0:
                            pb_data.append({
                                'Variable': var,
                                'Best': f"{var_data.max():.2f}",
                                'Average': f"{var_data.mean():.2f}",
                                'Latest': f"{var_data.iloc[-1]:.2f}" if len(var_data) > 0 else 'N/A'
                            })
                    
                    pb_df = pd.DataFrame(pb_data)
                    st.dataframe(pb_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("**Session Averages**")
                    session_avgs = []
                    for session in df[session_col].unique():
                        session_df = df[df[session_col] == session]
                        session_avgs.append({
                            'Session': session,
                            'Count': len(session_df)
                        })
                    
                    session_avg_df = pd.DataFrame(session_avgs)
                    st.dataframe(session_avg_df, use_container_width=True, hide_index=True)
        
        # Session distribution
        with st.expander("📊 Session Distribution", expanded=False):
            session_counts = df[session_col].value_counts().reset_index()
            session_counts.columns = ['Session', 'Count']
            
            if athlete_filter and athlete_filter != 'All Athletes':
                title_text = f'Session Distribution for {athlete_filter}'
            else:
                title_text = 'Number of Observations per Session Type'
            
            fig = px.bar(
                session_counts,
                x='Session',
                y='Count',
                title=title_text,
                color='Count',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Athlete breakdown (only show when viewing all athletes)
        if (not athlete_filter or athlete_filter == 'All Athletes') and 'Full Name' in df.columns:
            with st.expander("👥 Athlete Breakdown", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Observations per Athlete**")
                    athlete_counts = df['Full Name'].value_counts().reset_index()
                    athlete_counts.columns = ['Athlete', 'Count']
                    st.dataframe(athlete_counts.head(20), use_container_width=True, height=400)
                
                with col2:
                    st.markdown("**Session Distribution by Top Athletes**")
                    # Get top 10 athletes by count
                    top_athletes = athlete_counts.head(10)['Athlete'].tolist()
                    athlete_session = df[df['Full Name'].isin(top_athletes)].groupby(['Full Name', session_col]).size().reset_index(name='Count')
                    
                    fig = px.bar(
                        athlete_session,
                        x='Full Name',
                        y='Count',
                        color=session_col,
                        title='Session Distribution - Top 10 Athletes',
                        barmode='stack'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Statistical Summary
        st.markdown('<p class="sub-header">📊 Statistical Summary</p>', unsafe_allow_html=True)
        
        # Tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["📋 Descriptive Statistics", "🧪 Statistical Tests", "📈 Effect Sizes"])
        
        with tab1:
            for var in selected_vars:
                with st.expander(f"📊 {var}", expanded=False):
                    stats_df = calculate_statistics(df, session_col, var)
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Quick visualization (with sampling)
                    sample_size_viz = min(10000, len(df))
                    df_sample = df.sample(n=sample_size_viz, random_state=42) if len(df) > sample_size_viz else df
                    
                    fig = px.box(df_sample, x=session_col, y=var, color=session_col,
                               title=f"{var} Distribution by Session")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.write("**Significance Testing (ANOVA/Kruskal-Wallis)**")
            
            test_results = []
            for var in selected_vars:
                test_type, p_value, significance = perform_statistical_test(df, session_col, var)
                if test_type:
                    test_results.append({
                        'Variable': var,
                        'Test': test_type,
                        'P-Value': f"{p_value:.4f}",
                        'Significance': significance
                    })
            
            if test_results:
                results_df = pd.DataFrame(test_results)
                
                # Color code by significance
                def highlight_significance(row):
                    if row['Significance'] == '***':
                        return ['background-color: #90EE90'] * len(row)
                    elif row['Significance'] == '**':
                        return ['background-color: #FFD700'] * len(row)
                    elif row['Significance'] == '*':
                        return ['background-color: #FFA500'] * len(row)
                    else:
                        return [''] * len(row)
                
                st.dataframe(
                    results_df.style.apply(highlight_significance, axis=1),
                    use_container_width=True
                )
                
                st.caption("*** p < 0.001 | ** p < 0.01 | * p < 0.05 | ns = not significant")
            else:
                st.info("Not enough data to perform statistical tests")
        
        with tab3:
            st.write("**Effect Size Analysis (Cohen's d)**")
            
            sessions = df[session_col].unique()
            
            if len(sessions) >= 2 and len(selected_vars) > 0:
                var_to_analyze = st.selectbox(
                    "Select variable for pairwise effect size comparison",
                    options=selected_vars
                )
                
                effect_sizes = []
                for i in range(len(sessions)):
                    for j in range(i+1, len(sessions)):
                        g1 = df[df[session_col] == sessions[i]][var_to_analyze].dropna()
                        g2 = df[df[session_col] == sessions[j]][var_to_analyze].dropna()
                        
                        if len(g1) > 0 and len(g2) > 0:
                            d = calculate_cohens_d(g1, g2)
                            effect = "Small" if abs(d) < 0.5 else "Medium" if abs(d) < 0.8 else "Large"
                            
                            effect_sizes.append({
                                'Comparison': f"{sessions[i]} vs {sessions[j]}",
                                "Cohen's d": f"{d:.3f}",
                                'Effect Size': effect,
                                'Abs(d)': abs(d)
                            })
                
                if effect_sizes:
                    effect_df = pd.DataFrame(effect_sizes)
                    effect_df = effect_df.sort_values('Abs(d)', ascending=False)
                    effect_df = effect_df.drop('Abs(d)', axis=1)
                    
                    st.dataframe(effect_df, use_container_width=True)
                    
                    st.caption("Small: |d| < 0.5 | Medium: 0.5 ≤ |d| < 0.8 | Large: |d| ≥ 0.8")
            else:
                st.info("Need at least 2 sessions to calculate effect sizes")
        
        # Visualizations
        st.markdown('<p class="sub-header">📊 Visualizations</p>', unsafe_allow_html=True)
        
        viz_type = st.selectbox(
            "Select Visualization Type",
            options=[
                "Box Plots",
                "Violin Plots",
                "Bar Chart (Mean ± SD)",
                "Radar Chart",
                "Correlation Heatmap",
                "Time Series"
            ]
        )
        
        if viz_type == "Box Plots":
            fig = create_boxplot(df, session_col, selected_vars)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Violin Plots":
            fig = create_violin_plot(df, session_col, selected_vars)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Bar Chart (Mean ± SD)":
            fig = create_bar_chart(df, session_col, selected_vars)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Radar Chart":
            if len(selected_vars) >= 3 and df[session_col].nunique() >= 2:
                fig = create_radar_chart(df, session_col, selected_vars)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Unable to create radar chart with current data")
            else:
                st.warning("Radar chart requires at least 3 variables and 2 session types")
        
        elif viz_type == "Correlation Heatmap":
            if len(selected_vars) >= 2:
                fig = create_correlation_heatmap(df, selected_vars)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Unable to create correlation heatmap")
            else:
                st.warning("Correlation heatmap requires at least 2 variables")
        
        elif viz_type == "Time Series":
            if date_col != 'None':
                fig = create_time_series(df, session_col, selected_vars, date_col)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Unable to parse dates for time series plot")
            else:
                st.warning("Please select a date column in the configuration section")
        
        # Export options
        st.markdown('<p class="sub-header">💾 Export Data</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export summary statistics
            summary_data = []
            for var in selected_vars:
                stats_df = calculate_statistics(df, session_col, var)
                for session in stats_df.index:
                    row = {
                        'Variable': var,
                        'Session': session,
                        'N': stats_df.loc[session, 'n'],
                        'Mean': stats_df.loc[session, 'mean'],
                        'Std': stats_df.loc[session, 'std'],
                        'Min': stats_df.loc[session, 'min'],
                        'Max': stats_df.loc[session, 'max'],
                        'Median': stats_df.loc[session, 'median']
                    }
                    # Add athlete name if filtering by specific athlete
                    if athlete_filter and athlete_filter != 'All Athletes':
                        row['Athlete'] = athlete_filter
                    
                    summary_data.append(row)
            
            summary_export = pd.DataFrame(summary_data)
            
            # Reorder columns to put Athlete first if it exists
            if 'Athlete' in summary_export.columns:
                cols = ['Athlete'] + [col for col in summary_export.columns if col != 'Athlete']
                summary_export = summary_export[cols]
            
            csv = summary_export.to_csv(index=False)
            
            # Create filename with athlete name if applicable
            if athlete_filter and athlete_filter != 'All Athletes':
                filename = f"summary_stats_{athlete_filter.replace(' ', '_')}.csv"
            else:
                filename = "session_summary_statistics.csv"
            
            st.download_button(
                label="📥 Download Summary Statistics (CSV)",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
        
        with col2:
            # Export filtered data - always include names
            export_cols = []
            
            # Always include athlete names if columns exist
            if 'First Name' in df.columns:
                export_cols.append('First Name')
            if 'Last Name' in df.columns:
                export_cols.append('Last Name')
            
            export_cols.append(session_col)
            
            # Add date if selected
            if date_col != 'None' and date_col in df.columns:
                export_cols.append(date_col)
            
            export_cols.extend(selected_vars)
            
            # Only include columns that exist in the dataframe
            export_cols = [col for col in export_cols if col in df.columns]
            
            filtered_df = df[export_cols].copy()
            csv = filtered_df.to_csv(index=False)
            
            # Create filename with athlete name if applicable
            if athlete_filter and athlete_filter != 'All Athletes':
                filename = f"filtered_data_{athlete_filter.replace(' ', '_')}.csv"
            else:
                filename = "session_filtered_data.csv"
            
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
    
    else:
        # No data loaded
        st.info("👈 Please configure filters and click 'Load/Reload Data' to begin analysis")
        
        # Instructions
        st.markdown("### 📚 Getting Started")
        st.markdown("""
        1. **Configure filters** in the sidebar (sample size, date range)
        2. **Click "Load/Reload Data"** to load the dataset
        3. **Select variables** you want to analyze
        4. **Explore** the interactive visualizations and statistics
        5. **Export** your results
        
        #### Memory Management:
        - For large datasets (millions of rows), use sampling
        - Start with 100,000 rows and increase if needed
        - Use date filters to focus on recent data
        - Visualizations automatically sample data when needed
        
        #### Example Variables:
        - Jump Height
        - Peak Propulsive Power
        - RSI (Reactive Strength Index)
        - Takeoff Velocity
        - Peak Force
        - Countermovement Depth
        """)


if __name__ == "__main__":
    main()