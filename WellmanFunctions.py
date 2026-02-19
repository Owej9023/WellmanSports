#sandbox
import zipfile
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

import os
def create_dataframes():
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

    return(merged_df)

merged_df = create_dataframes()


def analyze_targets(merged_df,targets):
    if len(targets) > 0:
        print(f"Analyzing targets: {','.join(targets)}")

    # encode categorical/object columns
    cat_cols = merged_df.select_dtypes(include=['object','category']).columns.tolist()
    mappings = {}
    for col in cat_cols:
        codes, uniques = pd.factorize(merged_df[col], sort=True)
        merged_df[col + "_code"] = codes
        mappings[col] = {str(u): int(i) for i, u in enumerate(uniques)}
    merged_df = merged_df.drop(columns=cat_cols)

    # ensure numeric targets where present
    present_targets = [t for t in targets if t in merged_df.columns]
    for t in present_targets:
        merged_df[t] = pd.to_numeric(merged_df[t], errors='coerce')

    if not present_targets:
        print("None of the requested targets were found in merged_df columns.")
    else:
        for target in present_targets:
            print(f"\n{'='*60}")
            print(f"Analyzing: {target}")
            print('='*60)
            
            # prepare X/y
            numeric = merged_df.select_dtypes(include=[np.number]).copy()
            if target not in numeric.columns:
                print(f"Skipping {target}: not numeric after coercion.")
                continue
            
            X = numeric.drop(columns=[target], errors='ignore')
            y = numeric[target]
            
            # Filter to rows with valid target values
            mask = y.notna()
            X_valid = X.loc[mask]
            y_valid = y.loc[mask]
            
            print(f"Rows with valid target values: {len(y_valid)}")
            
            if len(y_valid) < 10:
                print(f"Insufficient data: need at least 10 rows, have {len(y_valid)}")
                continue
            
            # Remove features with >80% missing values
            missing_threshold = 0.8
            missing_frac = X_valid.isna().sum() / len(X_valid)
            features_to_keep = missing_frac[missing_frac < missing_threshold].index
            X_filtered = X_valid[features_to_keep]
            
            print(f"Features after removing sparse columns: {X_filtered.shape[1]} (removed {X_valid.shape[1] - X_filtered.shape[1]})")
            
            if X_filtered.shape[1] == 0:
                print("No features remaining after filtering.")
                continue
            
            # Impute remaining missing values with median
            imputer = SimpleImputer(strategy='median')
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X_filtered),
                columns=X_filtered.columns,
                index=X_filtered.index
            )
            
            print(f"Final dataset: {X_imputed.shape[0]} rows × {X_imputed.shape[1]} features")
            
            # Fit Random Forest and extract importances
            rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
            rf.fit(X_imputed, y_valid)
            importances = pd.Series(rf.feature_importances_, index=X_imputed.columns).sort_values(ascending=False)
            
            print(f"\nTop 10 features for predicting '{target}':")
            print(importances.head(10))
            print(f"\nModel R² score: {rf.score(X_imputed, y_valid):.3f}")



def randomforestCode(df, target):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import LabelEncoder
    from scipy import stats


    df["Date Completed"] = pd.to_datetime(df["Date Completed"], errors='coerce')
    df["Blocks Completed"] = pd.to_numeric(df["Blocks Completed"], errors="coerce")
    df["Blocks Prescribed"] = pd.to_numeric(df["Blocks Prescribed"], errors="coerce")
    df["Reps Volume (LB)"] = pd.to_numeric(df["Reps"], errors="coerce")
    df["Duration (min)"] = pd.to_numeric(df["Duration (min)"], errors="coerce")

    # # ============================================================================
    # # CORRELATION MATRIX
    # # ============================================================================
    # print("="*80)
    # print("CORRELATION MATRIX")
    # print("="*80)

    # corr2 = df.select_dtypes(include=['number']).corr()
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(corr2, annot=True, fmt=".2f", cmap='viridis', cbar=True, square=True)
    # plt.title('Correlation Matrix for Training Data Metrics', fontsize=14, fontweight='bold')
    # plt.tight_layout()
    # # plt.show()

    # # ============================================================================
    # # ANALYZE EACH TARGET VARIABLE
    # # ============================================================================
    # targets = ["Reps Volume (LB)", "Duration (min)", "Blocks Prescribed", "Blocks Completed"]
    # num = df.select_dtypes(include=[np.number])

    # for t in targets:
    #     print("\n" + "="*80)
    #     print(f"ANALYZING TARGET: {t}")
    #     print("="*80)
        
    #     if t not in num.columns:
    #         print(f"❌ Missing target: {t}")
    #         continue
        
    #     # Get clean target data
    #     y = num[t].dropna()
    #     X = num.loc[y.index].drop(columns=[t], errors='ignore')
        
    #     print(f"✓ Valid observations: {len(y)}")
    #     print(f"✓ Available features: {X.shape[1]}")
        
    #     if X.shape[0] < 10:
    #         print(f"❌ Insufficient rows for {t} (need 10+, have {X.shape[0]})")
    #         continue
        
    #     # -------------------------------------------------------------------------
    #     # 1. TOP CORRELATIONS
    #     # -------------------------------------------------------------------------
    #     print(f"\n📊 Top 10 Correlations with {t}:")
    #     print("-" * 60)
    #     corrs = X.corrwith(y).abs().sort_values(ascending=False)
    #     for feat, val in corrs.head(10).items():
    #         print(f"  {feat:.<50} {val:.3f}")
        
    #     # -------------------------------------------------------------------------
    #     # 2. FEATURE IMPORTANCE (Random Forest)
    #     # -------------------------------------------------------------------------
    #     # Filter sparse columns (>80% missing)
    #     missing_threshold = 0.8
    #     keep = (X.isna().sum() / len(X)) < missing_threshold
    #     Xf = X.loc[:, keep]
        
    #     print(f"\n✓ Features after filtering: {Xf.shape[1]} (removed {X.shape[1] - Xf.shape[1]} sparse features)")
        
    #     if Xf.shape[1] == 0:
    #         print("❌ No features remaining after filtering")
    #         continue
        
    #     # Impute missing values
    #     imputer = SimpleImputer(strategy='median')
    #     X_imputed = pd.DataFrame(
    #         imputer.fit_transform(Xf),
    #         columns=Xf.columns,
    #         index=Xf.index
    #     )
        
    #     # Train Random Forest
    #     rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    #     rf.fit(X_imputed, y)
        
    #     # Get permutation importance (more reliable than feature_importances_)
    #     print(f"\n🌲 Computing permutation importance...")
    #     pi = permutation_importance(
    #         rf, X_imputed, y,
    #         n_repeats=20,
    #         random_state=0,
    #         n_jobs=-1
    #     )
    #     imp = pd.Series(pi.importances_mean, index=X_imputed.columns).sort_values(ascending=False)
        
    #     print(f"\n🎯 Top 10 Feature Importances for {t}:")
    #     print("-" * 60)
    #     for feat, val in imp.head(10).items():
    #         print(f"  {feat:.<50} {val:.4f}")
        
    #     r2_score = rf.score(X_imputed, y)
    #     print(f"\n📈 Model R² Score: {r2_score:.3f}")
        
    #     if r2_score < 0.3:
    #         print("   ⚠️  Low R² suggests weak predictive relationships")
    #     elif r2_score > 0.7:
    #         print("   ✓ Strong predictive model")


    # # ============================================================================
    # # SESSION TITLE ANALYSIS
    # # ============================================================================
    # print("\n" + "="*80)
    # print("SESSION TITLE ANALYSIS")
    # print("="*80)

    # if "sessionTitle" not in df.columns:
    #     print("❌ 'sessionTitle' column not found")
    # else:
    #     # -------------------------------------------------------------------------
    #     # 1. SUMMARY STATISTICS BY SESSION
    #     # -------------------------------------------------------------------------
    #     print("\n📋 Average Metrics by Session Type:")
    #     print("-" * 80)
    #     grp = df.groupby("sessionTitle")[targets].mean()
    #     print(grp.to_string())
        
    #     # -------------------------------------------------------------------------
    #     # 2. ANOVA TESTS
    #     # -------------------------------------------------------------------------
    #     print("\n" + "="*80)
    #     print("ANOVA: Testing if session type affects metrics")
    #     print("="*80)
        
    #     for col in targets:
    #         if col not in df.columns:
    #             continue
            
    #         # Get groups with at least 5 observations
    #         groups = [
    #             g[col].dropna().values 
    #             for _, g in df.groupby("sessionTitle") 
    #             if g[col].notna().sum() >= 5
    #         ]
            
    #         if len(groups) >= 2:
    #             f_stat, p_value = stats.f_oneway(*groups)
    #             significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    #             print(f"\n{col}:")
    #             print(f"  F-statistic: {f_stat:.3f}")
    #             print(f"  p-value: {p_value:.3e} {significance}")
                
    #             if p_value < 0.05:
    #                 print(f"  → Session type significantly affects {col}")
    #             else:
    #                 print(f"  → No significant difference across session types")
    #         else:
    #             print(f"\n{col}: Insufficient groups for ANOVA")
        
    #     # -------------------------------------------------------------------------
    #     # 3. PREDICT SESSION TYPE FROM METRICS
    #     # -------------------------------------------------------------------------
    #     print("\n" + "="*80)
    #     print("PREDICTING SESSION TYPE FROM METRICS")
    #     print("="*80)
        
    #     le = LabelEncoder()
    #     mask = df["sessionTitle"].notna()
    #     Xc = df.loc[mask].select_dtypes(include=[np.number]).copy()
    #     y_c = le.fit_transform(df.loc[mask, "sessionTitle"])
        
    #     print(f"\n✓ Training samples: {len(y_c)}")
    #     print(f"✓ Session types: {len(le.classes_)} ({', '.join(le.classes_)})")
    #     print(f"✓ Features: {Xc.shape[1]}")
        
    #     if Xc.shape[0] >= 50 and Xc.shape[1] > 0:
    #         # Impute missing values
    #         Xc_imputed = pd.DataFrame(
    #             SimpleImputer(strategy='median').fit_transform(Xc),
    #             columns=Xc.columns
    #         )
            
    #         # Train classifier (using RandomForestClassifier not Regressor)
    #         rc = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    #         rc.fit(Xc_imputed, y_c)
            
    #         # Get feature importances
    #         impc = pd.Series(
    #             rc.feature_importances_,
    #             index=Xc_imputed.columns
    #         ).sort_values(ascending=False)
            
    #         accuracy = rc.score(Xc_imputed, y_c)
            
    #         print(f"\n🎯 Top 10 Features Predicting Session Type:")
    #         print("-" * 60)
    #         for feat, val in impc.head(10).items():
    #             print(f"  {feat:.<50} {val:.4f}")
            
    #         print(f"\n📈 Classifier Accuracy: {accuracy:.3f}")
    #     else:
    #         print(f"❌ Insufficient data for classification (need 50+ samples, have {Xc.shape[0]})")

    # print("\n" + "="*80)
    # print("ANALYSIS COMPLETE")
    # print("="*80)
    # print(merged_df.head())
    # merged_df = pd.merge(merged_df, workouts,
    #                      left_on=['first_name', 'last_name'],
    #                         right_on=['First Name', 'Last Name'],
    #                         how='inner')

    # print(merged_df.head())


    
# print(df.head())
# targets to evaluate
#x = analyze_targets(merged_df, targets=["Reps Volume (LB)", "Duration (min)", "Blocks Prescribed", "Blocks Completed"])

# # --- Lift trend and sessionTitle analysis ---------------------------------
# import re
# import seaborn as sns
# import matplotlib.pyplot as plt


# mask = df['Exercise'].str.contains(r'\b(deadlift|back squat|trap deadlift)\b', case=False, na=False)
# selected = df.loc[mask].copy()

# # ----- Analyze interaction with sessionTitle and Last Date Estimated 1RM (LB) -----
# try:
#     import scipy.stats as stats

#     sel = selected.copy()
#     # ensure datetime and numeric types
#     if 'Date Completed' in sel.columns:
#         sel['Date Completed'] = pd.to_datetime(sel['Date Completed'], errors='coerce')
#     if 'Last Date Estimated 1RM (LB)' in sel.columns:
#         sel['Last Date Estimated 1RM (LB)'] = pd.to_numeric(sel['Last Date Estimated 1RM (LB)'], errors='coerce')

#     # Summary by sessionTitle for 1RM
#     if 'sessionTitle' in sel.columns and 'Last Date Estimated 1RM (LB)' in sel.columns:
#         grp = sel.groupby('sessionTitle')['Last Date Estimated 1RM (LB)']
#         stats_by_session = grp.agg(['count', 'mean', 'std', 'min', 'max']).sort_values('count', ascending=False)
#         print('\nSessionTitle summary for Last Date Estimated 1RM (LB):')
#         print(stats_by_session.to_string())

#     # Time series of mean 1RM over time (weekly)
#     if 'Date Completed' in sel.columns and 'Last Date Estimated 1RM (LB)' in sel.columns:
#         ts = sel.set_index('Date Completed').resample('W')['Last Date Estimated 1RM (LB)'].mean().dropna()
#         if not ts.empty:
#             plt.figure(figsize=(10, 4))
#             plt.plot(ts.index, ts.values, marker='o')
#             plt.title('Weekly Mean Last Date Estimated 1RM (LB)')
#             plt.ylabel('1RM (LB)')
#             plt.tight_layout()
#             plt.savefig('weekly_mean_1RM.png')
#             plt.close()

#             # linear trend test
#             x = (ts.index - ts.index[0]).days.values
#             slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, ts.values)
#             print(f"\nWeekly 1RM trend: slope={slope:.4f} LB/day, r={rvalue:.3f}, p={pvalue:.3e}")

#     # Detect other numeric columns with significant change over time
#     numeric = sel.select_dtypes(include=[np.number]).columns.tolist()
#     sig_changes = []
#     if 'Date Completed' in sel.columns and len(numeric) > 0:
#         for col in numeric:
#             # require some non-null values with dates
#             tmp = sel[['Date Completed', col]].dropna()
#             if len(tmp) < 10:
#                 continue
#             tmp = tmp.set_index('Date Completed').resample('D').mean().dropna()
#             if len(tmp) < 10:
#                 continue
#             x = (tmp.index - tmp.index[0]).days.values
#             slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, tmp[col].values)
#             if pvalue < 0.05:
#                 sig_changes.append((col, slope, rvalue, pvalue))

#     if sig_changes:
#         sig_changes.sort(key=lambda x: x[3])
#         print('\nNumeric columns with significant time trends (p < 0.05):')
#         for col, slope, r, p in sig_changes:
#             print(f"  {col:.<40} slope={slope:.4f}, r={r:.3f}, p={p:.3e}")
#             # save a plot for top few
#         for col, slope, r, p in sig_changes[:3]:
#             tmp = sel[['Date Completed', col]].dropna().set_index('Date Completed').resample('W').mean().dropna()
#             if tmp.empty:
#                 continue
#             plt.figure(figsize=(8,3))
#             plt.plot(tmp.index, tmp[col].values, marker='o')
#             plt.title(f'Weekly mean: {col}')
#             plt.tight_layout()
#             fname = f"trend_{col.replace(' ', '_').replace('/', '_')}.png"
#             plt.savefig(fname)
#             plt.close()
#     else:
#         print('\nNo numeric columns found with significant time trends (p < 0.05).')
# except Exception as e:
#     print('Error during session/1RM analysis:', e)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# # Set style
# sns.set_style("whitegrid")
# plt.rcParams['figure.dpi'] = 100

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# # Set style
# sns.set_style("whitegrid")
# plt.rcParams['figure.dpi'] = 100

def analyze_lift_trends(df, first_name=None):
    """
    Comprehensive analysis of lift trends and stat relationships over time
    
    Parameters:
    -----------
    df : DataFrame
        The workout data
    first_name : str, optional
        Filter data by First Name. If None, analyzes all data.
    """
    
    # Filter by First Name if provided
    if first_name is not None:
        if 'First Name' in df.columns:
            df = df[df['First Name'].str.lower() == first_name.lower()].copy()
            print(f"Filtering data for: {first_name}")
            print(f"Records found: {len(df)}")
        else:
            print("Warning: 'First Name' column not found in data. Analyzing all data.")
    
    # Filter for main lifts
    mask = df['Exercise'].str.contains(r'\b(deadlift|back squat|trap deadlift)\b', 
                                       case=False, na=False)
    selected = df.loc[mask].copy()
    
    # Ensure proper data types
    if 'Date Completed' in selected.columns:
        selected['Date Completed'] = pd.to_datetime(selected['Date Completed'], errors='coerce')
    if 'Last Date Estimated 1RM (LB)' in selected.columns:
        selected['Last Date Estimated 1RM (LB)'] = pd.to_numeric(
            selected['Last Date Estimated 1RM (LB)'], errors='coerce')
    
    print("="*80)
    print(f"LIFT TREND ANALYSIS{' - ' + first_name.upper() if first_name else ''}")
    print("="*80)
    print(f"\nTotal records analyzed: {len(selected)}")
    print(f"Date range: {selected['Date Completed'].min()} to {selected['Date Completed'].max()}")
    
    # ========== 1. Session Title Summary ==========
    if 'sessionTitle' in selected.columns and 'Last Date Estimated 1RM (LB)' in selected.columns:
        print("\n" + "="*80)
        print("SESSION TITLE ANALYSIS")
        print("="*80)
        
        grp = selected.groupby('sessionTitle')['Last Date Estimated 1RM (LB)']
        stats_by_session = grp.agg(['count', 'mean', 'std', 'min', 'max']).sort_values('count', ascending=False)
        print('\n1RM Statistics by Session Type:')
        print(stats_by_session.to_string())
        
        # Visualize session comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot
        session_data = selected[['sessionTitle', 'Last Date Estimated 1RM (LB)']].dropna()
        session_data.boxplot(column='Last Date Estimated 1RM (LB)', 
                            by='sessionTitle', ax=axes[0])
        title = f"1RM Distribution by Session Type{' - ' + first_name if first_name else ''}"
        axes[0].set_title(title)
        axes[0].set_xlabel('Session Type')
        axes[0].set_ylabel('1RM (LB)')
        plt.sca(axes[0])
        plt.xticks(rotation=45, ha='right')
        
        # Mean comparison
        means = stats_by_session['mean'].sort_values(ascending=False)
        axes[1].barh(range(len(means)), means.values)
        axes[1].set_yticks(range(len(means)))
        axes[1].set_yticklabels(means.index)
        axes[1].set_xlabel('Mean 1RM (LB)')
        title = f"Average 1RM by Session Type{' - ' + first_name if first_name else ''}"
        axes[1].set_title(title)
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # ========== 2. Time Series Analysis ==========
    print("\n" + "="*80)
    print("TIME SERIES ANALYSIS")
    print("="*80)
    
    numeric_cols = selected.select_dtypes(include=[np.number]).columns.tolist()
    important_metrics = ['Last Date Estimated 1RM (LB)', 'Weight (LB)', 
                        'Reps', 'Distance', 'Duration (Seconds)']
    
    # Filter to existing important metrics
    metrics_to_analyze = [col for col in important_metrics if col in numeric_cols]
    
    if 'Date Completed' in selected.columns and metrics_to_analyze:
        # Create comprehensive time series plots
        n_metrics = len(metrics_to_analyze)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        trend_results = []
        
        for idx, col in enumerate(metrics_to_analyze):
            # Daily and weekly aggregation
            daily = selected[['Date Completed', col]].dropna().set_index('Date Completed')
            weekly = daily.resample('W').agg(['mean', 'min', 'max', 'count'])
            
            if len(weekly) < 2:
                continue
            
            # Plot weekly trends with confidence bands
            ax = axes[idx]
            ax.plot(weekly.index, weekly[(col, 'mean')], 
                   marker='o', linewidth=2, label='Weekly Mean', color='#2E86AB')
            ax.fill_between(weekly.index, 
                           weekly[(col, 'min')], 
                           weekly[(col, 'max')], 
                           alpha=0.2, color='#2E86AB', label='Min-Max Range')
            
            # Add trend line
            x_numeric = (weekly.index - weekly.index[0]).days.values
            y_values = weekly[(col, 'mean')].values
            
            # Remove NaN values for regression
            mask = ~np.isnan(y_values)
            if mask.sum() >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x_numeric[mask], y_values[mask])
                
                trend_line = slope * x_numeric + intercept
                ax.plot(weekly.index, trend_line, '--', 
                       color='red', linewidth=2, alpha=0.7,
                       label=f'Trend (slope={slope:.3f}/day, p={p_value:.3e})')
                
                trend_results.append({
                    'Metric': col,
                    'Slope': slope,
                    'R-value': r_value,
                    'P-value': p_value,
                    'Significant': p_value < 0.05
                })
            
            ax.set_title(f'{col} Over Time{" - " + first_name if first_name else ""}', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel(col)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add count annotations
            ax2 = ax.twinx()
            ax2.bar(weekly.index, weekly[(col, 'count')], 
                   alpha=0.15, color='gray', width=5)
            ax2.set_ylabel('Weekly Count', alpha=0.6)
            ax2.tick_params(axis='y', labelcolor='gray')
        
        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        plt.show()
        
        # Print trend summary
        if trend_results:
            print("\nTrend Analysis Summary:")
            print("-" * 80)
            trend_df = pd.DataFrame(trend_results)
            trend_df = trend_df.sort_values('P-value')
            for _, row in trend_df.iterrows():
                sig = "***" if row['Significant'] else ""
                print(f"{row['Metric']:.<45} slope={row['Slope']:>8.4f}, "
                      f"r={row['R-value']:>6.3f}, p={row['P-value']:.3e} {sig}")
    
    # ========== 3. Correlation Analysis ==========
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Select numeric columns for correlation
    corr_cols = [col for col in numeric_cols if col in selected.columns]
    if len(corr_cols) >= 2:
        corr_data = selected[corr_cols].dropna()
        
        if len(corr_data) > 10:
            # Compute correlation matrix
            corr_matrix = corr_data.corr()
            
            # Plot correlation heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, vmin=-1, vmax=1,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            title = f'Correlation Matrix of Training Metrics{" - " + first_name if first_name else ""}'
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.show()
            
            # Find strongest correlations
            print("\nStrongest Correlations (|r| > 0.5):")
            print("-" * 80)
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        print(f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: "
                              f"r = {corr_val:.3f}")
    
    # ========== 4. Progressive Overload Analysis ==========
    if 'Last Date Estimated 1RM (LB)' in selected.columns:
        print("\n" + "="*80)
        print("PROGRESSIVE OVERLOAD ANALYSIS")
        print("="*80)
        
        # Calculate rolling changes
        ts_1rm = selected[['Date Completed', 'Last Date Estimated 1RM (LB)']].dropna()
        ts_1rm = ts_1rm.sort_values('Date Completed')
        ts_1rm['Week'] = ts_1rm['Date Completed'].dt.to_period('W')
        weekly_max = ts_1rm.groupby('Week')['Last Date Estimated 1RM (LB)'].max()
        
        if len(weekly_max) > 4:
            weekly_max_series = weekly_max.to_timestamp()
            
            # Calculate percentage changes
            pct_changes = weekly_max.pct_change() * 100
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Weekly max 1RM
            axes[0].plot(weekly_max_series.index, weekly_max_series.values, 
                        marker='o', linewidth=2, color='#A23B72')
            title = f'Weekly Maximum 1RM Progression{" - " + first_name if first_name else ""}'
            axes[0].set_title(title, fontsize=12, fontweight='bold')
            axes[0].set_ylabel('1RM (LB)')
            axes[0].grid(True, alpha=0.3)
            
            # Week-over-week change
            pct_series = pct_changes.to_timestamp()
            colors = ['green' if x >= 0 else 'red' for x in pct_changes.values]
            axes[1].bar(pct_series.index[1:], pct_changes.values[1:], 
                       color=colors, alpha=0.6, width=5)
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            title = f'Week-over-Week 1RM Change (%){" - " + first_name if first_name else ""}'
            axes[1].set_title(title, fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Change (%)')
            axes[1].set_xlabel('Date')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Summary statistics
            print(f"\nTotal 1RM increase: {weekly_max.iloc[-1] - weekly_max.iloc[0]:.1f} LB")
            print(f"Percent increase: {((weekly_max.iloc[-1] / weekly_max.iloc[0]) - 1) * 100:.1f}%")
            print(f"Average weekly change: {pct_changes.mean():.2f}%")
            print(f"Weeks with increase: {(pct_changes > 0).sum()} / {len(pct_changes)}")
    
    # ========== 5. Exercise-Specific Comparison ==========
    if 'Exercise' in selected.columns:
        print("\n" + "="*80)
        print("EXERCISE-SPECIFIC ANALYSIS")
        print("="*80)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for exercise in selected['Exercise'].unique():
            ex_data = selected[selected['Exercise'] == exercise]
            if 'Last Date Estimated 1RM (LB)' in ex_data.columns:
                weekly = ex_data[['Date Completed', 'Last Date Estimated 1RM (LB)']].dropna()
                weekly = weekly.set_index('Date Completed').resample('W').mean()
                
                if len(weekly) >= 2:
                    ax.plot(weekly.index, weekly.values, 
                           marker='o', linewidth=2, label=exercise, alpha=0.8)
        
        title = f'1RM Progression by Exercise Type{" - " + first_name if first_name else ""}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('1RM (LB)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    print("\n" + "="*80)
    print("Analysis complete! All visualizations displayed.")
    print("="*80)
    
    return selected

# Example usage:
# df = pd.read_csv('your_workout_data.csv')
# 
# # Analyze all data
# results = analyze_lift_trends(df)
#
# # Analyze specific person
# results = analyze_lift_trends(df, first_name='John')
# Example usage:
# df = pd.read_csv('your_workout_data.csv')
# results = analyze_lift_trends(df)
