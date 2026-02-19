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
from pathlib import Path
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


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
def create_dataframes(training_path, workout_path, spartan_folder, 
                     sample_size=None, date_filter=None, session_filter=None):
    """
    Create merged dataframe from training, workout, and Spartan data
    
    Args:
        training_path: Path to training tracker CSV
        workout_path: Path to workouts/exercises CSV
        spartan_folder: Path to folder containing Spartan data zips
        sample_size: Optional number of rows to sample
        date_filter: Optional number of days to filter (e.g., 30 for last 30 days)
        session_filter: Optional session type to filter
    """
    
    try:
        # -----------------------
        # Load base CSVs
        # -----------------------
        training_df = pd.read_csv(training_path)
        workouts = pd.read_csv(workout_path)
        
        # -----------------------
        # Load Spartan Data from ZIPs
        # -----------------------
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

        # Trim columns
        col_cut = 'P1|P2 Propulsive Impulse Index'
        if col_cut in foreplate_df.columns:
            cut_idx = foreplate_df.columns.get_loc(col_cut)
            foreplate_df = foreplate_df.iloc[:, :cut_idx+1]
        else:
            st.warning(f"Column not found: {col_cut}. Keeping all columns.")

        # -----------------------
        # Rename columns
        # -----------------------
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
        # Create Full Name
        # -----------------------
        training_df['Full Name'] = (
            training_df['first_name'].astype(str).str.strip() + ' ' +
            training_df['last_name'].astype(str).str.strip()
        )

        workouts['Full Name'] = (
            workouts['first_name'].astype(str).str.strip() + ' ' +
            workouts['last_name'].astype(str).str.strip()
        )

        foreplate_df = foreplate_df.rename(columns={'Name': 'Full Name'})

        # Standardize Full Name across all dataframes
        for df in [training_df, workouts, foreplate_df]:
            df['Full Name'] = (
                df['Full Name']
                .astype(str)
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True)
                .str.upper()
            )

        # -----------------------
        # Drop unused columns
        # -----------------------
        workouts = workouts.drop(columns=["first_name", "last_name", "Unnamed: 7"], errors='ignore')
        training_df = training_df.drop(columns=["first_name", "last_name"], errors='ignore')

        # -----------------------
        # Standardize dates
        # -----------------------
        foreplate_df['Date Completed'] = pd.to_datetime(foreplate_df['Date'], errors='coerce')
        foreplate_df = foreplate_df.drop(columns=["Date"], errors='ignore')
        training_df['Date Completed'] = pd.to_datetime(training_df['Date Completed'], errors='coerce')
        workouts['Date Completed'] = pd.to_datetime(workouts['Date Completed'], errors='coerce')

        # Normalize dates (remove time component)
        for df in [foreplate_df, training_df, workouts]:
            df['Date Completed'] = df['Date Completed'].dt.normalize()

        # -----------------------
        # Apply date filter (BEFORE merge for efficiency)
        # -----------------------
        if date_filter:
            cutoff_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=date_filter)
            st.info(f"📅 Filtering data from {cutoff_date.date()} onwards")
            
            foreplate_df = foreplate_df[foreplate_df['Date Completed'] >= cutoff_date]
            training_df = training_df[training_df['Date Completed'] >= cutoff_date]
            workouts = workouts[workouts['Date Completed'] >= cutoff_date]
            
            st.write(f"After date filter - Spartan: {len(foreplate_df):,}, Training: {len(training_df):,}, Workouts: {len(workouts):,}")

        # -----------------------
        # Merge the dataframes on Full Name and Date Completed
        # -----------------------
        merged_df = (
            foreplate_df
            .merge(training_df, on=['Full Name'], how='inner')
            .merge(workouts, on=['Full Name'], how='inner')
        )

        st.write(f"After merge: {len(merged_df):,} rows")

        # -----------------------
        # Clean and drop columns
        # -----------------------
        merged_df.replace(['-', 'N/A', ''], np.nan, inplace=True)
        
        # Exclude columns not interested in
        merged_df = merged_df.drop(
            columns=["Peak Landing Force", "Jump Momentum", "mRSI", "Readiness", "Intensity"],
            errors='ignore'
        )

        # -----------------------
        # Apply session filter (AFTER merge)
        # -----------------------
        if session_filter and session_filter != 'All Sessions':
            if 'Session Title' in merged_df.columns:
                initial_count = len(merged_df)
                merged_df = merged_df[merged_df['Session Title'] == session_filter]
                st.write(f"Session filter applied: {initial_count:,} → {len(merged_df):,} rows")

        # -----------------------
        # Apply sampling if requested
        # -----------------------
        if sample_size and sample_size < len(merged_df):
            st.write(f"Sampling {sample_size:,} rows from {len(merged_df):,}")
            merged_df = merged_df.sample(n=sample_size, random_state=42)

        # -----------------------
        # Memory optimization
        # -----------------------
        for col in merged_df.select_dtypes(include=['float64']).columns:
            merged_df[col] = pd.to_numeric(merged_df[col], downcast='float')

        for col in merged_df.select_dtypes(include=['int64']).columns:
            merged_df[col] = pd.to_numeric(merged_df[col], downcast='integer')

        st.write(f"✅ Final dataset: {len(merged_df):,} rows, {len(merged_df.columns)} columns")
        return merged_df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
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
        'Braking Impulse', 'Propulsive Impulse','Weight (LB)', 'Exercise Reps','Volume (LB)','Reps'
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

def program_impact_analysis(df, var, date_col, session_col, athlete_filter=None):
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[date_col, var])
    df_copy = df_copy.sort_values(date_col)

    if len(df_copy) < 5:
        return None, None

    # Aggregate daily mean
    daily = df_copy.groupby(date_col)[var].mean().reset_index()
    daily = daily.sort_values(date_col)

    # Rolling 7-day average
    daily['Rolling_7'] = daily[var].rolling(7, min_periods=1).mean()

    # Time index
    daily['Time_Index'] = (daily[date_col] - daily[date_col].min()).dt.days

    # Linear regression
    X = daily[['Time_Index']]
    y = daily[var]

    model = LinearRegression()
    model.fit(X, y)

    daily['Trend'] = model.predict(X)

    slope_per_day = model.coef_[0]
    slope_per_week = slope_per_day * 7

    # Percent change from baseline
    baseline = daily[var].iloc[0]
    daily['Percent_Change'] = ((daily[var] - baseline) / baseline) * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily[date_col],
        y=daily[var],
        mode='markers',
        name='Mean Daily Value',
        marker=dict(size=8, color='blue', opacity=0.6)
    ))

    fig.add_trace(go.Scatter(
        x=daily[date_col],
        y=daily['Rolling_7'],
        mode='lines',
        name='7-Day Rolling Avg'
    ))

    fig.add_trace(go.Scatter(
        x=daily[date_col],
        y=daily['Trend'],
        mode='lines',
        name=f'Trend (+{slope_per_week:.3f} per week)'
    ))

    fig.update_layout(
        title=f"{var} Program Impact Over Time",
        height=500
    )

    return fig, slope_per_week, daily


def mixed_effects_program_model(df, var, date_col, session_col):
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[date_col, var, 'Full Name'])

    df_copy['Time_Index'] = (
        df_copy[date_col] - df_copy[date_col].min()
    ).dt.days

    if df_copy['Full Name'].nunique() < 2:
        return None

    formula = f"Q('{var}') ~ Time_Index + C(Q('{session_col}'))"

    try:
        model = smf.mixedlm(
            formula,
            df_copy,
            groups=df_copy["Full Name"]
        )
        result = model.fit()
        return result
    except:
        return None


# Main App added to github
def main():
    st.markdown('<p class="main-header">🏃 Session Title Impact Analysis</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Configuration")
        
        # File paths
        st.subheader("📂 File Paths")
        
        training_path = st.text_input(
            "Training Tracker CSV Path",
            value="C:/Users/dickh/Downloads/WellmanSports/WellmanWorkoutTracker1-28-2026.csv",
            help="Path to the training tracker CSV file"
        )
        
        workout_path = st.text_input(
            "Workouts/Exercises CSV Path",
            value="C:/Users/dickh/Downloads/WellmanSports/WellmanExercises.csv",
            help="Path to the exercises CSV file"
        )
        
        spartan_folder = st.text_input(
            "Spartan Data Folder Path",
            value="C:/Users/dickh/Downloads/WellmanSports/SpartanData",
            help="Path to folder containing Spartan data ZIP files"
        )
        
        # Validate paths
        paths_valid = all([
            Path(training_path).exists() if training_path else False,
            Path(workout_path).exists() if workout_path else False,
            Path(spartan_folder).exists() if spartan_folder else False
        ])
        
        if not paths_valid:
            st.warning("⚠️ One or more file paths don't exist. Please verify paths.")
        
        st.markdown("---")
        st.subheader("🎚️ Data Filtering")
        
        # Data size controls
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
        if st.button("🔄 Load/Reload Data", type="primary", disabled=not paths_valid):
            st.session_state.data_loaded = False
            st.cache_data.clear()
    
    # Load data
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if not st.session_state.data_loaded:
        if not paths_valid:
            st.error("❌ Invalid file paths. Please check the sidebar configuration.")
            return
            
        with st.spinner("Loading data..."):
            df = create_dataframes(
                training_path=training_path,
                workout_path=workout_path,
                spartan_folder=spartan_folder,
                sample_size=sample_size if sample_size > 0 else None,
                date_filter=date_filter
            )
            
            if df is not None and len(df) > 0:
                st.session_state.df = df
                st.session_state.data_loaded = True
                summary = get_data_summary(df)
                st.success(f"✅ Data loaded! {summary['total_rows']:,} rows, {summary['total_columns']} columns, {summary['memory_usage_mb']:.1f} MB in memory")
            else:
                st.error("Failed to load data or no data after filtering")
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
        
        # Use Full Name if available
        athlete_filter = None
        if 'Full Name' in df.columns:
            
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
            st.warning("⚠️ Full Name column not found in dataset. Cannot filter by athlete.")
        
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
                "Time Series",
                "Program Impact Over Time",

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
        elif viz_type == "Program Impact Over Time":

            if date_col == 'None':
                st.warning("Please select a date column in the configuration section.")
            else:
                var = st.selectbox("Select variable for program impact analysis", selected_vars)

                fig, slope_per_week, daily = program_impact_analysis(
                    df, var, date_col, session_col, athlete_filter
                )

                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                    col1, col2, col3 = st.columns(3)

                    total_change = daily[var].iloc[-1] - daily[var].iloc[0]
                    percent_change = daily['Percent_Change'].iloc[-1]

                    col1.metric("Weekly Improvement Rate", f"{slope_per_week:.3f}")
                    col2.metric("Total Change", f"{total_change:.2f}")
                    col3.metric("Percent Change", f"{percent_change:.2f}%")

                    # Mixed model (only when viewing all athletes)
                    if athlete_filter == 'All Athletes' and 'Full Name' in df.columns:
                        st.markdown("### Group-Level Program Effect (Mixed Model)")

                        result = mixed_effects_program_model(
                            df, var, date_col, session_col
                        )

                        if result:
                            st.text(result.summary())
                        else:
                            st.info("Not enough data for mixed effects model.")
                else:
                    st.warning("Not enough data to compute program impact.")

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
            if 'Full Name' in df.columns:
                export_cols.append('Full Name')
            
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
        1. **Configure file paths** in the sidebar
        2. **Configure filters** (sample size, date range)
        3. **Click "Load/Reload Data"** to load the dataset
        4. **Select variables** you want to analyze
        5. **Explore** the interactive visualizations and statistics
        6. **Export** your results
        
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