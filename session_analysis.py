import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def analyze_session_impact(df):
    """
    Comprehensive analysis of how Session Title affects performance variables
    
    Parameters:
    df : pandas DataFrame containing jump test data with 'Session Title' column
    """
    
    print("="*80)
    print("SESSION TITLE IMPACT ANALYSIS")
    print("="*80)
    
    # 1. DATA OVERVIEW
    print("\n1. DATA OVERVIEW")
    print("-" * 80)
    print(f"Total observations: {len(df)}")
    print(f"Unique Session Titles: {df['Session Title'].nunique()}")
    print(f"\nSession Title Distribution:")
    print(df['Session Title'].value_counts())
    
    # Check for other grouping variables
    if 'Position' in df.columns:
        print(f"\nUnique Positions: {df['Position'].nunique()}")
    if 'First Name' in df.columns and 'Last Name' in df.columns:
        df['Athlete'] = df['First Name'] + ' ' + df['Last Name']
        print(f"Unique Athletes: {df['Athlete'].nunique()}")
    
    # 2. IDENTIFY KEY PERFORMANCE VARIABLES
    performance_vars = [
        'Jump Height',
        'Peak Propulsive Power',
        'Avg. Propulsive Power',
        'Peak Propulsive Force',
        'Avg. Propulsive Force',
        'RSI',
        'Takeoff Velocity',
        'Peak Velocity',
        'Countermovement Depth',
        'Braking RFD',
        'Time To Takeoff',
        'Flight Time',
        'Peak Braking Power',
        'Impulse Ratio',
        'Positive Net Impulse',
        'Propulsive Net Impulse',
        'Braking Net Impulse'
    ]
    
    # Filter to variables that exist in the dataframe
    available_vars = [var for var in performance_vars if var in df.columns]
    
    print(f"\n2. KEY PERFORMANCE VARIABLES AVAILABLE ({len(available_vars)}):")
    print("-" * 80)
    for var in available_vars:
        print(f"  - {var}")
    
    # 3. DESCRIPTIVE STATISTICS BY SESSION
    print("\n3. DESCRIPTIVE STATISTICS BY SESSION TITLE")
    print("-" * 80)
    
    summary_stats = {}
    for var in available_vars:
        print(f"\n{var}:")
        session_summary = df.groupby('Session Title')[var].agg([
            ('n', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('median', 'median')
        ]).round(2)
        print(session_summary)
        summary_stats[var] = session_summary
    
    # 4. STATISTICAL TESTS
    print("\n4. STATISTICAL SIGNIFICANCE TESTS")
    print("-" * 80)
    print("Testing if session titles have significantly different means (ANOVA/Kruskal-Wallis)")
    print()
    
    stat_results = {}
    sessions = df['Session Title'].unique()
    
    if len(sessions) > 1:
        for var in available_vars:
            # Group data by session
            groups = [df[df['Session Title'] == session][var].dropna() for session in sessions]
            
            # Only test if we have data in multiple groups
            valid_groups = [g for g in groups if len(g) > 0]
            
            if len(valid_groups) > 1:
                # Check normality (Shapiro-Wilk test on each group)
                normal = all(stats.shapiro(g)[1] > 0.05 if len(g) >= 3 else True for g in valid_groups)
                
                if normal and all(len(g) >= 3 for g in valid_groups):
                    # ANOVA for normally distributed data
                    stat, p_value = f_oneway(*valid_groups)
                    test_used = "ANOVA"
                else:
                    # Kruskal-Wallis for non-normal data
                    stat, p_value = kruskal(*valid_groups)
                    test_used = "Kruskal-Wallis"
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                stat_results[var] = {
                    'test': test_used,
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': significance
                }
                
                print(f"{var:40s} | {test_used:15s} | p={p_value:.4f} {significance}")
    else:
        print("Only one session title present - statistical tests not applicable")
    
    # 5. EFFECT SIZE ANALYSIS
    if len(sessions) > 1:
        print("\n5. EFFECT SIZE ANALYSIS (Cohen's d for pairwise comparisons)")
        print("-" * 80)
        
        def cohens_d(group1, group2):
            """Calculate Cohen's d for effect size"""
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        # Calculate effect sizes for top 5 most significant variables
        if stat_results:
            top_vars = sorted(stat_results.items(), key=lambda x: x[1]['p_value'])[:5]
            
            for var, results in top_vars:
                if results['p_value'] < 0.05:
                    print(f"\n{var}:")
                    sessions_list = list(sessions)
                    for i in range(len(sessions_list)):
                        for j in range(i+1, len(sessions_list)):
                            g1 = df[df['Session Title'] == sessions_list[i]][var].dropna()
                            g2 = df[df['Session Title'] == sessions_list[j]][var].dropna()
                            if len(g1) > 0 and len(g2) > 0:
                                d = cohens_d(g1, g2)
                                effect = "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
                                print(f"  {sessions_list[i]} vs {sessions_list[j]}: d={d:.3f} ({effect})")
    
    # 6. CORRELATION ANALYSIS
    print("\n6. CORRELATION BETWEEN KEY VARIABLES")
    print("-" * 80)
    
    # Select numeric columns for correlation
    numeric_cols = df[available_vars].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        # Print strongest correlations
        print("\nTop 10 Strongest Correlations:")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
        
        sorted_pairs = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
        for var1, var2, corr in sorted_pairs[:10]:
            print(f"  {var1:30s} <-> {var2:30s} : r={corr:.3f}")
    
    # 7. COEFFICIENT OF VARIATION (CV) ANALYSIS
    print("\n7. VARIABILITY ANALYSIS (Coefficient of Variation)")
    print("-" * 80)
    print("CV% by Session Title for key variables:")
    
    cv_results = {}
    for var in available_vars[:5]:  # Top 5 variables
        print(f"\n{var}:")
        for session in sessions:
            session_data = df[df['Session Title'] == session][var].dropna()
            if len(session_data) > 0 and session_data.mean() != 0:
                cv = (session_data.std() / session_data.mean()) * 100
                cv_results[f"{session}_{var}"] = cv
                print(f"  {session}: CV = {cv:.2f}%")
    
    return {
        'summary_stats': summary_stats,
        'statistical_tests': stat_results,
        'available_vars': available_vars,
        'sessions': sessions,
        'cv_results': cv_results
    }


def create_visualizations(df, results):
    """
    Create comprehensive visualizations of session impact
    
    Parameters:
    df : pandas DataFrame
    results : dict from analyze_session_impact function
    """
    
    available_vars = results['available_vars']
    sessions = results['sessions']
    
    # Select top variables for visualization (most significant or most important)
    viz_vars = available_vars[:6] if len(available_vars) >= 6 else available_vars
    
    # 1. BOX PLOTS - Distribution comparison
    print("\nCreating box plot comparisons...")
    n_vars = len(viz_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    for idx, var in enumerate(viz_vars):
        ax = axes[idx]
        df_clean = df[[var, 'Session Title']].dropna()
        
        if len(df_clean) > 0:
            sns.boxplot(data=df_clean, x='Session Title', y=var, ax=ax, palette='Set2')
            ax.set_title(f'{var} by Session Title', fontsize=12, fontweight='bold')
            ax.set_xlabel('Session Title', fontsize=10)
            ax.set_ylabel(var, fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            
            # Add sample size annotations
            for i, session in enumerate(sessions):
                n = len(df[df['Session Title'] == session][var].dropna())
                ax.text(i, ax.get_ylim()[0], f'n={n}', ha='center', va='top', fontsize=8)
    
    # Hide extra subplots
    for idx in range(len(viz_vars), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\dickh\Downloads\WellmanSports\session_boxplots.png', dpi=300, bbox_inches='tight')
    print("Saved: session_boxplots.png")
    plt.close()
    
    # 2. VIOLIN PLOTS - Distribution shape
    print("Creating violin plot comparisons...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    for idx, var in enumerate(viz_vars):
        ax = axes[idx]
        df_clean = df[[var, 'Session Title']].dropna()
        
        if len(df_clean) > 0 and len(sessions) > 1:
            sns.violinplot(data=df_clean, x='Session Title', y=var, ax=ax, palette='Set3')
            ax.set_title(f'{var} Distribution by Session', fontsize=12, fontweight='bold')
            ax.set_xlabel('Session Title', fontsize=10)
            ax.set_ylabel(var, fontsize=10)
            ax.tick_params(axis='x', rotation=45)
    
    for idx in range(len(viz_vars), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\dickh\Downloads\WellmanSports\session_violinplots.png', dpi=300, bbox_inches='tight')
    print("Saved: session_violinplots.png")
    plt.close()
    
    # 3. MEAN COMPARISON PLOTS with error bars
    print("Creating mean comparison plots...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    for idx, var in enumerate(viz_vars):
        ax = axes[idx]
        
        means = []
        stds = []
        session_labels = []
        
        for session in sessions:
            session_data = df[df['Session Title'] == session][var].dropna()
            if len(session_data) > 0:
                means.append(session_data.mean())
                stds.append(session_data.std())
                session_labels.append(session)
        
        if len(means) > 0:
            x_pos = np.arange(len(session_labels))
            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(session_labels, rotation=45, ha='right')
            ax.set_title(f'{var} - Mean ± SD', fontsize=12, fontweight='bold')
            ax.set_ylabel(var, fontsize=10)
            ax.grid(axis='y', alpha=0.3)
    
    for idx in range(len(viz_vars), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\dickh\Downloads\WellmanSports\session_means.png', dpi=300, bbox_inches='tight')
    print("Saved: session_means.png")
    plt.close()
    
    # 4. CORRELATION HEATMAP
    if len(viz_vars) > 1:
        print("Creating correlation heatmap...")
        plt.figure(figsize=(12, 10))
        
        numeric_data = df[viz_vars].select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                    cmap='coolwarm', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix - Key Performance Variables', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(r'C:\Users\dickh\Downloads\WellmanSports\correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Saved: correlation_heatmap.png")
        plt.close()
    
    # 5. RADAR CHART comparing sessions (if multiple sessions exist)
    if len(sessions) > 1 and len(viz_vars) >= 3:
        print("Creating radar chart comparison...")
        
        # Normalize data for radar chart (0-100 scale)
        radar_data = {}
        for session in sessions:
            radar_data[session] = []
            for var in viz_vars[:6]:  # Limit to 6 variables for readability
                session_values = df[df['Session Title'] == session][var].dropna()
                if len(session_values) > 0:
                    # Normalize to 0-100 based on global min/max
                    global_min = df[var].min()
                    global_max = df[var].max()
                    mean_val = session_values.mean()
                    normalized = ((mean_val - global_min) / (global_max - global_min)) * 100 if global_max != global_min else 50
                    radar_data[session].append(normalized)
                else:
                    radar_data[session].append(0)
        
        # Create radar chart
        categories = viz_vars[:6]
        n_cats = len(categories)
        
        angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(sessions)))
        
        for idx, (session, values) in enumerate(radar_data.items()):
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=session, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 100)
        ax.set_title('Performance Profile by Session Title\n(Normalized 0-100)', 
                     size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(r'C:\Users\dickh\Downloads\WellmanSports\session_radar.png', dpi=300, bbox_inches='tight')
        print("Saved: session_radar.png")
        plt.close()
    
    # 6. TIME SERIES if date information available
    if 'Date Completed' in df.columns:
        print("Creating time series plot...")
        
        try:
            df['Date_parsed'] = pd.to_datetime(df['Date Completed'])
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.flatten()
            
            for idx, var in enumerate(viz_vars[:4]):
                ax = axes[idx]
                
                for session in sessions:
                    session_df = df[df['Session Title'] == session].sort_values('Date_parsed')
                    if len(session_df) > 0:
                        ax.plot(session_df['Date_parsed'], session_df[var], 
                               marker='o', label=session, linewidth=2)
                
                ax.set_title(f'{var} Over Time', fontsize=12, fontweight='bold')
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel(var, fontsize=10)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(r'C:\Users\dickh\Downloads\WellmanSports\session_timeseries.png', dpi=300, bbox_inches='tight')
            print("Saved: session_timeseries.png")
            plt.close()
        except:
            print("Could not parse dates for time series")
    
    print("\nAll visualizations created successfully!")


def export_summary_report(df, results, filename='session_analysis_report.csv'):
    """Export summary statistics to CSV"""
    
    summary_data = []
    
    for var, stats_df in results['summary_stats'].items():
        for session in stats_df.index:
            row = {
                'Variable': var,
                'Session_Title': session,
                'N': stats_df.loc[session, 'n'],
                'Mean': stats_df.loc[session, 'mean'],
                'Std': stats_df.loc[session, 'std'],
                'Min': stats_df.loc[session, 'min'],
                'Max': stats_df.loc[session, 'max'],
                'Median': stats_df.loc[session, 'median']
            }
            
            # Add p-value if available
            if var in results['statistical_tests']:
                row['P_Value'] = results['statistical_tests'][var]['p_value']
                row['Test_Type'] = results['statistical_tests'][var]['test']
                row['Significant'] = results['statistical_tests'][var]['significant']
            
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(rf'C:\Users\dickh\Downloads\WellmanSports\{filename}', index=False)
    print(f"\nSummary report exported to: {filename}")
    
    return summary_df


# MAIN EXECUTION EXAMPLE
if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("SESSION TITLE ANALYSIS - EXAMPLE USAGE")
    print("="*80)
    print("\nThis script assumes your data is in a pandas DataFrame named 'df'")
    print("\nTo use this script:")
    print("  1. Load your data: df = pd.read_csv('your_data.csv')")
    print("  2. Run analysis: results = analyze_session_impact(df)")
    print("  3. Create visualizations: create_visualizations(df, results)")
    print("  4. Export report: export_summary_report(df, results)")
    print("\n" + "="*80)
    
    # Example with dummy data
    print("\nRunning example with simulated data...")
    
    np.random.seed(42)
    n_observations = 100
    
    example_df = pd.DataFrame({
        'Session Title': np.random.choice([
            'Load/Explode Day 1',
            'Speed Development',
            'Recovery Session',
            'Max Strength Day'
        ], n_observations),
        'Jump Height': np.random.normal(25, 5, n_observations),
        'Peak Propulsive Power': np.random.normal(7000, 1000, n_observations),
        'RSI': np.random.normal(1.2, 0.3, n_observations),
        'Takeoff Velocity': np.random.normal(3.5, 0.5, n_observations),
        'Peak Propulsive Force': np.random.normal(2500, 400, n_observations),
        'Countermovement Depth': np.random.normal(-18, 3, n_observations),
        'Position': np.random.choice(['Catcher', 'Pitcher', 'Infield', 'Outfield'], n_observations),
        'First Name': ['John'] * n_observations,
        'Last Name': ['Doe'] * n_observations,
        'Date Completed': pd.date_range('2025-01-01', periods=n_observations, freq='D')
    })
    
    # Add some variance based on session type
    for idx, row in example_df.iterrows():
        if row['Session Title'] == 'Speed Development':
            example_df.loc[idx, 'Jump Height'] *= 1.1
            example_df.loc[idx, 'RSI'] *= 1.15
        elif row['Session Title'] == 'Recovery Session':
            example_df.loc[idx, 'Jump Height'] *= 0.9
            example_df.loc[idx, 'Peak Propulsive Power'] *= 0.85
    
    results = analyze_session_impact(example_df)
    create_visualizations(example_df, results)
    export_summary_report(example_df, results)
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETE - Check output files!")
    print("="*80)
