import pandas as pd
import seaborn as sns

# Load the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\dickh\Downloads\WellmanSports\WellmanWorkoutTracker1-28-2026.csv')
# Display the first few rows of the DataFrame
pd.set_option('display.max_columns', None)

print(df.head())
def FullDataInfo():
    summary = (
        df
        .groupby(["First Name", "Last Name"])
        .size()
        .reset_index(name="total_entries")
        .sort_values(["First Name", "Last Name"])
    )

    print("------------------- Overall Summary ------------------")
    print(summary.describe())


    seaborn_plot = sns.histplot(summary['total_entries'], bins=100)
    seaborn_plot.figure.savefig(r'C:\Users\dickh\Downloads\WellmanSports\WellmanWorkoutTracker1-28-2026_histogram.png')

def ReadinessInfo():
    readiness_summary = (
        df["Readiness"]
        .value_counts()
        .sort_index()
        .reset_index(name="count")
        .rename(columns={"index": "Readiness"})
    )

    print(readiness_summary)

    # Create the barplot with filtered data
    import matplotlib.ticker as mticker

    ax = sns.barplot(
        data=readiness_summary,
        x="Readiness",
        y="count",
        errorbar=None
    )

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    seaborn_plot = ax
    seaborn_plot.figure.savefig(r'C:\Users\dickh\Downloads\WellmanSports\WellmanWorkoutTracker1-28-2026_readiness_lineplot.png')


df["Date Completed"] = pd.to_datetime(df["Date Completed"])

last_month = df["Date Completed"].max() - pd.DateOffset(months=1)

active_people = (
    df[df["Date Completed"] >= last_month]
    [["First Name", "Last Name"]]
    .drop_duplicates()
)

summary = (
    df.merge(active_people, on=["First Name", "Last Name"], how="inner")
    .groupby(["First Name", "Last Name"])
    .size()
    .reset_index(name="total_entries")
    .sort_values(["First Name", "Last Name"])
)

onlyLargeNames = summary[summary['total_entries'] >= 100]

print("------------------- Last 30 Days Summary ------------------") 
print(summary.describe())
print("------------------- Only Large Names (100+ entries) ------------------")
print(onlyLargeNames.describe())


# create a consistency score column based on block consistency and frequency of entries

df["Blocks Completed"] = pd.to_numeric(df["Blocks Completed"], errors="coerce")
df["Blocks Prescribed"] = pd.to_numeric(df["Blocks Prescribed"], errors="coerce")
person_summary = (
    df
    .groupby(["First Name", "Last Name"])
    .agg(
        total_entries=("Date Completed", "count"),
        total_blocks_prescribed=("Blocks Prescribed", "sum"),
        total_blocks_completed=("Blocks Completed", "sum")
    )
    .reset_index()
)
DAYS_WINDOW = 30  # adjust if needed

person_summary["time_consistency_score"] = (
    person_summary["total_entries"] / DAYS_WINDOW
)

person_summary["block_consistency_score"] = (
    person_summary["total_blocks_completed"] /
    person_summary["total_blocks_prescribed"]
).fillna(0)
TIME_WEIGHT = 0.5
BLOCK_WEIGHT = 0.5

person_summary["overall_consistency_score"] = (
    TIME_WEIGHT * person_summary["time_consistency_score"] +
    BLOCK_WEIGHT * person_summary["block_consistency_score"]
)
print("------------------- Consistency Breakdown ------------------")
print(
    person_summary[[
        "First Name",
        "Last Name",
        "total_entries",
        "total_blocks_prescribed",
        "total_blocks_completed",
        "time_consistency_score",
        "block_consistency_score",
        "overall_consistency_score"
    ]]
)

print("------------------- Top 10 Most Consistent Individuals ------------------")
print(
    person_summary
    .sort_values("overall_consistency_score", ascending=False)
    .head(10)
)


# checking monthly

df["Date Completed"] = pd.to_datetime(df["Date Completed"])
df["Blocks Completed"] = pd.to_numeric(df["Blocks Completed"], errors="coerce")
df["Blocks Prescribed"] = pd.to_numeric(df["Blocks Prescribed"], errors="coerce")

df["month"] = df["Date Completed"].dt.to_period("M")
monthly = (
    df
    .groupby(["First Name", "Last Name", "month"], as_index=False)
    .agg(
        total_entries=("Date Completed", "count"),
        total_blocks_prescribed=("Blocks Prescribed", "sum"),
        total_blocks_completed=("Blocks Completed", "sum")
    )
)
monthly["days_in_month"] = monthly["month"].dt.days_in_month

monthly["time_consistency_score"] = (
    monthly["total_entries"] / monthly["days_in_month"]
).clip(upper=1)

monthly["block_consistency_score"] = (
    monthly["total_blocks_completed"] /
    monthly["total_blocks_prescribed"]
).fillna(0).clip(upper=1)

monthly["overall_consistency_score"] = (
    0.5 * monthly["time_consistency_score"] +
    0.5 * monthly["block_consistency_score"]
)
latest_month = monthly["month"].max()

print(
    monthly[monthly["month"] == latest_month]
    .sort_values("overall_consistency_score", ascending=False)
    .head(10)
)
monthly[
    (monthly["First Name"] == "Caleb") &
    (monthly["Last Name"] == "Auyeung")
].sort_values("month")

