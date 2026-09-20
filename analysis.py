# import python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Libraries imported successfully!")

# Load dataset
df = pd.read_csv("seasonal_agriculture_performance_dataset (2).csv")

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Check dataset shape
print("\nDataset Shape:")
print(df.shape)

# Check column names
print("\nColumn Names:")
print(df.columns.tolist())

# Check data types
print("\nData Types:")
print(df.dtypes)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Basic statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Find columns having missing values
missing = df.isnull().sum()

print("\nColumns with Missing Values:")
print(missing[missing > 0])

# Fill missing numerical values with median
df["Rainfall_mm"] = df["Rainfall_mm"].fillna(df["Rainfall_mm"].median())
df["Soil_Moisture_pct"] = df["Soil_Moisture_pct"].fillna(
    df["Soil_Moisture_pct"].median()
)
df["Yield_Tonnes_Ha"] = df["Yield_Tonnes_Ha"].fillna(
    df["Yield_Tonnes_Ha"].median()
)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

df.to_csv("cleaned_agriculture_data.csv", index=False)

print("\nCleaned dataset saved successfully!")

# -----------------------------------
# STEP 6: Exploratory Data Analysis
# -----------------------------------

# Count records in each season
print("\nRecords by Season:")
print(df["Season"].value_counts())

# Average yield by season
print("\nAverage Yield by Season:")
print(df.groupby("Season")["Yield_Tonnes_Ha"].mean())

# Average profit by season
print("\nAverage Profit by Season:")
print(df.groupby("Season")["Profit_INR"].mean())

# Total production by season
print("\nTotal Production by Season:")
print(df.groupby("Season")["Production_Tonnes"].sum())


# -----------------------------------
# STEP 7: Season vs Average Yield
# -----------------------------------

season_yield = df.groupby("Season")["Yield_Tonnes_Ha"].mean()

plt.figure(figsize=(8, 5))

season_yield.plot(
    kind="bar",
    color="#F2B6C1",          # soft rose
    edgecolor="#333333",
    linewidth=0.8,
    width=0.6
)

plt.title("Average Agricultural Yield by Season")
plt.xlabel("Season")
plt.ylabel("Average Yield (Tonnes/Ha)")

plt.xticks(rotation=0)

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.5
)

plt.tight_layout()

# Save graph
plt.savefig("season_vs_yield.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 8: Season vs Average Profit
# -----------------------------------

season_profit = df.groupby("Season")["Profit_INR"].mean()

plt.figure(figsize=(8, 5))

season_profit.plot(
    kind="bar",
    color="#A8D5A2",          # soft sage green
    edgecolor="#333333",
    linewidth=0.8,
    width=0.6
)

plt.title("Average Agricultural Profit by Season")
plt.xlabel("Season")
plt.ylabel("Average Profit (INR)")

plt.xticks(rotation=0)

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("season_vs_profit.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 9: Season vs Total Production
# -----------------------------------

season_production = df.groupby("Season")["Production_Tonnes"].sum()

plt.figure(figsize=(8, 5))

season_production.plot(
    kind="bar",
    color="#8FCFC4",          # soft teal
    edgecolor="#333333",
    linewidth=0.8,
    width=0.6
)

plt.title("Total Agricultural Production by Season")
plt.xlabel("Season")
plt.ylabel("Total Production (Tonnes)")

plt.xticks(rotation=0)

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("season_vs_production.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 10: Rainfall vs Yield
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Rainfall_mm"],
    df["Yield_Tonnes_Ha"],
    color="#9EC8E3",          # soft blue
    alpha=0.75,
    edgecolors="#333333",
    linewidths=0.5
)

plt.title("Rainfall vs Agricultural Yield")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Yield (Tonnes/Ha)")

# Grid
plt.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("rainfall_vs_yield.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 11: Soil Moisture vs Yield
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Soil_Moisture_pct"],
    df["Yield_Tonnes_Ha"],
    color="#F5C6A5",          # soft peach
    alpha=0.75,
    edgecolors="#333333",
    linewidths=0.5
)

plt.title("Soil Moisture vs Agricultural Yield")
plt.xlabel("Soil Moisture (%)")
plt.ylabel("Yield (Tonnes/Ha)")

# Grid
plt.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("soil_moisture_vs_yield.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 12: Season vs Average Water Usage
# -----------------------------------

season_water = df.groupby("Season")["Water_Used_m3"].mean()

plt.figure(figsize=(8, 5))

season_water.plot(
    kind="bar",
    color="#C3B1E1",          # soft lavender
    edgecolor="#333333",
    linewidth=0.8,
    width=0.6
)

plt.title("Average Water Usage by Season")
plt.xlabel("Season")
plt.ylabel("Average Water Usage (m³)")

plt.xticks(rotation=0)

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("season_vs_water_usage.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 13: Season vs Water Efficiency
# -----------------------------------

season_efficiency = df.groupby("Season")[
    "Water_Efficiency_t_per_1000m3"
].mean()

plt.figure(figsize=(8, 5))

season_efficiency.plot(
    kind="bar",
    color="lightyellow",
    edgecolor="black",
    linewidth=0.5,
    width=0.6

)

plt.title("Average Water Efficiency by Season")
plt.xlabel("Season")
plt.ylabel("Water Efficiency (Tonnes / 1000 m³)")

plt.xticks(rotation=0)

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("season_vs_water_efficiency.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 14: Season vs Disease & Pest Risk
# -----------------------------------

season_risk = df.groupby("Season")[
    "Disease_Pest_Risk_pct"
].mean()

plt.figure(figsize=(8, 5))

season_risk.plot(
    kind="bar",
    color="crimson",
    edgecolor="black",
    linewidth=0.5,
    width=0.6
)

plt.title("Average Disease and Pest Risk by Season")
plt.xlabel("Season")
plt.ylabel("Disease & Pest Risk (%)")

plt.xticks(rotation=0)

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("season_vs_disease_pest_risk.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 15: Crop vs Average Yield
# -----------------------------------

crop_yield = df.groupby("Crop")["Yield_Tonnes_Ha"].mean().sort_values(
    ascending=False
)

plt.figure(figsize=(10, 6))

crop_yield.plot(
    kind="bar",
    color="teal",
    edgecolor="black",
    linewidth=0.5,
    width=0.6
)

plt.title("Average Agricultural Yield by Crop")
plt.xlabel("Crop")
plt.ylabel("Average Yield (Tonnes/Ha)")

plt.xticks(rotation=45, ha="right")

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("crop_vs_yield.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 16: Crop vs Average Profit
# -----------------------------------

crop_profit = df.groupby("Crop")["Profit_INR"].mean().sort_values(
    ascending=False
)

plt.figure(figsize=(10, 6))

crop_profit.plot(
    kind="bar",
    color="slateblue",
    edgecolor="black",
    linewidth=0.5,
    width=0.6
)

plt.title("Average Agricultural Profit by Crop")
plt.xlabel("Crop")
plt.ylabel("Average Profit (INR)")

plt.xticks(rotation=45, ha="right")

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("crop_vs_profit.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 17: Crop vs Total Production
# -----------------------------------

crop_production = df.groupby("Crop")["Production_Tonnes"].sum().sort_values(
    ascending=False
)

plt.figure(figsize=(10, 6))

crop_production.plot(
    kind="bar",
    color="coral",
    edgecolor="black",
    linewidth=0.5,
    width=0.6
)

plt.title("Total Agricultural Production by Crop")
plt.xlabel("Crop")
plt.ylabel("Total Production (Tonnes)")

plt.xticks(rotation=45, ha="right")

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("crop_vs_production.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 18: Crop vs Water Efficiency
# -----------------------------------

crop_efficiency = df.groupby("Crop")[
    "Water_Efficiency_t_per_1000m3"
].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))

crop_efficiency.plot(
    kind="bar",
    color="mediumseagreen",
    edgecolor="black",
    linewidth=0.5,
    width=0.6
)

plt.title("Average Water Efficiency by Crop")
plt.xlabel("Crop")
plt.ylabel("Water Efficiency (Tonnes / 1000 m³)")

plt.xticks(rotation=45, ha="right")

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("crop_vs_water_efficiency.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 19: Crop vs Disease & Pest Risk
# -----------------------------------

crop_risk = df.groupby("Crop")[
    "Disease_Pest_Risk_pct"
].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))

crop_risk.plot(
    kind="bar",
    color="indianred",
    edgecolor="black",
    linewidth=0.5,
    width=0.6
)

plt.title("Average Disease and Pest Risk by Crop")
plt.xlabel("Crop")
plt.ylabel("Disease & Pest Risk (%)")

plt.xticks(rotation=45, ha="right")

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("crop_vs_disease_pest_risk.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 20: Season vs Crop Production
# -----------------------------------

season_crop_production = df.pivot_table(
    values="Production_Tonnes",
    index="Crop",
    columns="Season",
    aggfunc="mean"
)

season_crop_production.plot(
    kind="bar",
    figsize=(11, 6),
    edgecolor="black",
    linewidth=0.5,
    width=0.6
)

plt.title("Average Crop Production Across Seasons")
plt.xlabel("Crop")
plt.ylabel("Average Production (Tonnes)")

plt.xticks(rotation=45, ha="right")

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.legend(title="Season")

plt.tight_layout()

# Save graph
plt.savefig("season_crop_production.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 21: Rainfall vs Soil Moisture
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Rainfall_mm"],
    df["Soil_Moisture_pct"],
    alpha=0.6,
    edgecolors="black",
    linewidths=0.5
)

plt.title("Rainfall vs Soil Moisture")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Soil Moisture (%)")

# Grid
plt.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("rainfall_vs_soil_moisture.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 22: Rainfall vs Water Usage
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Rainfall_mm"],
    df["Water_Used_m3"],
    alpha=0.6,
    edgecolors="black",
    linewidths=0.5
)

plt.title("Rainfall vs Water Usage")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Water Used (m³)")

# Grid
plt.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("rainfall_vs_water_usage.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 23: Crop vs Average Water Usage
# -----------------------------------

crop_water = df.groupby("Crop")["Water_Used_m3"].mean().sort_values(
    ascending=False
)

plt.figure(figsize=(10, 6))

crop_water.plot(
    kind="bar",
    color="royalblue",
    edgecolor="black",
    linewidth=0.5,
    width=0.6
)

plt.title("Average Water Usage by Crop")
plt.xlabel("Crop")
plt.ylabel("Average Water Usage (m³)")

plt.xticks(rotation=45, ha="right")

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("crop_vs_water_usage.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 24: Crop vs Average Soil Moisture
# -----------------------------------

crop_soil_moisture = df.groupby("Crop")[
    "Soil_Moisture_pct"
].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))

crop_soil_moisture.plot(
    kind="bar",
    color="darkcyan",
    edgecolor="black",
    linewidth=0.5,
    width=0.6
)

plt.title("Average Soil Moisture by Crop")
plt.xlabel("Crop")
plt.ylabel("Average Soil Moisture (%)")

plt.xticks(rotation=45, ha="right")

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("crop_vs_soil_moisture.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 25: Season vs Average Soil Moisture
# -----------------------------------

season_soil_moisture = df.groupby("Season")[
    "Soil_Moisture_pct"
].mean()

plt.figure(figsize=(8, 5))

season_soil_moisture.plot(
    kind="bar",
    color="darkgoldenrod",
    edgecolor="black",
    linewidth=0.5,
    width=0.6
)

plt.title("Average Soil Moisture by Season")
plt.xlabel("Season")
plt.ylabel("Average Soil Moisture (%)")

plt.xticks(rotation=0)

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("season_vs_soil_moisture.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 26: Rainfall vs Production
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Rainfall_mm"],
    df["Production_Tonnes"],
    alpha=0.6,
    edgecolors="black",
    linewidths=0.5
)

plt.title("Rainfall vs Agricultural Production")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Production (Tonnes)")

# Grid
plt.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("rainfall_vs_production.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 27: Soil Moisture vs Water Usage
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Soil_Moisture_pct"],
    df["Water_Used_m3"],
    alpha=0.6,
    edgecolors="black",
    linewidths=0.5
)

plt.title("Soil Moisture vs Water Usage")
plt.xlabel("Soil Moisture (%)")
plt.ylabel("Water Used (m³)")

# Grid
plt.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("soil_moisture_vs_water_usage.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 28: Yield vs Profit
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Yield_Tonnes_Ha"],
    df["Profit_INR"],
    alpha=0.6,
    edgecolors="black",
    linewidths=0.5
)

plt.title("Yield vs Agricultural Profit")
plt.xlabel("Yield (Tonnes/Ha)")
plt.ylabel("Profit (INR)")

# Grid
plt.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

# Save graph
plt.savefig("yield_vs_profit.png", dpi=300)

plt.show()

# -----------------------------------
# STEP 29: Correlation Analysis
# -----------------------------------

correlation = df[
    [
        "Rainfall_mm",
        "Soil_Moisture_pct",
        "Yield_Tonnes_Ha",
        "Production_Tonnes",
        "Water_Used_m3",
        "Water_Efficiency_t_per_1000m3",
        "Disease_Pest_Risk_pct",
        "Profit_INR"
    ]
].corr()

print("\nCorrelation Matrix:")
print(correlation.round(2))

# Save correlation matrix
correlation.to_csv("correlation_matrix.csv")

print("\nCorrelation matrix saved successfully!")


# -----------------------------------
# STEP 30: Season-wise Summary
# -----------------------------------

season_summary = df.groupby("Season").agg({
    "Yield_Tonnes_Ha": "mean",
    "Profit_INR": "mean",
    "Production_Tonnes": "sum",
    "Water_Used_m3": "mean",
    "Water_Efficiency_t_per_1000m3": "mean",
    "Disease_Pest_Risk_pct": "mean"
}).round(2)

print("\nSeason-wise Summary:")
print(season_summary)

# Save summary table
season_summary.to_csv("season_wise_summary.csv")

print("\nSeason-wise summary saved successfully!")
# -----------------------------------
# STEP 31: Crop-wise Summary
# -----------------------------------

crop_summary = df.groupby("Crop").agg({
    "Yield_Tonnes_Ha": "mean",
    "Profit_INR": "mean",
    "Production_Tonnes": "sum",
    "Water_Used_m3": "mean",
    "Water_Efficiency_t_per_1000m3": "mean",
    "Disease_Pest_Risk_pct": "mean"
}).round(2)

print("\nCrop-wise Summary:")
print(crop_summary)

# Save summary table
crop_summary.to_csv("crop_wise_summary.csv")

print("\nCrop-wise summary saved successfully!")

# -----------------------------------
# STEP 32: Overall Crop Performance
# -----------------------------------

performance = df.groupby("Crop").agg({
    "Yield_Tonnes_Ha": "mean",
    "Profit_INR": "mean",
    "Water_Efficiency_t_per_1000m3": "mean",
    "Disease_Pest_Risk_pct": "mean"
})

# Normalize important metrics
performance["Yield_Score"] = (
    performance["Yield_Tonnes_Ha"] /
    performance["Yield_Tonnes_Ha"].max()
)

performance["Profit_Score"] = (
    performance["Profit_INR"] /
    performance["Profit_INR"].max()
)

performance["Water_Efficiency_Score"] = (
    performance["Water_Efficiency_t_per_1000m3"] /
    performance["Water_Efficiency_t_per_1000m3"].max()
)

# Lower disease/pest risk is better
performance["Risk_Score"] = (
    1 - (
        performance["Disease_Pest_Risk_pct"] /
        performance["Disease_Pest_Risk_pct"].max()
    )
)

# Overall score
performance["Overall_Score"] = (
    performance["Yield_Score"] * 0.30 +
    performance["Profit_Score"] * 0.30 +
    performance["Water_Efficiency_Score"] * 0.20 +
    performance["Risk_Score"] * 0.20
)

performance = performance.sort_values(
    "Overall_Score",
    ascending=False
).round(3)

print("\nOverall Crop Performance:")
print(performance)

# Save result
performance.to_csv("overall_crop_performance.csv")

print("\nOverall performance saved successfully!")

# -----------------------------------
# STEP 33: Final Statistical Summary
# -----------------------------------

final_summary = df[
    [
        "Rainfall_mm",
        "Soil_Moisture_pct",
        "Yield_Tonnes_Ha",
        "Production_Tonnes",
        "Water_Used_m3",
        "Water_Efficiency_t_per_1000m3",
        "Disease_Pest_Risk_pct",
        "Profit_INR"
    ]
].describe().round(2)

print("\nFinal Statistical Summary:")
print(final_summary)

# Save final statistical summary
final_summary.to_csv("final_statistical_summary.csv")

print("\nFinal statistical summary saved successfully!")

# -----------------------------------
# STEP 34: Important Insights
# -----------------------------------

# Season-wise insights
season_insights = df.groupby("Season").agg({
    "Yield_Tonnes_Ha": "mean",
    "Profit_INR": "mean",
    "Water_Efficiency_t_per_1000m3": "mean",
    "Disease_Pest_Risk_pct": "mean"
})

print("\n========== SEASON INSIGHTS ==========")

print("\nHighest Average Yield:")
print(season_insights["Yield_Tonnes_Ha"].idxmax())

print("\nHighest Average Profit:")
print(season_insights["Profit_INR"].idxmax())

print("\nHighest Water Efficiency:")
print(season_insights[
    "Water_Efficiency_t_per_1000m3"
].idxmax())

print("\nHighest Disease & Pest Risk:")
print(season_insights[
    "Disease_Pest_Risk_pct"
].idxmax())


# Crop-wise insights
crop_insights = df.groupby("Crop").agg({
    "Yield_Tonnes_Ha": "mean",
    "Profit_INR": "mean",
    "Water_Efficiency_t_per_1000m3": "mean",
    "Disease_Pest_Risk_pct": "mean"
})

print("\n========== CROP INSIGHTS ==========")

print("\nHighest Average Yield Crop:")
print(crop_insights["Yield_Tonnes_Ha"].idxmax())

print("\nHighest Average Profit Crop:")
print(crop_insights["Profit_INR"].idxmax())

print("\nHighest Water Efficiency Crop:")
print(crop_insights[
    "Water_Efficiency_t_per_1000m3"
].idxmax())

print("\nHighest Disease & Pest Risk Crop:")
print(crop_insights[
    "Disease_Pest_Risk_pct"
].idxmax())

# -----------------------------------
# STEP 35: Final Recommendations
# -----------------------------------

print("\n========== FINAL RECOMMENDATIONS ==========")

# Season with highest yield
best_yield_season = season_insights["Yield_Tonnes_Ha"].idxmax()

# Season with highest water efficiency
best_water_season = season_insights[
    "Water_Efficiency_t_per_1000m3"
].idxmax()

# Season with highest profit
best_profit_season = season_insights["Profit_INR"].idxmax()

# Season with highest risk
high_risk_season = season_insights[
    "Disease_Pest_Risk_pct"
].idxmax()

print(
    f"1. {best_yield_season} shows the highest average agricultural yield."
)

print(
    f"2. {best_profit_season} shows the highest average profit."
)

print(
    f"3. {best_water_season} shows the highest water efficiency."
)

print(
    f"4. {high_risk_season} has the highest average disease and pest risk."
)

print(
    "5. Seasonal conditions should be considered when planning crop production."
)

print(
    "6. Water usage and water efficiency should be monitored for better resource management."
)

print(
    "7. Disease and pest risk should be monitored to support timely crop management."
)

# -----------------------------------
# STEP 36: Important Correlations
# -----------------------------------

# Get correlation pairs
corr_pairs = correlation.where(
    np.triu(
        np.ones(correlation.shape),
        k=1
    ).astype(bool)
).stack()

# Sort by absolute correlation value
corr_pairs = corr_pairs.reindex(
    corr_pairs.abs().sort_values(ascending=False).index
)

print("\n========== IMPORTANT CORRELATIONS ==========")

print(corr_pairs.head(10).round(3))

# Save important correlations
corr_pairs.head(10).round(3).to_csv(
    "important_correlations.csv"
)

print("\nImportant correlations saved successfully!")

# -----------------------------------
# STEP 37: Final Project Findings
# -----------------------------------

final_findings = pd.DataFrame({
    "Metric": [
        "Highest Average Yield Season",
        "Highest Average Profit Season",
        "Highest Water Efficiency Season",
        "Highest Disease & Pest Risk Season",
        "Highest Average Yield Crop",
        "Highest Average Profit Crop",
        "Highest Water Efficiency Crop",
        "Highest Disease & Pest Risk Crop"
    ],

    "Finding": [
        season_insights["Yield_Tonnes_Ha"].idxmax(),
        season_insights["Profit_INR"].idxmax(),
        season_insights[
            "Water_Efficiency_t_per_1000m3"
        ].idxmax(),
        season_insights[
            "Disease_Pest_Risk_pct"
        ].idxmax(),

        crop_insights["Yield_Tonnes_Ha"].idxmax(),
        crop_insights["Profit_INR"].idxmax(),
        crop_insights[
            "Water_Efficiency_t_per_1000m3"
        ].idxmax(),
        crop_insights[
            "Disease_Pest_Risk_pct"
        ].idxmax()
    ]
})

print("\n========== FINAL PROJECT FINDINGS ==========")
print(final_findings)

# Save findings
final_findings.to_csv(
    "final_project_findings.csv",
    index=False
)

print("\nFinal project findings saved successfully!")

# -----------------------------------
# STEP 38: Correlation Heatmap
# -----------------------------------

plt.figure(figsize=(10, 8))

plt.imshow(correlation, cmap="coolwarm")

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap of Agricultural Variables")

plt.tight_layout()

# Save graph
plt.savefig(
    "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------------
# STEP 39: Season-wise Performance Comparison
# -----------------------------------

season_performance = df.groupby("Season").agg({
    "Yield_Tonnes_Ha": "mean",
    "Profit_INR": "mean",
    "Water_Efficiency_t_per_1000m3": "mean",
    "Disease_Pest_Risk_pct": "mean"
})

# Normalize values using min-max normalization
season_normalized = (
    season_performance - season_performance.min()
) / (
    season_performance.max() - season_performance.min()
)

# Risk is better when it is lower
season_normalized["Disease_Pest_Risk_pct"] = (
    1 - season_normalized["Disease_Pest_Risk_pct"]
)

print("\n========== NORMALIZED SEASON PERFORMANCE ==========")
print(season_normalized.round(2))

# Plot
season_normalized.plot(
    kind="bar",
    figsize=(10, 6),
    edgecolor="black",
    linewidth=1.2
)

plt.title("Season-wise Agricultural Performance Comparison")
plt.xlabel("Season")
plt.ylabel("Normalized Score")

plt.xticks(rotation=0)

plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.legend(
    title="Performance Metrics"
)

plt.tight_layout()

# Save graph
plt.savefig(
    "season_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -----------------------------------
# STEP 40: Season-wise Production Distribution
# -----------------------------------

season_production = df.groupby("Season")[
    "Production_Tonnes"
].sum()

print("\n========== SEASON-WISE PRODUCTION ==========")
print(season_production.round(2))

plt.figure(figsize=(8, 5))

plt.pie(
    season_production,
    labels=season_production.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Season-wise Production Distribution")

plt.tight_layout()

# Save graph
plt.savefig(
    "season_production_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------------
# STEP 41: Crop-wise Production Distribution
# -----------------------------------

crop_production = df.groupby("Crop")[
    "Production_Tonnes"
].sum().sort_values(ascending=False)

print("\n========== CROP-WISE PRODUCTION ==========")
print(crop_production.round(2))

plt.figure(figsize=(10, 6))

plt.pie(
    crop_production,
    labels=crop_production.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Crop-wise Production Distribution")

plt.tight_layout()

# Save graph
plt.savefig(
    "crop_production_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------------
# STEP 42: Crop-wise Profit Distribution
# -----------------------------------

crop_profit = df.groupby("Crop")["Profit_INR"].mean().sort_values(
    ascending=False
)

print("\n========== CROP-WISE AVERAGE PROFIT ==========")
print(crop_profit.round(2))

plt.figure(figsize=(10, 6))

crop_profit.plot(
    kind="bar",
    edgecolor="black",
    linewidth=1.5
)

plt.title("Average Profit by Crop")
plt.xlabel("Crop")
plt.ylabel("Average Profit (INR)")

plt.xticks(rotation=45, ha="right")

plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

plt.savefig(
    "crop_profit_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------------
# STEP 43: Crop-wise Yield Comparison
# -----------------------------------

crop_yield = df.groupby("Crop")[
    "Yield_Tonnes_Ha"
].mean().sort_values(ascending=False)

print("\n========== CROP-WISE AVERAGE YIELD ==========")
print(crop_yield.round(2))

plt.figure(figsize=(10, 6))

crop_yield.plot(
    kind="bar",
    edgecolor="black",
    linewidth=1.5
)

plt.title("Average Yield by Crop")
plt.xlabel("Crop")
plt.ylabel("Yield (Tonnes/Ha)")

plt.xticks(rotation=45, ha="right")

plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

plt.savefig(
    "crop_yield_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------------
# STEP 44: Season-wise Average Rainfall
# -----------------------------------

season_rainfall = df.groupby("Season")[
    "Rainfall_mm"
].mean()

print("\n========== SEASON-WISE RAINFALL ==========")
print(season_rainfall.round(2))

plt.figure(figsize=(8, 5))

season_rainfall.plot(
    kind="bar",
    edgecolor="black",
    linewidth=1.5
)

plt.title("Average Rainfall by Season")
plt.xlabel("Season")
plt.ylabel("Rainfall (mm)")

plt.xticks(rotation=0)

plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

plt.savefig(
    "season_vs_rainfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------------
# STEP 45: Season-wise Average Temperature
# -----------------------------------

Avg_temperature = df.groupby("Season")[
    "Avg_Temperature_C"
].mean()

print("\n========== SEASON-WISE TEMPERATURE ==========")
print(Avg_temperature.round(2))

plt.figure(figsize=(8, 5))

Avg_temperature.plot(
    kind="bar",
    edgecolor="black",
    linewidth=1.5
)

plt.title("Average Temperature by Season")
plt.xlabel("Season")
plt.ylabel("Temperature (°C)")

plt.xticks(rotation=0)

plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

plt.savefig(
    "season_vs_temperature.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------------
# STEP 46: Temperature vs Yield
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Avg_Temperature_C"],
    df["Yield_Tonnes_Ha"],
    alpha=0.6,
    edgecolors="black",
    linewidths=0.5
)

plt.title("Temperature vs Agricultural Yield")
plt.xlabel("Temperature (°C)")
plt.ylabel("Yield (Tonnes/Ha)")

plt.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

plt.savefig(
    "temperature_vs_yield.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------------
# STEP 47: Temperature vs Profit
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Avg_Temperature_C"],
    df["Profit_INR"],
    alpha=0.6,
    edgecolors="black",
    linewidths=0.5
)

plt.title("Temperature vs Agricultural Profit")
plt.xlabel("Temperature (°C)")
plt.ylabel("Profit (INR)")

plt.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.7
)

plt.tight_layout()

plt.savefig(
    "temperature_vs_profit.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------------
# STEP 48: Factors Correlated with Yield
# -----------------------------------

yield_correlation = correlation[
    "Yield_Tonnes_Ha"
].sort_values(ascending=False)

print("\n========== CORRELATION WITH YIELD ==========")
print(yield_correlation.round(3))

yield_correlation.to_csv(
    "yield_correlations.csv"
)

print("\nYield correlations saved successfully!")

# -----------------------------------
# STEP 49: Factors Correlated with Profit
# -----------------------------------

profit_correlation = correlation[
    "Profit_INR"
].sort_values(ascending=False)

print("\n========== CORRELATION WITH PROFIT ==========")
print(profit_correlation.round(3))

profit_correlation.to_csv(
    "profit_correlations.csv"
)

print("\nProfit correlations saved successfully!")

# -----------------------------------
# STEP 50: Complete Project Summary
# -----------------------------------

print("\n")
print("=" * 55)
print("     SEASONAL AGRICULTURE PERFORMANCE ANALYSIS")
print("=" * 55)

print("\nDataset Information:")
print("Total Records:", len(df))
print("Total Columns:", len(df.columns))

print("\nSeasons:")
print(df["Season"].unique())

print("\nCrops:")
print(df["Crop"].nunique())

print("\nHighest Average Yield Season:")
print(
    season_insights["Yield_Tonnes_Ha"].idxmax()
)

print("\nHighest Average Profit Season:")
print(
    season_insights["Profit_INR"].idxmax()
)

print("\nHighest Water Efficiency Season:")
print(
    season_insights[
        "Water_Efficiency_t_per_1000m3"
    ].idxmax()
)

print("\nHighest Average Yield Crop:")
print(
    crop_insights["Yield_Tonnes_Ha"].idxmax()
)

print("\nHighest Average Profit Crop:")
print(
    crop_insights["Profit_INR"].idxmax()
)

print("\nHighest Water Efficiency Crop:")
print(
    crop_insights[
        "Water_Efficiency_t_per_1000m3"
    ].idxmax()
)

print("\nAnalysis Completed Successfully!")
print("=" * 55)

