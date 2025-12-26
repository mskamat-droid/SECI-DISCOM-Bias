import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from sklearn.cluster import KMeans
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('Raw_data.csv')

# Original weights (exact approximation from report: DISCOM 40%, non-DISCOM 60% split)
weights = {
    'DISCOM': 0.40,
    'Access_Affordability_Reliability': 0.12,
    'Clean_Energy_Initiatives': 0.15,
    'Energy_Efficiency': 0.12,
    'Environmental_Sustainability': 0.12,
    'New_Initiatives': 0.09
}

# Verify original score
df['Calculated_Original'] = df.apply(lambda row: sum(row[col] * weights[col] for col in weights), axis=1)

# Simulations: 30%, 20%, 10% DISCOM
for discom_w in [0.30, 0.20, 0.10]:
    excess = 0.40 - discom_w
    boost = excess / 5
    sim_weights = {k: v + boost if k != 'DISCOM' else discom_w for k, v in weights.items()}
    df[f'SECI_{int(discom_w*100)}'] = df.apply(lambda row: sum(row[p] * sim_weights[p] for p in sim_weights), axis=1)

# Ranks
df['Rank_Original'] = df['Original_SECI_Score'].rank(ascending=False, method='min')
for col in ['SECI_30', 'SECI_20', 'SECI_10']:
    df[f'Rank_{col}'] = df[col].rank(ascending=False, method='min')

# Kendall Tau
print("Kendall Tau (vs Original):")
for col in ['SECI_30', 'SECI_20', 'SECI_10']:
    tau, _ = kendalltau(df['Rank_Original'], df[f'Rank_{col}'])
    print(f"{col}: {tau:.3f}")

# Volatility for K-means (rank range)
rank_cols = ['Rank_Original', 'Rank_SECI_30', 'Rank_SECI_20', 'Rank_SECI_10']
df['Rank_Range'] = df[rank_cols].max(axis=1) - df[rank_cols].min(axis=1)
kmeans = KMeans(n_clusters=3, random_state=42).fit(df[['Rank_Range']])
df['Volatility_Cluster'] = kmeans.labels_

# OLS Models
socio_cols = ['Per_Capita_Income_2022_Rs_lakh', 'Literacy_2011_percent', 'Poverty_Headcount_2023_percent',
              'Industrial_Share_2022_percent_GSDP', 'Forest_Cover_2023_percent']
X = sm.add_constant(df[socio_cols])

# Original
model_orig = sm.OLS(df['Original_SECI_Score'], X).fit()
print("\nOLS Original SECI:")
print(model_orig.summary())

# Recalculated (10%)
model_recalc = sm.OLS(df['SECI_10'], X).fit()
print("\nOLS Recalculated (10% DISCOM):")
print(model_recalc.summary())

# Visual 1: Rank shifts bar chart (top 10 original)
top10 = df.nlargest(10, 'Original_SECI_Score')[['State', 'Rank_Original', 'Rank_SECI_10']]
top10 = top10.melt(id_vars='State', value_vars=['Rank_Original', 'Rank_SECI_10'], var_name='Scenario', value_name='Rank')
plt.figure(figsize=(12,6))
sns.barplot(data=top10, x='State', y='Rank', hue='Scenario')
plt.title('Top 10 States: Rank Shift (Original vs 10% DISCOM)')
plt.ylabel('Rank (Lower = Better)')
plt.xticks(rotation=45)
plt.savefig('rank_shifts_top10.png')
plt.close()

# Visual 2: Volatility clusters
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Original_SECI_Score', y='Rank_Range', hue='Volatility_Cluster', palette='deep')
plt.title('State Volatility by Original Score and Rank Range')
plt.savefig('volatility_clusters.png')

print("Replication complete. Visuals saved: rank_shifts_top10.png, volatility_clusters.png")
