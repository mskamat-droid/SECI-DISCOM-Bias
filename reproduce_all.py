# enhanced_replicate_seci_paper.py
# Enhanced full replication package with CSV exports, interactive plots, and robustness checks
# 100% self-contained and transparent. Run as-is (Python 3.8+).

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, pearsonr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans
import os

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    print("Plotly not installed. Install with: pip install plotly")
    exit()

print("=== Enhanced Replication Package Started ===")

# Create output directory
os.makedirs("replication_outputs", exist_ok=True)

# Load data (embedded CSVs)
seci_csv = """Rank,State,DISCOM,AAR,CEI,EE,ES,NI,SECI_score_original
1,Goa,63.4,59.6,62.4,16.6,43.7,12.4,51.4
2,Gujarat,72.7,52.4,39.2,40.1,35.1,5.5,50.1
3,Kerala,64.4,67.3,21.5,58.0,46.9,7.7,49.1
4,Punjab,77.1,46.8,26.1,35.1,37.0,2.3,48.6
5,Haryana,69.8,53.6,42.9,11.7,33.4,6.9,47.9
6,Uttarakhand,61.9,55.3,18.5,50.5,48.7,14.7,46.5
7,Maharashtra,57.7,51.2,34.0,75.7,36.2,10.4,46.0
8,Himachal Pradesh,57.0,56.3,14.3,20.1,52.1,38.1,45.4
9,Tripura,57.3,33.1,22.9,31.7,39.6,58.7,45.0
10,Karnataka,56.8,45.5,27.0,57.2,41.7,14.5,43.8
11,Tamil Nadu,57.3,46.3,21.7,85.4,39.2,4.0,43.4
12,Assam,67.3,38.3,4.3,39.0,39.9,17.6,42.6
13,Telangana,55.1,60.4,18.0,64.7,34.6,0.4,41.9
14,Andhra Pradesh,65.1,42.6,16.9,40.0,35.0,0.0,41.6
15,Uttar Pradesh,59.9,37.8,12.6,42.0,30.9,27.4,41.0
16,West Bengal,55.3,52.0,8.5,27.7,40.9,9.0,38.9
17,Bihar,61.3,45.0,4.9,22.8,33.7,7.6,38.3
18,Odisha,59.0,57.4,4.8,21.8,22.6,0.9,37.1
19,Manipur,57.6,34.1,4.7,22.1,41.3,7.3,36.0
20,Mizoram,51.7,39.3,18.9,29.7,38.2,1.1,35.9
21,Rajasthan,49.2,42.9,15.5,44.0,31.4,4.8,35.4
22,Jharkhand,58.3,46.5,2.9,17.2,19.0,9.3,35.2
23,Sikkim,43.2,37.6,13.8,33.3,52.2,0.6,33.3
24,Madhya Pradesh,53.7,42.7,6.2,8.3,24.1,3.3,32.6
25,Chhattisgarh,58.4,45.4,2.1,0.0,5.8,4.2,31.7
26,Meghalaya,47.9,30.9,1.9,4.0,39.8,2.8,29.4
27,Nagaland,35.9,32.9,12.2,26.4,40.0,3.4,27.9
28,Arunachal Pradesh,31.1,43.2,5.8,19.8,49.0,1.1,27.0"""

df_seci = pd.read_csv(pd.compat.StringIO(seci_csv)).set_index("State")

# Socio-economic and NFHS data (same as before)
socio_csv = """State,Per_Capita_Income_thou,Literacy_pct,Poverty_pct,Industrial_Share_pct,Forest_pct
Goa,431,92,3.8,25,60.2
Gujarat,220,82,12.7,35,9.7
Kerala,223,97,0.6,28,55.0
Punjab,164,82,6.2,25,3.7
Haryana,236,80,7.1,30,3.6
Uttarakhand,182,88,10.5,28,45.4
Maharashtra,190,85,14.0,30,16.7
Himachal Pradesh,180,90,7.6,25,27.0
Tripura,110,90,12.4,20,73.8
Karnataka,220,82,10.0,28,20.0
Tamil Nadu,220,85,4.9,32,20.3
Telangana,240,75,8.0,26,21.0
Andhra Pradesh,170,70,12.3,24,23.0
Uttar Pradesh,70,75,22.9,22,6.1
West Bengal,130,82,10.0,22,16.9
Assam,100,85,20.0,25,35.3
Odisha,120,80,25.0,25,33.5
Bihar,44,68,33.8,20,7.8
Rajasthan,120,75,15.0,25,9.6
Madhya Pradesh,110,75,22.0,25,25.1
Jharkhand,80,75,28.8,30,29.6
Chhattisgarh,110,78,25.0,35,44.2
Sikkim,413,85,3.0,20,47.1
Meghalaya,90,80,20.0,18,76.0
Nagaland,100,85,15.0,15,75.3
Mizoram,150,95,5.0,15,84.5
Arunachal Pradesh,120,75,18.0,20,79.3"""
df_socio = pd.read_csv(pd.compat.StringIO(socio_csv)).set_index("State")

nfhs_csv = """State,Mean_Years_Schooling,Women_10plus_Yrs_Schooling_pct,Households_Internet_pct
Goa,9.6,68.3,68.8
Gujarat,8.2,47.8,52.4
Kerala,10.7,71.9,69.8
Punjab,8.1,52.8,58.3
Haryana,8.0,48.4,62.1
Uttarakhand,8.8,55.8,55.2
Maharashtra,8.6,55.3,58.9
Himachal Pradesh,9.3,62.6,65.4
Tripura,8.4,42.1,38.7
Karnataka,8.0,49.2,54.6
Tamil Nadu,8.6,55.7,59.3
Telangana,7.8,46.9,56.8
Andhra Pradesh,7.1,38.6,42.3
Uttar Pradesh,7.2,42.3,44.1
West Bengal,7.8,46.8,42.9
Assam,7.8,45.2,38.5
Odisha,7.6,42.8,39.2
Bihar,6.5,33.8,34.7
Rajasthan,6.9,36.2,48.3
Madhya Pradesh,7.0,38.9,41.6
Jharkhand,6.8,37.4,36.8
Chhattisgarh,7.3,40.1,37.9
Sikkim,8.9,58.7,62.5
Meghalaya,7.5,45.6,35.4
Nagaland,8.7,52.3,42.1
Mizoram,9.4,65.8,48.9
Arunachal Pradesh,7.2,40.8,38.2"""
df_nfhs = pd.read_csv(pd.compat.StringIO(nfhs_csv)).set_index("State")

# Weight calculation function
def calc_score(row, weights):
    return sum(row[col] * weights[col] for col in weights)

# Baseline
weights_base = {'DISCOM': 0.40, 'AAR': 0.15, 'CEI': 0.15, 'EE': 0.15, 'ES': 0.08, 'NI': 0.07}
df_seci['score_40'] = df_seci.apply(calc_score, axis=1, weights=weights_base)
df_seci['rank_40'] = df_seci['score_40'].rank(ascending=False).astype(int)

# Focused Net-Zero scenarios (Appendix A4)
scenarios = []
weights_table = []
for discom_pct in [40, 30, 20, 10]:
    discom_w = discom_pct / 100
    excess = 0.40 - discom_w
    boost = excess / 3 if discom_pct < 40 else 0
    weights = {
        'DISCOM': discom_w,
        'AAR': 0.15,
        'CEI': 0.15 + boost,
        'EE': 0.15 + boost,
        'ES': 0.08 + boost,
        'NI': 0.07
    }
    score_col = f'score_{discom_pct}'
    rank_col = f'rank_{discom_pct}'
    df_seci[score_col] = df_seci.apply(calc_score, axis=1, weights=weights)
    df_seci[rank_col] = df_seci[score_col].rank(ascending=False).astype(int)
    scenarios.append((score_col, rank_col))
    weights_table.append({'DISCOM_pct': discom_pct, **weights})

# Export Appendix tables
# A3: Scores & Ranks
a3_cols = ['score_40', 'rank_40', 'score_30', 'rank_30', 'score_20', 'rank_20', 'score_10', 'rank_10']
df_a3 = df_seci[a3_cols].round(2)
df_a3.to_csv("replication_outputs/appendix_a3_simulated_scores_ranks.csv")
print("\nExported Appendix A3: replication_outputs/appendix_a3_simulated_scores_ranks.csv")

# A4: Weights
pd.DataFrame(weights_table).to_csv("replication_outputs/appendix_a4_weights.csv", index=False)
print("Exported Appendix A4: replication_outputs/appendix_a4_weights.csv")

# A5: Rank shifts & volatility
df_a5 = df_seci[['rank_40', 'rank_10']].copy()
df_a5['shift_40_to_10'] = df_a5['rank_40'] - df_a5['rank_10']
rank_cols = [f'rank_{p}' for p in [40,30,20,10]]
df_a5['rank_range'] = df_seci[rank_cols].max(axis=1) - df_seci[rank_cols].min(axis=1)
df_a5['rank_std'] = df_seci[rank_cols].std(axis=1)
df_a5.to_csv("replication_outputs/appendix_a5_rank_shifts_volatility.csv")
print("Exported Appendix A5: replication_outputs/appendix_a5_rank_shifts_volatility.csv")

# Example shifts
print("\nKey state rank shifts (40% → 10% DISCOM):")
print(df_a5.loc[['Punjab', 'Gujarat', 'Maharashtra', 'Tamil Nadu', 'Chhattisgarh']])

# Kendall Tau
print("\nKendall Tau vs baseline:")
for p in [30,20,10]:
    tau, _ = kendalltau(df_seci['rank_40'], df_seci[f'rank_{p}'])
    print(f"{p}% DISCOM: {tau:.3f}")

# Socio-economic analysis
df_full = df_seci.join(df_socio).join(df_nfhs)

# Main OLS
X = df_full[['Literacy_pct', 'Forest_pct', 'Per_Capita_Income_thou', 'Poverty_pct', 'Industrial_Share_pct']]
X = sm.add_constant(X)
y = df_full['score_40']
model_main = sm.OLS(y, X.dropna()).fit()
print("\nMain OLS Coefficients:")
print(model_main.params.round(3))

# Robustness: NFHS-5 human capital
X_hc = df_full[['Mean_Years_Schooling', 'Women_10plus_Yrs_Schooling_pct', 'Households_Internet_pct', 'Forest_pct']]
X_hc = sm.add_constant(X_hc)
model_hc = sm.OLS(y, X_hc.dropna()).fit()
print("\nRobustness OLS (NFHS-5 proxies):")
print(model_hc.params.round(3))

# VIF check
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.dropna().values, i) for i in range(len(X.columns))]
print("\nVIF (multicollinearity check):")
print(vif_data)

# Mediation proxy
print("\nForest → Literacy correlation (indirect effect proxy):", pearsonr(df_full['Forest_pct'], df_full['Literacy_pct'])[0].round(3))

# Volatility clustering
df_vol = df_seci[rank_cols].copy()
df_vol['range'] = df_a5['rank_range']
df_vol['std'] = df_a5['rank_std']
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(df_vol[['range', 'std']])
df_vol['cluster'] = kmeans.labels_
print("\nTop high-volatility states:")
print(df_vol.sort_values('std', ascending=False).head(10)[['cluster'] + rank_cols])

# Interactive plots
# 1. SECI vs Literacy & Forest (scatter with hover)
fig1 = px.scatter(df_full.reset_index(), x='Literacy_pct', y='score_40', color='Forest_pct',
                  hover_name='State', title="SECI Score vs Literacy (colored by Forest Cover)")
fig1.write_html("replication_outputs/interactive_seci_vs_literacy_forest.html")

# 2. Rank changes over scenarios
df_ranks_melt = df_seci[rank_cols + ['State']].reset_index().melt(id_vars='State', value_vars=rank_cols, var_name='Scenario', value_name='Rank')
fig2 = px.line(df_ranks_melt, x='Scenario', y='Rank', color='State', title="Rank Trajectories Across DISCOM Weights")
fig2.update_yaxes(autorange="reversed")
fig2.write_html("replication_outputs/interactive_rank_trajectories.html")

# 3. Robustness: Even redistribution scenario (example at 10%)
excess_even = 0.30  # From 40% to 10%
boost_even = excess_even / 5  # Spread to AAR, CEI, EE, ES, NI
weights_even = {'DISCOM': 0.10, 'AAR': 0.15 + boost_even, 'CEI': 0.15 + boost_even,
                'EE': 0.15 + boost_even, 'ES': 0.08 + boost_even, 'NI': 0.07 + boost_even}
df_seci['score_10_even'] = df_seci.apply(calc_score, axis=1, weights=weights_even)
df_seci['rank_10_even'] = df_seci['score_10_even'].rank(ascending=False).astype(int)
tau_even, _ = kendalltau(df_seci['rank_40'], df_seci['rank_10_even'])
print(f"\nRobustness (Even redistribution at 10% DISCOM) Tau: {tau_even:.3f}")

print("\n=== All enhancements complete ===")
print("Outputs saved in replication_outputs/ folder:")
print("- CSVs for Appendices A3–A5")
print("- Interactive HTML plots (open in browser)")
print("Upload this script + outputs folder to GitHub/Zenodo for full transparency.")
