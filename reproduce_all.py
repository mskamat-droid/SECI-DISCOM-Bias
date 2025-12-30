import os
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import kendalltau
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
os.makedirs('replication_outputs', exist_ok=True)

# --- 1. Robust Data Loading ---
def load_clean_data(filepath, expected_cols=None):
    with open(filepath, 'r') as f:
        # Filter: skip comments (#) and empty lines
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    if 'seci_rankings' in filepath:
        # Specific fix for the comma in 'Access, affordability & reliability'
        data = [row.split(',') for row in lines[1:]]
        cols = ['Rank', 'States/UTs', "DISCOM's Performance", 'Access, affordability & reliability', 
                'Clean Energy Initiatives', 'Energy Efficiency', 'Env Sustainability', 'New Initiatives', 'SECI score']
        df = pd.DataFrame(data, columns=cols)
    else:
        data = [row.split(',') for row in lines[1:]]
        cols = lines[0].split(',')
        df = pd.DataFrame(data, columns=cols)
        
    # Convert to numeric and clean whitespace
    for col in df.columns:
        df[col] = df[col].str.strip() if hasattr(df[col], 'str') else df[col]
        if col not in ['State', 'States/UTs', 'GGI_Group']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Load files (handling local extensions)
seci_df = load_clean_data('seci_rankings.csv.txt')
socio_df = load_clean_data('clean_socio_economic_data.csv')
nfhs_df = load_clean_data('nfhs5_human_capital_proxies.csv.txt')
ggi_df = load_clean_data('ggi_proxy.csv.txt')

# Consistent state merge
socio_merged = socio_df.merge(seci_df[['States/UTs', 'SECI score']], left_on='State', right_on='States/UTs').drop('States/UTs', axis=1)

# --- 2. Simulations and Fragility Audit ---
params = ["DISCOM's Performance", 'Access, affordability & reliability', 'Clean Energy Initiatives',
          'Energy Efficiency', 'Env Sustainability', 'New Initiatives']

def simulate_scores(df, discom_weight):
    excess = 40 - discom_weight
    redist = excess / 3
    new_weights = [discom_weight, 15, 15 + redist, 6 + redist, 12 + redist, 12]
    valid_df = df.dropna(subset=params).copy()
    new_scores = np.dot(valid_df[params].values, np.array(new_weights) / 100)
    valid_df['Simulated Score'] = new_scores
    valid_df = valid_df.sort_values('Simulated Score', ascending=False).reset_index(drop=True)
    valid_df['Simulated Rank'] = valid_df.index + 1
    return valid_df

weights = [40, 30, 20, 10]
sim_results = {w: simulate_scores(seci_df, w) for w in weights}

# Save Table A3
a3_data = pd.DataFrame()
for w in weights:
    temp = sim_results[w][['States/UTs', 'Simulated Rank', 'Simulated Score']]
    temp.columns = ['State', f'Rank_{w}%', f'Score_{w}%']
    if a3_data.empty: a3_data = temp
    else: a3_data = a3_data.merge(temp, on='State')
a3_data.to_csv('replication_outputs/table_a3.csv', index=False)

# Kendall's Tau (Aligned)
baseline_ranks = sim_results[40].set_index('States/UTs')['Simulated Rank']
taus = {}
for w in [30, 20, 10]:
    sim_ranks = sim_results[w].set_index('States/UTs')['Simulated Rank']
    tau, _ = kendalltau(baseline_ranks.sort_index(), sim_ranks.sort_index())
    taus[w] = tau

# Volatility Heatmap
a5_data = sim_results[40][['States/UTs', 'Simulated Rank']].rename(columns={'Simulated Rank': 'Rank_Baseline'})
a5_data = a5_data.merge(sim_results[10][['States/UTs', 'Simulated Rank']], on='States/UTs').rename(columns={'Simulated Rank': 'Rank_10%'})
a5_data['Shift'] = a5_data['Rank_10%'] - a5_data['Rank_Baseline']
pivot_heatmap = a5_data.set_index('States/UTs')[['Shift']].sort_values('Shift')
plt.figure(figsize=(10, 10))
sns.heatmap(pivot_heatmap, cmap='coolwarm', annot=True, center=0)
plt.title('Rank Volatility Heatmap (Weighting Sensitivity)')
plt.savefig('replication_outputs/volatility_heatmap.png')

# --- 3. OLS and Mediation ---
socio_merged = socio_merged.dropna()
X = add_constant(socio_merged[['Per_Capita_Income_thou', 'Literacy_pct', 'Poverty_pct', 'Industrial_Share_pct', 'Forest_pct']])
y = socio_merged['SECI score']
model = OLS(y, X).fit()
pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0].to_csv('replication_outputs/ols_results.csv')

# Mediation Logic
reg_a = LinearRegression().fit(socio_merged[['Forest_pct']], socio_merged['Literacy_pct'])
reg_b = LinearRegression().fit(socio_merged[['Forest_pct', 'Literacy_pct']], socio_merged['SECI score'])
total_eff = LinearRegression().fit(socio_merged[['Forest_pct']], socio_merged['SECI score']).coef_[0]
indirect = reg_a.coef_[0] * reg_b.coef_[1]
indirect_pct = (indirect / total_eff) * 100

# Save Mediation Figure
fig, ax = plt.subplots(figsize=(8, 4))
ax.text(0.5, 0.7, f"Indirect Effect via Literacy: {indirect_pct:.1f}%", ha='center', fontweight='bold')
ax.text(0.1, 0.5, 'Forest Cover', bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax.text(0.5, 0.5, 'Literacy', bbox=dict(boxstyle='round', facecolor='skyblue'))
ax.text(0.9, 0.5, 'SECI Score', bbox=dict(boxstyle='round', facecolor='orange'))
ax.axis('off')
plt.savefig('replication_outputs/mediation_path.png')

# --- 4. Diagnostics & Robustness ---
vif_df = pd.DataFrame({'feature': X.columns, 'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
vif_df.to_csv('replication_outputs/vif_results.csv', index=False)

with open('replication_outputs/robustness.txt', 'w') as f:
    nfhs_m = socio_merged.merge(nfhs_df, on='State').dropna()
    m_nfhs = OLS(nfhs_m['SECI score'], add_constant(nfhs_m[['Mean_Years_Schooling', 'Households_Internet_pct']])).fit()
    f.write("NFHS-5 Model Summary:\n" + str(m_nfhs.summary()))

print("Replication Complete. Key results:")
print(f"Kendall Taus: {taus}")
print(f"Literacy Beta: {model.params['Literacy_pct']:.4f} (p={model.pvalues['Literacy_pct']:.4f})")
