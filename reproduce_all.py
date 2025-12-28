import pandas as pd
import statsmodels.api as sm
from scipy.stats import kendalltau
from sklearn.cluster import KMeans

# 1. LOAD DATA
seci = pd.read_csv('data/seci_rankings.csv')
socio = pd.read_csv('data/socio_economic_data.csv')

# 2. RANK SENSITIVITY (Table 1 & 3)
def simulate(df, w_discom):
    # Logic: Redistribute reduced weight equally to other 5 parameters
    inc = (40 - w_discom) / 5
    score = (df['DISCOM'] * (w_discom/100) + 
             df['AAR'] * ((15+inc)/100) + 
             df['CEI'] * ((15+inc)/100) + 
             df['EE'] * ((6+inc)/100) + 
             df['ES'] * ((12+inc)/100) + 
             df['NI'] * ((12+inc)/100))
    return score

seci['Score_10pct'] = simulate(seci, 10)
tau, _ = kendalltau(seci['Rank'], seci['Score_10pct'].rank(ascending=False))
print(f"Verified Kendall's Tau (10%): {tau:.3f}")

# 3. DRIVERS REGRESSION (Table 5 - The 'Honesty' Model)
merged = pd.merge(seci, socio, on='State') # Resulting N=27
X = sm.add_constant(merged[['Per_Capita_Income_thou', 'Literacy_pct', 'Poverty_pct', 'Industrial_Share_pct', 'Forest_pct']])
model = sm.OLS(merged['Score'], X).fit()

print("\n--- REPRODUCED TABLE 5 ---")
print(model.summary().tables[1])
