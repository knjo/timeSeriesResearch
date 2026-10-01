import pandas as pd, numpy as np
S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
p = pd.read_csv(f'{S}/feature_ic_daily.csv')
p['ym'] = p.Date.astype(str).str[:6]
feats = [c for c in p.columns if c not in ('Date','y','ym')]

def half(df):
    return df.groupby('ym')[feats].mean()

for target in ['z','raw']:
    sub = p[p.y == target]
    # period means
    per = {
        '2025H2(0909-12)': sub[(sub.ym >= '202509') & (sub.ym <= '202512')],
        '2026_JanMay': sub[(sub.ym >= '202601') & (sub.ym <= '202605')],
        '2026_JunAug': sub[(sub.ym >= '202606') & (sub.ym <= '202608')],
    }
    rows = {}
    for k, d in per.items():
        m = d[feats].mean()
        rows[k] = m
    t = pd.DataFrame(rows)
    t['absH2'] = t['2025H2(0909-12)'].abs()
    t = t.sort_values('absH2', ascending=False).drop(columns='absH2')
    print(f'\n================ per-day Spearman IC vs [{target}] — period means ================')
    print(t.round(4).to_string())
    # stability: how many of top-10 2025H2 features keep sign & >1/3 magnitude in 2026JunAug
    top10 = t.index[:10]
    keep = [(f, t.loc[f].iloc[0], t.loc[f].iloc[2]) for f in top10]
    n_ok = sum(1 for f, a, b in keep if np.sign(a) == np.sign(b) and abs(b) >= abs(a) / 3)
    print(f'top10 2025H2 features surviving in 2026JunAug (same sign & >=1/3 magnitude): {n_ok}/10')
