import pickle
import numpy as np, pandas as pd
S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
df = pd.DataFrame(pickle.load(open(f'{S}/fh2_daily.pkl', 'rb')))
df['yr'] = df.Date.str[:4]
bl = df[df.tag == 'baseline'].set_index('Date')
out = []
for tag in ['FH2_pure','FH2_flow','FH2_r6','FH2_big','FH2_lowpos']:
    for per, m in [('全期', df.tag.notna()), ('2025', df.yr == '2025'), ('2026', df.yr == '2026')]:
        sub = df[(df.tag == tag) & m]
        if not len(sub) or sub.n.sum() == 0: continue
        b = df[(df.tag == 'baseline') & m]
        zt = sub.s_zs.sum()/sub.n.sum() - b.s_zs.sum()/b.n.sum()
        si = sub.set_index('Date')
        dzs = (si.s_zs/si.n - bl.s_zs/bl.n).dropna()
        mm = dzs.groupby(dzs.index.str[:6]).mean()
        out.append({'cond': tag, '期間': per, 'n/day': round(sub.n.sum()/sub.Date.nunique()),
                    'Δz_short': round(zt, 3), 't': round(dzs.mean()/dzs.std()*np.sqrt(len(dzs)), 2),
                    '正月': f'{(mm>0).sum()}/{len(mm)}'})
print(pd.DataFrame(out).to_string(index=False))
print('\n===== 相似度 vs negFill =====')
for tag in ['FH2_pure','FH2_flow','FH2_r6','FH2_big','FH2_lowpos']:
    sub = df[df.tag == tag]
    if not len(sub) or sub.n.sum() == 0: continue
    b_all = df[df.tag == 'baseline']
    print(f"{tag:10s} tick重疊 {sub.n_in_nf.sum()/sub.n.sum():5.1%} | stock-day重疊 {sub.n_sd_in_nf.sum()/sub.n_sd.sum():5.1%} | "
          f"disjoint Δz {sub.s_zs_nf0.sum()/max(sub.n_nf0.sum(),1) - b_all.s_zs.sum()/b_all.n.sum():+.3f} "
          f"(n/day {sub.n_nf0.sum()/sub.Date.nunique():,.0f})")
