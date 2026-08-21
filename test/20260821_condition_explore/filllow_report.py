import pickle
import numpy as np, pandas as pd
S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
df = pd.DataFrame(pickle.load(open(f'{S}/filllow_daily.pkl', 'rb')))
df['yr'] = df.Date.str[:4]

def pooled(sub):
    zs = sub.s_zs.sum()/sub.n.sum(); zl = sub.s_zl.sum()/sub.n.sum()
    return {'n_day': sub.n.sum()/sub.Date.nunique(), 'z_short': zs, 'z_long': zl,
            'zs_std': np.sqrt(sub.s_zs2.sum()/sub.n.sum() - zs**2),
            'zl_std': np.sqrt(sub.s_zl2.sum()/sub.n.sum() - zl**2)}

base = {p: pooled(df[(df.tag == 'baseline') & m]) for p, m in
        [('全期', df.tag.notna()), ('2025', df.yr == '2025'), ('2026', df.yr == '2026')]}
out = []
tags = ['FL_neg_chg','FL_neg2_chg','FL_neg8_chg','FL_onset','FL_pos_chg','FL_pos8_chg']
for tag in tags:
    for per, m in [('全期', df.tag.notna()), ('2025', df.yr == '2025'), ('2026', df.yr == '2026')]:
        sub = df[(df.tag == tag) & m]
        if not len(sub): continue
        s = pooled(sub); b = base[per]
        bl = df[(df.tag == 'baseline') & m].set_index('Date')
        si = sub.set_index('Date')
        dzs = (si.s_zs/si.n - bl.s_zs/bl.n).dropna()
        dzl = (si.s_zl/si.n - bl.s_zl/bl.n).dropna()
        mm_s = dzs.groupby(dzs.index.str[:6]).mean(); mm_l = dzl.groupby(dzl.index.str[:6]).mean()
        out.append({'cond': tag, '期間': per, 'n/day': round(s['n_day']),
                    'Δz_short': s['z_short']-b['z_short'], 't_s': dzs.mean()/dzs.std()*np.sqrt(len(dzs)),
                    '正月_s': f"{(mm_s>0).sum()}/{len(mm_s)}",
                    'Δz_long': s['z_long']-b['z_long'], 't_l': dzl.mean()/dzl.std()*np.sqrt(len(dzl)),
                    '正月_l': f"{(mm_l>0).sum()}/{len(mm_l)}",
                    'zs_std': s['zs_std']})
t = pd.DataFrame(out)
print('baseline z_short:', {k: round(v['z_short'], 3) for k, v in base.items()},
      '| z_long:', {k: round(v['z_long'], 3) for k, v in base.items()})
print()
print(t.round(3).to_string(index=False))
# selectivity
piv = df[df.tag.isin(['FL_neg_chg','baseline'])].pivot_table(index='Date', columns='tag', values=['n','n_pool'])
sel = piv[('n','FL_neg_chg')] / piv[('n_pool','baseline')] * 100
print(f"\nFL_neg_chg 選擇率 median {sel.median():.2f}%")
