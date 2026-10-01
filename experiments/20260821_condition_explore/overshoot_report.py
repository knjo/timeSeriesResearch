import pickle
import numpy as np, pandas as pd
S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
df = pd.DataFrame(pickle.load(open(f'{S}/overshoot_daily.pkl', 'rb')))
df['yr'] = df.Date.str[:4]
df['ym'] = df.Date.str[:6]

def pooled(sub):
    zs = sub.s_zs.sum()/sub.n.sum(); zl = sub.s_zl.sum()/sub.n.sum()
    return {'n_day': sub.n.sum()/sub.Date.nunique(),
            'z_short': zs, 'zs_std': np.sqrt(sub.s_zs2.sum()/sub.n.sum() - zs**2),
            'z_long': zl, 'zl_std': np.sqrt(sub.s_zl2.sum()/sub.n.sum() - zl**2),
            'raw_s': sub.s_raws.sum()/sub.n.sum(), 'raw_l': sub.s_rawl.sum()/sub.n.sum()}

base = {p: pooled(df[(df.tag == 'baseline') & m]) for p, m in
        [('全期', df.tag.notna()), ('2025', df.yr == '2025'), ('2026', df.yr == '2026')]}
print('===== baseline =====')
print(pd.DataFrame(base).T.round(3).to_string())

print('\n===== conditions: lift vs baseline（Δz>0 = 該方向有利）=====')
out = []
for tag in ['OS_any','OS_buy','OS_sell','OS_buy_cd60','OS_sell_cd60']:
    for per, m in [('全期', df.tag.notna()), ('2025', df.yr == '2025'), ('2026', df.yr == '2026')]:
        sub = df[(df.tag == tag) & m]
        if not len(sub): continue
        s = pooled(sub); b = base[per]
        # daily paired t for both sides
        mzs = sub.set_index('Date').s_zs / sub.set_index('Date').n
        mzl = sub.set_index('Date').s_zl / sub.set_index('Date').n
        bl = df[(df.tag == 'baseline') & m].set_index('Date')
        dzs = (mzs - bl.s_zs/bl.n).dropna(); dzl = (mzl - bl.s_zl/bl.n).dropna()
        out.append({'cond': tag, '期間': per, 'n/day': round(s['n_day']),
                    'Δz_short': s['z_short']-b['z_short'], 't_s': dzs.mean()/dzs.std()*np.sqrt(len(dzs)),
                    'Δz_long': s['z_long']-b['z_long'], 't_l': dzl.mean()/dzl.std()*np.sqrt(len(dzl)),
                    'zs_std': s['zs_std'], 'zl_std': s['zl_std'],
                    'Δraw_s': s['raw_s']-b['raw_s'], 'Δraw_l': s['raw_l']-b['raw_l']})
t = pd.DataFrame(out)
print(t.round(3).to_string(index=False))

# monthly consistency for the most promising
print('\n===== 月度 Δz（挑顯著者看穩定性）=====')
for tag, col in [('OS_buy','zl'), ('OS_sell','zl'), ('OS_buy','zs'), ('OS_sell','zs')]:
    sub = df[df.tag == tag].set_index('Date'); bl = df[df.tag == 'baseline'].set_index('Date')
    dz = (sub[f's_{col}']/sub.n - bl[f's_{col}']/bl.n).dropna()
    mm = dz.groupby(dz.index.str[:6]).mean()
    pos = (mm > 0).sum()
    print(f"{tag}|{'long' if col=='zl' else 'short'}: 正月 {pos}/{len(mm)} | " +
          ' '.join(f"{k}:{v:+.2f}" for k, v in mm.items() if k in ['202503','202506','202509','202512','202603','202606','202608']))
