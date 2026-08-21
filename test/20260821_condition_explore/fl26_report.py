import pickle
import numpy as np, pandas as pd
S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
df = pd.DataFrame(pickle.load(open(f'{S}/filllow26_daily.pkl', 'rb')))
df['ym'] = df.Date.str[:6]

def pooled(sub):
    zs = sub.s_zs.sum()/sub.n.sum(); zl = sub.s_zl.sum()/sub.n.sum()
    return {'n_day': sub.n.sum()/sub.Date.nunique(), 'z_short': zs, 'z_long': zl}

b = pooled(df[df.tag == 'baseline'])
bl = df[df.tag == 'baseline'].set_index('Date')
print(f"2026 加密（{df.Date.nunique()} 天）baseline z_short {b['z_short']:+.3f} | z_long {b['z_long']:+.3f}\n")
out = []
for tag in ['FL_neg_chg','FL_neg2_chg','FL_neg8_chg','FL_onset','FL_pos_chg','FL_pos8_chg']:
    sub = df[df.tag == tag]
    if not len(sub): continue
    s = pooled(sub); si = sub.set_index('Date')
    dzs = (si.s_zs/si.n - bl.s_zs/bl.n).dropna()
    dzl = (si.s_zl/si.n - bl.s_zl/bl.n).dropna()
    mm = dzs.groupby(dzs.index.str[:6]).mean()
    out.append({'cond': tag, 'n/day': round(s['n_day']),
                'Δz_short': s['z_short']-b['z_short'], 't_s': dzs.mean()/dzs.std()*np.sqrt(len(dzs)),
                '正月_s': f"{(mm>0).sum()}/{len(mm)}",
                'Δz_long': s['z_long']-b['z_long'], 't_l': dzl.mean()/dzl.std()*np.sqrt(len(dzl)),
                '月度s': ' '.join(f'{k[-2:]}:{v:+.2f}' for k, v in mm.items())})
print(pd.DataFrame(out).round(3).to_string(index=False))
