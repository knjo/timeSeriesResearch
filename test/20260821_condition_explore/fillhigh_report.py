import pickle
import numpy as np, pandas as pd
S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
df = pd.DataFrame(pickle.load(open(f'{S}/fillhigh_daily.pkl', 'rb')))
df['yr'] = df.Date.str[:4]

def pooled(sub):
    zs = sub.s_zs.sum()/sub.n.sum(); zl = sub.s_zl.sum()/sub.n.sum()
    return {'n_day': sub.n.sum()/sub.Date.nunique(), 'z_short': zs, 'z_long': zl}

base = {p: pooled(df[(df.tag == 'baseline') & m]) for p, m in
        [('全期', df.tag.notna()), ('2025', df.yr == '2025'), ('2026', df.yr == '2026')]}
bl = df[df.tag == 'baseline'].set_index('Date')
tags = ['FH_pos_chg','FH_pos2_chg','FH_pos8_chg','FH_pos_onset','FH_neg_chg','FH_neg15_chg']
out = []
for tag in tags:
    for per, m in [('全期', df.tag.notna()), ('2025', df.yr == '2025'), ('2026', df.yr == '2026')]:
        sub = df[(df.tag == tag) & m]
        if not len(sub): continue
        s = pooled(sub); b = base[per]; si = sub.set_index('Date')
        dzs = (si.s_zs/si.n - bl.s_zs/bl.n).dropna()
        dzl = (si.s_zl/si.n - bl.s_zl/bl.n).dropna()
        mm = dzs.groupby(dzs.index.str[:6]).mean()
        out.append({'cond': tag, '期間': per, 'n/day': round(s['n_day']),
                    'Δz_short': round(s['z_short']-b['z_short'], 3),
                    't_s': round(dzs.mean()/dzs.std()*np.sqrt(len(dzs)), 2),
                    '正月_s': f"{(mm>0).sum()}/{len(mm)}",
                    'Δz_long': round(s['z_long']-b['z_long'], 3),
                    't_l': round(dzl.mean()/dzl.std()*np.sqrt(len(dzl)), 2)})
print(pd.DataFrame(out).to_string(index=False))

print('\n===== 相似度前篩 vs negFill =====')
for tag in tags:
    sub = df[df.tag == tag]
    if not len(sub): continue
    tick_ov = sub.n_in_nf.sum()/sub.n.sum()
    sd_ov = sub.n_sd_in_nf.sum()/sub.n_sd.sum()
    # disjoint carve
    zs_nf0 = sub.s_zs_nf0.sum()/max(sub.n_nf0.sum(), 1)
    b_all = df[df.tag == 'baseline']
    zs_bl = b_all.s_zs.sum()/b_all.n.sum()
    # proxy corr vs negFill daily mean z (stage1 daily)
    s1 = pd.read_parquet(f'{S}/stage1_daily.parquet')[['Date','s_z','n_z']]
    s1['nf_mz'] = s1.s_z/s1.n_z
    si = sub.set_index('Date'); mz = (si.s_zs/si.n)
    j = pd.DataFrame({'c': mz}).join(s1.set_index('Date')['nf_mz']).dropna()
    print(f"{tag:14s} tick重疊 {tick_ov:5.1%} | stock-day重疊 {sd_ov:5.1%} | "
          f"disjoint段 Δz_short {zs_nf0 - zs_bl:+.3f} (n/day {sub.n_nf0.sum()/sub.Date.nunique():,.0f}) | proxy corr {j.c.corr(j.nf_mz):.2f}")
